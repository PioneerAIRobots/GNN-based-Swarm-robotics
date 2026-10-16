"""Local motion: artificial potential fields with inter-agent deconfliction.

The allocator says *which* task; this says *how to get there this instant*. It
is deliberately reactive and stateless rather than a full planner:

  * The coordination layer re-decides at 2 Hz, so a long global plan would be
    discarded before it was flown. What is needed is a controller that converts
    "go to that point" into a velocity command every tick, respecting the
    obstacles the LiDAR can see right now.
  * nav2 is available and is the right tool for a single ground robot in a known
    map, but it will not plan for a quadrotor in 3-D or arbitrate between three
    heterogeneous platforms sharing a corridor. The deconfliction term below
    does that in two lines.

Known limitation, stated plainly: potential fields have local minima, and a
deep U-shaped courtyard can trap a ground platform. Two mitigations are in
place === a tangential escape term that fires when the robot is stuck, and the
allocator's own re-tasking, which will eventually hand a cell nobody can reach
to the drone. For the corridor-and-blocks sector here that is sufficient; a
sector with real dead ends would want nav2 under the same command interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .spec import PROFILES, Platform


@dataclass
class NavConfig:
    """Gains and envelope for one platform's local controller."""

    #: Pull toward the goal.
    k_attract: float = 1.0
    #: Push from LiDAR returns.
    k_repel: float = 2.6
    #: Push from teammates, so three robots do not converge on one doorway.
    k_deconflict: float = 3.2
    #: Range at which obstacles start to matter.
    obstacle_influence: float = 4.5
    #: Range at which teammates start to matter.
    agent_influence: float = 5.0
    #: Acceptance radius: inside this the controller reports arrival.
    goal_tolerance: float = 2.0
    #: Decelerate linearly inside this radius so the robot settles.
    slowdown_radius: float = 6.0
    max_speed: float = 3.0
    max_yaw_rate: float = 1.5
    #: Heading error beyond which a ground platform turns before driving.
    turn_in_place_threshold: float = 1.05
    #: Speed below which the robot is considered stuck.
    stuck_speed: float = 0.12
    #: Seconds of being stuck before the escape term engages.
    stuck_timeout: float = 3.0


def config_for(platform: Platform, caution: float = 1.0) -> NavConfig:
    """Per-platform gains, scaled by the GNN's caution output.

    `caution` in [0, 1] comes from the network's caution head: it is the
    learned, risk-aware speed scalar. A low value shortens the speed envelope
    and widens the obstacle influence, which is what "move carefully through a
    contested cell" means in terms this controller can execute.
    """
    profile = PROFILES[platform]
    speed = profile.max_speed * (0.35 + 0.65 * max(0.0, min(1.0, caution)))

    if platform == Platform.DRONE:
        return NavConfig(
            k_attract=1.15,
            # Airborne, so ground clutter is mostly irrelevant; the AGL scan
            # only has to keep it off rooftops and masts.
            k_repel=1.5,
            k_deconflict=1.2,
            obstacle_influence=6.0,
            agent_influence=7.0,
            goal_tolerance=3.0,
            slowdown_radius=9.0,
            max_speed=speed,
            max_yaw_rate=2.0,
            # A quadrotor translates without turning, so never turn in place.
            turn_in_place_threshold=math.pi,
        )
    if platform == Platform.UGV:
        return NavConfig(
            k_attract=1.0,
            k_repel=3.1,
            k_deconflict=3.4,
            obstacle_influence=5.0,
            agent_influence=5.5,
            goal_tolerance=2.0,
            slowdown_radius=6.5,
            max_speed=speed,
            max_yaw_rate=1.4,
            turn_in_place_threshold=0.9,
        )
    return NavConfig(  # quadruped
        k_attract=1.0,
        k_repel=2.4,
        k_deconflict=2.6,
        # Narrow alleys: a wide influence radius would make it refuse to enter.
        obstacle_influence=3.0,
        agent_influence=3.5,
        goal_tolerance=1.4,
        slowdown_radius=4.0,
        max_speed=speed,
        max_yaw_rate=1.2,
        # It can crab sideways, so it tolerates a large heading error.
        turn_in_place_threshold=2.1,
    )


@dataclass
class ScanObstacles:
    """Polar obstacle set from a LaserScan, in the robot's body frame."""

    angles: np.ndarray
    ranges: np.ndarray

    @classmethod
    def from_scan(cls, angle_min: float, angle_increment: float, ranges,
                  range_max: float) -> "ScanObstacles":
        r = np.asarray(ranges, dtype=np.float64)
        a = angle_min + angle_increment * np.arange(r.size)
        # Infs and zeros are "no return", not "obstacle at zero metres".
        valid = np.isfinite(r) & (r > 0.05) & (r < range_max)
        return cls(a[valid], r[valid])

    def __len__(self) -> int:
        return int(self.ranges.size)


@dataclass
class NavState:
    """Per-agent controller memory. Only the stuck detector needs state."""

    #: Seconds since the robot last made real progress toward its goal.
    stuck_time: float = 0.0
    #: Which way round an obstacle to go, +1 left. Held for `escape_hold`
    #: seconds once chosen so the robot commits instead of dithering.
    escape_sign: float = 0.0
    escape_hold: float = 0.0
    #: Best distance-to-goal achieved so far, and when it was achieved.
    best_distance: float = float("inf")


def potential_field_command(
    position: tuple[float, float, float],
    yaw: float,
    goal: tuple[float, float, float],
    cfg: NavConfig,
    obstacles: ScanObstacles | None = None,
    teammates: list[tuple[float, float, float]] = (),
    state: NavState | None = None,
    dt: float = 0.1,
    altitude_goal: float | None = None,
    path_following: bool = True,
) -> tuple[float, float, float, float, bool]:
    """One control step.

    Returns (vx, vy, vz, yaw_rate, arrived). vx/vy are in the body frame,
    which is what the platform drivers expect; vz is a world-frame climb rate
    and is only non-zero for the drone.

    `path_following` says the goal is the next waypoint of a route a global
    planner has already certified as free. That changes what the repulsive
    term is allowed to do === see the projection step below. Set it false to
    use the field as a standalone controller against an unverified goal, which
    is slower and can trap.
    """
    state = state or NavState()

    dx = goal[0] - position[0]
    dy = goal[1] - position[1]
    planar_dist = math.hypot(dx, dy)

    # ============================ attraction ================================
    # World-frame unit vector to the goal, scaled down inside the slowdown ring
    # so the robot settles instead of oscillating across the acceptance radius.
    if planar_dist > 1e-6:
        ax = cfg.k_attract * dx / planar_dist
        ay = cfg.k_attract * dy / planar_dist
    else:
        ax = ay = 0.0
    taper = min(1.0, planar_dist / max(0.1, cfg.slowdown_radius))

    # ============================= repulsion ================================
    # Obstacles arrive in the body frame, so rotate their contribution out to
    # the world frame before summing with the attraction term.
    rx = ry = 0.0
    closest_ahead = float("inf")
    if obstacles is not None and len(obstacles):
        near = obstacles.ranges < cfg.obstacle_influence
        if near.any():
            r = obstacles.ranges[near]
            a = obstacles.angles[near] + yaw
            # Classic 1/d - 1/d0 field: zero at the influence radius, rising
            # steeply as the obstacle gets close.
            #
            # Averaged, not summed, over the contributing returns. A raw sum
            # makes the force proportional to the scanner's angular
            # resolution, so the same wall pushes harder simply for being
            # sampled more finely, and a corridor seen by 400 rays becomes
            # impassable.
            mag = cfg.k_repel * (1.0 / r - 1.0 / cfg.obstacle_influence) / r
            scale = 1.0 / math.sqrt(float(near.sum()))
            rx -= float((mag * np.cos(a)).sum()) * scale
            ry -= float((mag * np.sin(a)).sum()) * scale

            # Nearest return within a narrow cone about the direction of
            # travel, used only for the emergency brake below.
            if planar_dist > 1e-6:
                heading = math.atan2(dy, dx)
                rel = np.arctan2(
                    np.sin(obstacles.angles[near] + yaw - heading),
                    np.cos(obstacles.angles[near] + yaw - heading),
                )
                cone = np.abs(rel) < 0.4
                if cone.any():
                    closest_ahead = float(r[cone].min())

    # ========================== deconfliction ===============================
    # Teammates repel more strongly than walls: two robots in one doorway is a
    # coordination failure, and the recovery is expensive.
    for tm in teammates:
        tdx = position[0] - tm[0]
        tdy = position[1] - tm[1]
        # Vertical separation already deconflicts the drone from the ground pair.
        if abs(position[2] - tm[2]) > 4.0:
            continue
        d = math.hypot(tdx, tdy)
        if 1e-3 < d < cfg.agent_influence:
            mag = cfg.k_deconflict * (1.0 / d - 1.0 / cfg.agent_influence) / d
            rx += mag * tdx / d
            ry += mag * tdy / d

    # ================== keep repulsion out of the route ====================
    # The global planner has already certified that the waypoint being steered
    # at is reachable through free space, so the field's job is to stop the
    # body scraping the walls === not to re-decide whether to go. Projecting
    # out the component of repulsion that opposes the direction of travel does
    # exactly that, leaving the lateral part which centres the robot in a
    # corridor.
    #
    # Without this the platforms could not enter a 4 m alley at all: the
    # influence radius (3.0 m for the quadruped) exceeds the passage
    # half-width, so the two flanking buildings pushed the robot back out of
    # the mouth. Measured, both ground platforms bounced at ~13 m from a
    # courtyard goal indefinitely while open-route travel worked fine.
    if path_following and planar_dist > 1e-6:
        ux, uy = dx / planar_dist, dy / planar_dist
        along = rx * ux + ry * uy
        if along < 0.0:
            rx -= along * ux
            ry -= along * uy

    fx = ax * taper + rx
    fy = ay * taper + ry

    # ============================== escape ==================================
    # Local minimum detection is by *lack of progress*, not by a small net
    # force. Driving head-on into a wall produces a large force (strong
    # repulsion opposing strong attraction) and the robot oscillates in place,
    # so a force-magnitude test misses the most common trap entirely.
    if planar_dist < state.best_distance - 0.3:
        state.best_distance = planar_dist
        state.stuck_time = 0.0
    elif planar_dist > cfg.goal_tolerance:
        state.stuck_time += dt

    # Emergency brake: whatever the route says, do not drive into something
    # that is right in front. This is what the projection above gives up, so
    # it is reinstated explicitly and only at very short range.
    if closest_ahead < 0.9:
        fx *= 0.15
        fy *= 0.15

    if state.stuck_time > cfg.stuck_timeout:
        goal_heading = math.atan2(dy, dx)

        if state.escape_hold <= 0.0:
            # Commit to whichever side has more clear space. Summing the free
            # range either side of the goal bearing is a cheap, robust proxy
            # for "which way round is open".
            state.escape_sign = 1.0
            if obstacles is not None and len(obstacles):
                body_bearing = math.atan2(
                    math.sin(goal_heading - yaw), math.cos(goal_heading - yaw)
                )
                rel = np.arctan2(
                    np.sin(obstacles.angles - body_bearing),
                    np.cos(obstacles.angles - body_bearing),
                )
                ahead = np.abs(rel) < math.pi / 2.0
                if ahead.any():
                    left = obstacles.ranges[ahead & (rel > 0)].sum()
                    right = obstacles.ranges[ahead & (rel < 0)].sum()
                    # More clear range on a side means fewer/further returns
                    # there, so go toward the larger sum.
                    state.escape_sign = 1.0 if left >= right else -1.0
            # Hold the choice long enough to actually clear the obstacle.
            state.escape_hold = 6.0

        # Slide along the obstacle: suppress the attraction that is driving the
        # robot into it, and push perpendicular to the goal bearing.
        tangent = goal_heading + state.escape_sign * math.pi / 2.0
        fx = 0.45 * fx + 1.9 * cfg.k_attract * math.cos(tangent)
        fy = 0.45 * fy + 1.9 * cfg.k_attract * math.sin(tangent)

    if state.escape_hold > 0.0:
        state.escape_hold -= dt
        if state.escape_hold <= 0.0:
            # Re-arm: give the robot a fresh chance to head straight at the
            # goal, and a fresh progress baseline to judge that attempt by.
            state.stuck_time = 0.0
            state.best_distance = planar_dist

    # ====================== world frame -> body frame =======================
    desired_speed = min(cfg.max_speed, math.hypot(fx, fy) * cfg.max_speed)
    desired_heading = math.atan2(fy, fx) if (abs(fx) + abs(fy)) > 1e-9 else yaw
    heading_error = math.atan2(
        math.sin(desired_heading - yaw), math.cos(desired_heading - yaw)
    )

    yaw_rate = max(-cfg.max_yaw_rate, min(cfg.max_yaw_rate, 1.6 * heading_error))

    if abs(heading_error) > cfg.turn_in_place_threshold:
        # Too far off heading to drive: rotate first. A skid-steer that tries
        # to do both just carves a wide arc into the obstacle it is avoiding.
        vx = vy = 0.0
    else:
        vx = desired_speed * math.cos(heading_error)
        vy = desired_speed * math.sin(heading_error)
        # Only platforms that can actually translate sideways get a vy command.
        if cfg.turn_in_place_threshold < 1.5:
            vy = 0.0

    # ============================== altitude ================================
    vz = 0.0
    if altitude_goal is not None:
        vz = max(-2.5, min(2.5, 1.2 * (altitude_goal - position[2])))

    arrived = planar_dist <= cfg.goal_tolerance
    if altitude_goal is not None:
        arrived = arrived and abs(position[2] - altitude_goal) < 2.0

    return vx, vy, vz, yaw_rate, arrived
