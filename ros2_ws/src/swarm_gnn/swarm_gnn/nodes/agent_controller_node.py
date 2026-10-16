#!/usr/bin/env python3
"""Per-robot executive: turns an assignment into motion and sensor actions.

One instance runs per platform, inside that platform's namespace. It is the
only node that writes cmd_vel, and the only one that reports AgentState, so it
is the boundary between the swarm layer and the robot.

Responsibilities
================
  * publish AgentState (pose, battery, health, capabilities, comms) at 5 Hz ===
    this is the agent's entry in the coordination graph
  * drive to the assigned task with the potential-field controller, using the
    platform's own gains and the GNN's caution scalar for this agent
  * execute the on-station behaviour once there: hold a thermal standoff, take
    a CBRN sample, hold a relay station, dwell for a DF cut
  * report sensor corroborations to /swarm/confirmations, which is what moves a
    track from LOCALIZED to CONFIRMED

Navigation is two-layer, and it has to be. A* over the known sector geometry
decides which way round the blocks; the potential field tracks the next
waypoint and handles what the map does not contain === the other two robots and
live scanner returns. With the field alone the platforms stuck: the quadruped
managed 4 m in 70 s with its goal behind a building, because a street grid is
a field of concave traps and a reactive controller cannot leave one.

Battery is simulated here rather than read from hardware: it drains with
distance travelled using the platform's energy_per_m, which is what makes the
allocator's energy term bite over a long mission.
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray, String
from swarm_interfaces.msg import (
    AgentState,
    AssignmentArray,
    TaskArray,
    TaskSpec,
    ThreatArray,
    ThreatEstimate,
)

from ..model.navigation import (
    NavState,
    ScanObstacles,
    config_for,
    potential_field_command,
)
from ..model.planner import GridPlanner, PathFollower
from ..model import sector
from ..model.spec import PROFILES, Platform, TaskType, platform_capabilities
from ..util.conversions import quaternion_from_yaw, yaw_from_quaternion
from ..util.qos import COORDINATION_QOS, SENSOR_QOS, WORLD_MODEL_QOS

_PLATFORM_BY_NAME = {
    "drone": Platform.DRONE,
    "ugv": Platform.UGV,
    "quadruped": Platform.QUADRUPED,
    "dog": Platform.QUADRUPED,
}


class AgentControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("agent_controller")

        self.declare_parameter("agent_id", "ugv_1")
        self.declare_parameter("platform", "ugv")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("state_rate", 5.0)
        # Transit altitude for the aircraft. Must clear the tallest structure:
        # the planner gives the drone straight lines on the assumption it can
        # overfly anything, and the platform's collision test is height-aware,
        # so a cruise below the rooftops means the drone flies into a block and
        # stops. Measured with a 14 m cruise against 26 m buildings: the drone
        # sat 30 m from its confirm task for an entire 567 s mission.
        self.declare_parameter("cruise_altitude", sector.MAX_STRUCTURE_HEIGHT + 5.0)
        # Altitude the drone drops to for a thermal confirm or BDA pass; the
        # narrow LWIR cannot resolve a target from cruise.
        self.declare_parameter("confirm_altitude", 7.0)
        # Horizontal distance within which the aircraft descends to the
        # confirm altitude. Outside it the aircraft stays at transit height,
        # because descending early puts it among the rooftops en route.
        self.declare_parameter("descend_radius", 14.0)
        self.declare_parameter("rally_point", [-52.0, 52.0])
        # CBRN reading above which a sample counts as a positive detection.
        self.declare_parameter("cbrn_threshold", 0.15)
        # Link quality below which the platform reports active jamming. The
        # jammer drives comms to d/jam_radius, so standing at a 9 m thermal
        # standoff inside a 38 m radius reads about 0.24.
        self.declare_parameter("jam_threshold", 0.60)
        # Seconds on station before an action is reported complete.
        self.declare_parameter("dwell_time", 4.0)
        self.declare_parameter("scan_topic", "scan")
        # Replan if the robot has made no progress along its path for this
        # long: the map is static, so a stall means the plan no longer fits
        # (a teammate in the way, or a goal that has moved).
        self.declare_parameter("replan_stall_time", 6.0)
        # Hard ceiling on replanning, so a genuinely unreachable goal costs a
        # few searches rather than one per control tick.
        self.declare_parameter("replan_min_interval", 1.5)
        # Pose of this platform's odometry origin in the map frame, as
        # [x, y, yaw].
        #
        # Wheel odometry is integrated from wherever the robot started, so a
        # wheeled platform reports positions relative to its own spawn point
        # while every task is specified in the map frame. Left uncorrected the
        # UGV drove off the sector entirely: it believed it was at the origin,
        # so each map-frame goal was executed as an offset from its real
        # position (observed running to (-148, -134) in a 110 m sector).
        #
        # On a real vehicle this registration comes from GNSS or AMCL. Here the
        # launch file supplies the known insertion pose, which is the same
        # information a crew would enter before stepping off. Platforms whose
        # driver already reports absolute world pose use [0, 0, 0].
        self.declare_parameter("odom_offset", [0.0, 0.0, 0.0])

        self.agent_id = str(self.get_parameter("agent_id").value)
        offset = [float(v) for v in self.get_parameter("odom_offset").value]
        if len(offset) != 3:
            raise RuntimeError("odom_offset must be [x, y, yaw]")
        self.odom_offset = tuple(offset)
        plat_name = str(self.get_parameter("platform").value).lower()
        if plat_name not in _PLATFORM_BY_NAME:
            raise RuntimeError(
                f"platform must be one of {sorted(_PLATFORM_BY_NAME)}, got '{plat_name}'"
            )
        self.platform = _PLATFORM_BY_NAME[plat_name]
        self.profile = PROFILES[self.platform]

        # ---------------------------------------------------------- state
        self.position = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.yaw = 0.0
        self.battery = 1.0
        self.health = 1.0
        self.comms_quality = 1.0
        self.cbrn_reading = 0.0
        self.caution = 0.85
        self.obstacles: ScanObstacles | None = None
        self.teammates: dict[str, tuple[float, float, float]] = {}
        self.tasks: dict[str, TaskSpec] = {}
        self.threats: dict[str, ThreatEstimate] = {}
        self.assigned_task_id = ""
        self.nav_state = NavState()
        self.planner = GridPlanner()
        self.follower = PathFollower()
        self._plan_goal: tuple[float, float] | None = None
        self._plan_time = 0.0
        self._plan_progress = float("inf")
        self._plan_progress_time = 0.0
        #: Set when the planner could only reach an approach position, so
        #: arrival is reported against that instead of the unreachable goal.
        self._plan_approach_only = False
        #: The point the planner was actually sent to. May differ from the
        #: task position when that position is inside a building.
        self._effective_goal: tuple[float, float] | None = None
        #: Task ids this agent has finished servicing.
        self._completed: set[str] = set()
        self._dwell = 0.0
        self._last_position: tuple[float, float, float] | None = None
        self._reported: set[str] = set()
        self._mode = AgentState.MODE_IDLE

        # -------------------------------------------------------- interfaces
        self.create_subscription(Odometry, "odom", self._on_odom, SENSOR_QOS)
        self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value),
            self._on_scan, SENSOR_QOS,
        )
        self.create_subscription(Float32, "comms_quality", self._on_comms, SENSOR_QOS)
        self.create_subscription(Float32, "cbrn_reading", self._on_cbrn, SENSOR_QOS)

        self.create_subscription(
            AssignmentArray, "/swarm/assignments", self._on_assignments, COORDINATION_QOS
        )
        self.create_subscription(TaskArray, "/swarm/tasks", self._on_tasks, COORDINATION_QOS)
        self.create_subscription(ThreatArray, "/swarm/threats", self._on_threats, WORLD_MODEL_QOS)
        self.create_subscription(AgentState, "/swarm/agent_states", self._on_peer, SENSOR_QOS)
        self.create_subscription(
            Float32MultiArray, "/swarm/caution", self._on_caution, SENSOR_QOS
        )
        self.create_subscription(
            String, "/swarm/caution_order", self._on_caution_order, SENSOR_QOS
        )
        self._caution_order: list[str] = []

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(AgentState, "/swarm/agent_states", SENSOR_QOS)
        self.confirm_pub = self.create_publisher(
            String, "/swarm/confirmations", WORLD_MODEL_QOS
        )
        # "<agent_id>:<task_id>" once the on-station work is done. Without
        # this nothing retires a task: an agent that had dwelled its service
        # time kept holding the assignment, and with commitment in the
        # allocator it never received new work. The whole team parked.
        self.complete_pub = self.create_publisher(
            String, "/swarm/task_complete", WORLD_MODEL_QOS
        )

        control_rate = float(self.get_parameter("control_rate").value)
        self._dt = 1.0 / max(1.0, control_rate)
        self.create_timer(self._dt, self._control_tick)
        self.create_timer(
            1.0 / max(1.0, float(self.get_parameter("state_rate").value)),
            self._publish_state,
        )

        self.get_logger().info(
            f"agent_controller up: {self.agent_id} ({plat_name}), "
            f"{self.profile.max_speed:.1f} m/s, caps="
            f"{[c.name for c in self.profile.capabilities]}"
        )

    # ------------------------------------------------------------- callbacks
    def _on_odom(self, msg: Odometry) -> None:
        """Compose the driver's odometry with the map<-odom registration."""
        p = msg.pose.pose.position
        ox, oy, oyaw = self.odom_offset
        cos_o, sin_o = math.cos(oyaw), math.sin(oyaw)
        self.position = (
            ox + p.x * cos_o - p.y * sin_o,
            oy + p.x * sin_o + p.y * cos_o,
            p.z,
        )
        raw_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.yaw = math.atan2(
            math.sin(oyaw + raw_yaw), math.cos(oyaw + raw_yaw)
        )
        # Twist is already body-frame per nav_msgs/Odometry, so it needs no
        # rotation into the map frame.
        v = msg.twist.twist.linear
        self.velocity = (v.x, v.y, v.z)

    def _on_scan(self, msg: LaserScan) -> None:
        self.obstacles = ScanObstacles.from_scan(
            msg.angle_min, msg.angle_increment, msg.ranges, msg.range_max
        )

    def _on_comms(self, msg: Float32) -> None:
        self.comms_quality = float(msg.data)

    def _on_cbrn(self, msg: Float32) -> None:
        self.cbrn_reading = float(msg.data)

    def _on_assignments(self, msg: AssignmentArray) -> None:
        for a in msg.assignments:
            if a.agent_id != self.agent_id:
                continue
            if a.task_id != self.assigned_task_id:
                # New task: the stuck detector, the dwell timer and the global
                # plan are all relative to the old goal, so drop all three.
                self.assigned_task_id = a.task_id
                self.nav_state = NavState()
                self._dwell = 0.0
                self._invalidate_plan()
            return
        # Not in the assignment list at all: stand down.
        if self.assigned_task_id:
            self.assigned_task_id = ""
            self.nav_state = NavState()
            self._invalidate_plan()

    def _on_tasks(self, msg: TaskArray) -> None:
        self.tasks = {t.task_id: t for t in msg.tasks}

    def _on_threats(self, msg: ThreatArray) -> None:
        for est in msg.threats:
            self.threats[est.threat_id] = est

    def _on_peer(self, msg: AgentState) -> None:
        if msg.agent_id == self.agent_id:
            return
        self.teammates[msg.agent_id] = (
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        )

    def _on_caution_order(self, msg: String) -> None:
        self._caution_order = [s for s in msg.data.split(",") if s]

    def _on_caution(self, msg: Float32MultiArray) -> None:
        """Pick this agent's entry out of the allocator's caution vector."""
        if self.agent_id in self._caution_order:
            idx = self._caution_order.index(self.agent_id)
            if idx < len(msg.data):
                self.caution = float(msg.data[idx])

    # --------------------------------------------------------- control loop
    def _control_tick(self) -> None:
        self._drain_battery()

        goal, altitude_goal, mode = self._resolve_goal()
        self._mode = mode

        if goal is None:
            self.cmd_pub.publish(Twist())
            return

        # Global layer: route around the blocks, then steer at the next
        # waypoint rather than straight at the goal.
        steer_to = self._steer_target(goal)
        cfg = config_for(self.platform, self.caution)

        vx, vy, vz, yaw_rate, reached_waypoint = potential_field_command(
            position=self.position,
            yaw=self.yaw,
            goal=(steer_to[0], steer_to[1], goal[2]),
            cfg=cfg,
            obstacles=self.obstacles,
            teammates=list(self.teammates.values()),
            state=self.nav_state,
            dt=self._dt,
            altitude_goal=altitude_goal,
        )

        # Arrival is judged against the *task*, not the waypoint: clearing an
        # intermediate waypoint is not the same as being on station.
        arrived = self._arrived_at(goal, altitude_goal)

        if arrived:
            self._dwell += self._dt
            self._on_station()
            self._maybe_complete(goal)
            cmd = Twist()
            if altitude_goal is not None:
                cmd.linear.z = vz
            self.cmd_pub.publish(cmd)
            return

        self._dwell = 0.0
        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.linear.z = vz
        cmd.angular.z = yaw_rate

        # An aircraft below transit height with a long way to go climbs first.
        # Translating while still among the rooftops is what the height-aware
        # collision test blocks, and the drone would simply stop.
        if (
            self.platform == Platform.DRONE
            and altitude_goal is not None
            and altitude_goal - self.position[2] > 2.0
            and math.dist(self.position[:2], (goal[0], goal[1]))
            > float(self.get_parameter("descend_radius").value)
        ):
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            self._mode = AgentState.MODE_TRANSIT

        self.cmd_pub.publish(cmd)

    # --------------------------------------------------------- global layer
    def _invalidate_plan(self) -> None:
        self.follower.clear()
        self._plan_goal = None
        self._plan_progress = float("inf")
        self._plan_approach_only = False

    def _steer_target(self, goal: tuple[float, float, float]) -> tuple[float, float]:
        """Next waypoint on the global path toward `goal`.

        Replans when the goal moves, when the path runs out, or when the robot
        stops making progress along it. Everything else is left to the local
        controller.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        goal_xy = (goal[0], goal[1])
        here = (self.position[0], self.position[1])

        needs_plan = (
            self._plan_goal is None
            or math.dist(self._plan_goal, goal_xy) > 2.0
            or not self.follower.active
        )

        # Stall detection against progress along the path, not against raw
        # distance to the goal: a path that legitimately rounds a block moves
        # away from the goal for a while.
        target = self.follower.current(here) if self.follower.active else None
        if target is not None:
            remaining = math.dist(here, target) + 3.0 * (
                len(self.follower.waypoints) - self.follower.index
            )
            if remaining < self._plan_progress - 0.5:
                self._plan_progress = remaining
                self._plan_progress_time = now
            elif now - self._plan_progress_time > float(
                self.get_parameter("replan_stall_time").value
            ):
                needs_plan = True

        min_interval = float(self.get_parameter("replan_min_interval").value)
        if needs_plan and now - self._plan_time >= min_interval:
            # Nudge a goal that sits inside a building out to reachable ground
            # before searching, so a task generated from the belief grid does
            # not simply fail to plan.
            safe_goal = sector.nearest_free(goal_xy[0], goal_xy[1], margin=0.6)
            # Record it: arrival has to be judged against the point we are
            # actually driving to. Judging it against the original task
            # position left robots parked at a reachable approach point,
            # 12 m from a goal inside a building, holding the task forever.
            self._effective_goal = safe_goal
            result = self.planner.plan(self.platform, here, safe_goal)
            self._plan_time = now
            self._plan_goal = goal_xy
            self._plan_progress = float("inf")
            self._plan_progress_time = now
            if result:
                self.follower.set_plan(result.waypoints)
                self._plan_approach_only = not result.reaches_goal
                if result.truncated:
                    self.get_logger().warn(
                        f"no route to ({goal_xy[0]:.1f}, {goal_xy[1]:.1f}); "
                        f"heading to the nearest reachable point instead"
                    )
            else:
                # Planning failed outright; fall back to steering straight at
                # the goal so the platform is never simply parked.
                self.follower.set_plan([safe_goal])
                self._plan_approach_only = False

        target = self.follower.current(here)
        return target if target is not None else goal_xy

    def _arrived_at(
        self, goal: tuple[float, float, float], altitude_goal: float | None
    ) -> bool:
        """On-station test against the task's own acceptance radius."""
        task = self.tasks.get(self.assigned_task_id)
        cfg = config_for(self.platform, self.caution)
        tolerance = cfg.goal_tolerance
        if task is not None:
            # Honour the task's radius, but never demand better than the
            # controller can actually hold.
            tolerance = max(min(float(task.radius), 12.0), cfg.goal_tolerance * 0.6)

        planar = math.dist(self.position[:2], (goal[0], goal[1]))

        # When the task position itself is not reachable, the approach point
        # the planner was given is the best this platform can do, so arrival
        # is judged there. Otherwise a task whose position sits inside a
        # building can never be completed and blocks the agent indefinitely.
        if self._effective_goal is not None:
            offset = math.dist(self._effective_goal, (goal[0], goal[1]))
            if offset > 0.5:
                tolerance = max(tolerance, offset + 1.5)
        if planar > tolerance:
            return False
        if altitude_goal is not None and abs(self.position[2] - altitude_goal) > 2.5:
            return False
        return True

    def _resolve_goal(self):
        """Where to go, at what altitude, and in what mode.

        Returns (goal, altitude_goal, mode). `altitude_goal` is None for the
        ground platforms, which do not control z.
        """
        cruise = float(self.get_parameter("cruise_altitude").value)
        confirm_alt = float(self.get_parameter("confirm_altitude").value)
        is_air = self.platform == Platform.DRONE

        task = self.tasks.get(self.assigned_task_id)
        if task is None:
            if not self.assigned_task_id:
                # Unassigned: an aircraft still has to hold altitude.
                return (
                    (self.position[0], self.position[1], self.position[2]) if is_air else None,
                    cruise if is_air else None,
                    AgentState.MODE_IDLE,
                )
            # Assigned to a task we have not received the spec for yet.
            return None, cruise if is_air else None, AgentState.MODE_IDLE

        tx, ty = task.position.x, task.position.y
        ttype = TaskType(task.task_type)
        horizontal = math.dist(self.position[:2], (tx, ty))
        descend_radius = float(self.get_parameter("descend_radius").value)

        if ttype in (TaskType.VISUAL_CONFIRM, TaskType.BDA):
            # Hold transit height until the target is underneath AND the
            # aircraft is actually over ground it can occupy at the lower
            # altitude. Descending en route puts it among structures the
            # planner routed it over, where the height-aware collision test
            # stops it translating at all === the drone sat 13 m from its
            # confirm task for the rest of a mission that way.
            if is_air and (
                horizontal > descend_radius
                or sector.blocked_at_height(
                    self.position[0], self.position[1], confirm_alt, margin=0.5
                )
            ):
                return (tx, ty, cruise), cruise, AgentState.MODE_TRANSIT
            return (
                (tx, ty, confirm_alt),
                confirm_alt if is_air else None,
                AgentState.MODE_CONFIRM,
            )
        if ttype == TaskType.COMMS_RELAY:
            # Relay as high as the platform can get: altitude buys range.
            return (tx, ty, cruise), cruise if is_air else None, AgentState.MODE_RELAY
        if ttype == TaskType.CBRN_SAMPLE:
            return (tx, ty, 0.0), None, AgentState.MODE_ACT
        if ttype == TaskType.EMPLACE_JAMMER:
            return (tx, ty, 0.0), None, AgentState.MODE_ACT
        if ttype == TaskType.RF_TRIANGULATE:
            return (tx, ty, cruise), cruise if is_air else None, AgentState.MODE_SEARCH
        # SEARCH_CELL and anything else: transit at cruise.
        return (tx, ty, cruise), cruise if is_air else None, AgentState.MODE_SEARCH

    def _on_station(self) -> None:
        """Run the sensor action for the assigned task, once dwelled.

        Each action reports what it *observed*, never what it concluded; the
        localiser derives the emitter class from the pooled evidence. See
        ThreatTrack.classify for why: a jammer and a licensed transmitter look
        identical on RF power and both read cold on CBRN, so a robot that
        asserts a class from one sensor gets it wrong.
        """
        task = self.tasks.get(self.assigned_task_id)
        if task is None:
            return
        if self._dwell < float(self.get_parameter("dwell_time").value):
            return

        ttype = TaskType(task.task_type)
        threat_id = self._threat_for_task(task.task_id)

        if ttype == TaskType.VISUAL_CONFIRM and threat_id:
            if self._report_once(threat_id, "thermal"):
                # Two observations come out of one overflight: the thermal
                # return says a powered emitter is really there, and the
                # platform's own link quality says whether it is being jammed.
                # The jamming cut is the only thing that separates a jammer
                # from a licensed transmitter, so it is taken here where the
                # aircraft is closest to the emitter.
                jam_threshold = float(self.get_parameter("jam_threshold").value)
                jammed = self.comms_quality < jam_threshold
                tokens = ["thermal_hot", "jam_detected" if jammed else "no_jam"]
                self.confirm_pub.publish(
                    String(data=f"{threat_id}:thermal:{','.join(tokens)}")
                )
                self.get_logger().info(
                    f"thermal confirm on {threat_id}: powered emitter, "
                    f"link {self.comms_quality:.2f} -> "
                    f"{'JAMMING DETECTED' if jammed else 'no jamming'}"
                )

        elif ttype == TaskType.CBRN_SAMPLE and threat_id:
            if self._report_once(threat_id, "cbrn"):
                threshold = float(self.get_parameter("cbrn_threshold").value)
                positive = self.cbrn_reading >= threshold
                token = "cbrn_pos" if positive else "cbrn_neg"
                self.confirm_pub.publish(
                    String(data=f"{threat_id}:cbrn:{token}")
                )
                if positive:
                    self.get_logger().info(
                        f"CBRN POSITIVE on {threat_id} ({self.cbrn_reading:.2f}) "
                        f"=== explosive precursor present"
                    )
                else:
                    # A clean sample rules out a trigger. It does NOT make the
                    # emitter civilian; that needs the jamming cut too.
                    self.get_logger().info(
                        f"CBRN clean on {threat_id} ({self.cbrn_reading:.2f}) "
                        f"=== not a trigger"
                    )

        elif ttype == TaskType.BDA and threat_id:
            if self._report_once(threat_id, "bda"):
                self.get_logger().info(f"BDA pass complete on {threat_id}")

    def _maybe_complete(self, goal) -> None:
        """Release the task once its service time has been spent on station.

        `service_time` is the task's own statement of how long the work takes,
        so it is the right completion criterion for the dwell-type tasks
        (a DF cut, a relay station, a sample). Search cells are retired by the
        executive on proximity instead, and are left alone here.
        """
        task = self.tasks.get(self.assigned_task_id)
        if task is None or self.assigned_task_id in self._completed:
            return
        required = max(
            float(task.service_time), float(self.get_parameter("dwell_time").value)
        )
        if self._dwell < required:
            return
        self._completed.add(self.assigned_task_id)
        self.complete_pub.publish(
            String(data=f"{self.agent_id}:{self.assigned_task_id}")
        )
        self.get_logger().info(
            f"completed {self.assigned_task_id} after {self._dwell:.0f} s on station"
        )

    def _report_once(self, threat_id: str, modality: str) -> bool:
        """Latch so a long dwell does not re-report the same observation."""
        key = f"{threat_id}:{modality}"
        if key in self._reported:
            return False
        self._reported.add(key)
        return True

    def _threat_for_task(self, task_id: str) -> str:
        """Recover the track id a threat task was generated for.

        The executive names them "<threat_id>_<suffix>"; parsing it back here
        avoids adding a threat_id field to TaskSpec that only some task types
        would ever populate.
        """
        for suffix in ("_confirm", "_cbrn", "_mark", "_bda"):
            if task_id.endswith(suffix):
                return task_id[: -len(suffix)]
        if "_tri" in task_id:
            return task_id.split("_tri")[0]
        return ""

    def _drain_battery(self) -> None:
        """Deplete the battery by distance travelled, per the platform profile."""
        if self._last_position is not None:
            step = math.dist(self.position[:2], self._last_position[:2])
            # Hovering costs a rotorcraft energy even at zero ground speed.
            idle = 2.0e-5 if self.platform == Platform.DRONE else 2.0e-6
            self.battery = max(
                0.0, self.battery - self.profile.energy_per_m * step - idle
            )
        self._last_position = self.position

    # -------------------------------------------------------------- reporting
    def _publish_state(self) -> None:
        msg = AgentState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.agent_id = self.agent_id
        msg.platform = int(self.platform)
        msg.pose.position.x = self.position[0]
        msg.pose.position.y = self.position[1]
        msg.pose.position.z = self.position[2]
        msg.pose.orientation = quaternion_from_yaw(self.yaw)
        msg.twist.linear.x = self.velocity[0]
        msg.twist.linear.y = self.velocity[1]
        msg.twist.linear.z = self.velocity[2]
        msg.battery = float(self.battery)
        msg.max_speed = float(self.profile.max_speed)
        # Health degrades with a flat battery: a platform that cannot make the
        # return leg is not a healthy one, and the allocator should know.
        msg.health = float(min(1.0, max(0.1, self.battery * 4.0)))
        msg.capabilities = [v > 0.5 for v in platform_capabilities(self.platform)]
        msg.current_task_id = self.assigned_task_id
        msg.agent_mode = int(self._mode)
        msg.comms_quality = float(self.comms_quality)
        self.state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgentControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
