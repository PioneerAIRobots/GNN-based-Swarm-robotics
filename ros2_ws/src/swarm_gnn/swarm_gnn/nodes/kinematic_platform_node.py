#!/usr/bin/env python3
"""One robot, simulated kinematically. This replaces Gazebo entirely.

Why there is no physics engine here
===================================
The subject of this project is swarm coordination: who goes where, on what
evidence, under what constraints. None of that is decided by contact dynamics,
and running a 1 ms physics step for three sensor-laden models made the mission
far slower than real time on this machine while contributing nothing to the
question. It also cost significant debugging effort on problems that were
purely artifacts of the simulator setup rather than of the system under study.

So each platform integrates its own commanded velocity against the known
sector geometry, publishes the same topics the drive plugins did, and animates
its joints so RViz shows a real articulated robot rather than a floating box.
What is modelled:

  * the platform's true velocity envelope, clamped as the drive would
  * building and rubble collision, with a slide-along-wall response
  * a synthetic 360-degree LaserScan ray-cast against the same geometry
  * per-platform joint animation: a trot gait for the quadruped, rolling
    wheels for the UGV, spinning rotors and a slewing gimbal for the drone

What is not: contact forces, slip, rotor aerodynamics, dynamic balance. Those
are stated as out of scope rather than approximated badly.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from tf2_ros import TransformBroadcaster

from ..model import sector
from ..model.spec import PROFILES, Platform
from ..util.conversions import quaternion_from_yaw
from ..util.qos import SENSOR_QOS

_PLATFORM_BY_NAME = {
    "drone": Platform.DRONE,
    "ugv": Platform.UGV,
    "quadruped": Platform.QUADRUPED,
    "dog": Platform.QUADRUPED,
}

#: Quadruped stance pose and gait shape, matching the URDF's stance_height
#: derivation so the feet meet the ground.
_STANCE_HIP = 0.42
_STANCE_KNEE = -0.95
_HIP_AMPLITUDE = 0.50
_KNEE_AMPLITUDE = 0.60
_STRIDE_HZ = 2.0
#: Trot: diagonal pairs swing together, half a cycle apart.
_LEG_PHASE = {"fl": 0.0, "fr": math.pi, "rl": math.pi, "rr": 0.0}


class KinematicPlatformNode(Node):
    def __init__(self) -> None:
        super().__init__("kinematic_platform")

        self.declare_parameter("agent_id", "ugv_1")
        self.declare_parameter("platform", "ugv")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("scan_rate", 10.0)
        self.declare_parameter("scan_samples", 120)
        self.declare_parameter("scan_range_max", 25.0)
        self.declare_parameter("joint_rate", 30.0)
        self.declare_parameter("frame_prefix", "")
        # Radius used for the collision test, per platform body.
        self.declare_parameter("body_radius", 0.0)

        self.agent_id = str(self.get_parameter("agent_id").value)
        plat_name = str(self.get_parameter("platform").value).lower()
        if plat_name not in _PLATFORM_BY_NAME:
            raise RuntimeError(
                f"platform must be one of {sorted(_PLATFORM_BY_NAME)}, got '{plat_name}'"
            )
        self.platform = _PLATFORM_BY_NAME[plat_name]
        self.profile = PROFILES[self.platform]

        pose = sector.SPAWN_POSES.get(
            self.agent_id, (sector.INSERTION[0], sector.INSERTION[1], 0.2, 0.0)
        )
        self.x, self.y, self.z, self.yaw = pose
        #: Ride height the body holds; the drone is the only platform that
        #: controls z, so the others are pinned here.
        self.ride_height = self.z
        self.cmd = Twist()
        self._cmd_time = 0.0

        radius = float(self.get_parameter("body_radius").value)
        if radius <= 0.0:
            # Body half-diagonal plus a small margin, matching the planner's
            # inflation so the controller is never asked to enter a cell the
            # planner called free.
            radius = {
                Platform.DRONE: 0.0, Platform.UGV: 0.75, Platform.QUADRUPED: 0.4
            }[self.platform]
        self.body_radius = radius

        prefix = str(self.get_parameter("frame_prefix").value)
        self.base_frame = f"{prefix}base_footprint" if prefix else "base_footprint"
        self.scan_frame = f"{prefix}lidar_link" if prefix else "lidar_link"

        self._gait_phase = 0.0
        self._wheel_angle = 0.0
        self._rotor_angle = 0.0
        self._gimbal_tilt = 0.0

        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", SENSOR_QOS)
        # `ground_truth` is what threat_field reads. In a physics run it came
        # from the Gazebo p3d plugin, so the name has to match exactly.
        self.truth_pub = self.create_publisher(Odometry, "ground_truth", SENSOR_QOS)
        self.scan_pub = self.create_publisher(LaserScan, "scan", SENSOR_QOS)
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._boxes = np.array(sector.building_boxes(), dtype=np.float64)
        # The UGV cannot pass the rubble; the other two can.
        include_rubble = self.platform == Platform.UGV
        obstacles = sector.obstacles(include_rubble=include_rubble)
        # Footprint plus height: an aircraft is only obstructed by structures
        # that reach its altitude.
        self._collision_boxes = np.array(
            [[b.x, b.y, b.half_x, b.half_y, b.height] for b in obstacles],
            dtype=np.float64,
        ) if obstacles else np.zeros((0, 5))

        self._dt = 1.0 / max(1.0, float(self.get_parameter("rate").value))
        self.create_timer(self._dt, self._step)
        self.create_timer(
            1.0 / max(1.0, float(self.get_parameter("scan_rate").value)),
            self._publish_scan,
        )
        self.create_timer(
            1.0 / max(1.0, float(self.get_parameter("joint_rate").value)),
            self._publish_joints,
        )

        self.get_logger().info(
            f"kinematic_platform {self.agent_id} ({plat_name}) at "
            f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f}), "
            f"{self.profile.max_speed:.1f} m/s, body radius {self.body_radius:.2f} m"
        )

    # ------------------------------------------------------------- dynamics
    def _on_cmd(self, msg: Twist) -> None:
        self.cmd = msg
        self._cmd_time = self.get_clock().now().nanoseconds * 1e-9

    def _step(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        # Fail safe: a stale command means the controller is gone, so stop.
        if self._cmd_time and now - self._cmd_time > 1.5:
            self.cmd = Twist()

        vmax = self.profile.max_speed
        vx = max(-vmax, min(vmax, self.cmd.linear.x))
        vy = max(-vmax, min(vmax, self.cmd.linear.y))
        wz = max(-2.2, min(2.2, self.cmd.angular.z))

        self.yaw = math.atan2(
            math.sin(self.yaw + wz * self._dt), math.cos(self.yaw + wz * self._dt)
        )
        dx = (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * self._dt
        dy = (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * self._dt

        self._advance(dx, dy)

        if self.platform == Platform.DRONE:
            self.z = max(0.5, min(40.0, self.z + max(-3.5, min(3.5, self.cmd.linear.z)) * self._dt))
        else:
            self.z = self.ride_height

        self._animate(vx, vy, wz)
        self._publish_odom(vx, vy, wz)

    def _advance(self, dx: float, dy: float) -> None:
        """Move, sliding along a wall rather than stopping dead against it.

        A hard block is what makes a kinematic robot look stuck: it ends up
        pressed into a corner with the controller still pushing. Decomposing
        the step and keeping whichever axis is free reproduces the sliding
        contact a real chassis has, and lets the local controller work its way
        out while the global planner supplies a route.
        """
        if not self._in_obstacle(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy
            return
        if not self._in_obstacle(self.x + dx, self.y):
            self.x += dx
            return
        if not self._in_obstacle(self.x, self.y + dy):
            self.y += dy
            return
        # Fully boxed in: hold position and let the planner replan.

    def _in_obstacle(self, x: float, y: float) -> bool:
        """Footprint *and* height test.

        The height term is not a refinement, it is required. An earlier
        version blocked the drone on any footprint below a fixed 8 m ceiling,
        so when it descended to its 7 m thermal-confirm altitude above a
        building it became unable to translate at all and froze 50 m from its
        task for the rest of the mission. An aircraft is obstructed only by
        structures that actually reach it.
        """
        if abs(x) > sector.SECTOR_HALF or abs(y) > sector.SECTOR_HALF:
            return True
        if not len(self._collision_boxes):
            return False
        b = self._collision_boxes
        overlaps = (
            (np.abs(x - b[:, 0]) <= b[:, 2] + self.body_radius)
            & (np.abs(y - b[:, 1]) <= b[:, 3] + self.body_radius)
        )
        if self.platform == Platform.DRONE:
            # Clear anything shorter than the aircraft, with a rotor margin.
            overlaps &= b[:, 4] > self.z - 1.0
        return bool(overlaps.any())

    # ------------------------------------------------------------ animation
    def _animate(self, vx: float, vy: float, wz: float) -> None:
        """Advance the joint animation from the commanded motion.

        The gait and the wheels are driven by the same command that moves the
        body, so what RViz shows always matches what the coordination layer
        asked for: a robot that is holding station stands still rather than
        marching on the spot.
        """
        speed = math.hypot(vx, vy)
        if self.platform == Platform.QUADRUPED:
            # Turning in place still has to cycle the legs.
            drive = min(1.0, (speed + 0.25 * abs(wz)) / 1.2)
            activity = 0.0 if drive < 0.05 else drive
            self._gait_phase = (
                self._gait_phase + 2.0 * math.pi * _STRIDE_HZ * activity * self._dt
            ) % (2.0 * math.pi)
            self._gait_activity = activity
        elif self.platform == Platform.UGV:
            self._wheel_angle += (vx / 0.165) * self._dt
        else:
            # Rotor rate is only indicative; it idles at hover and spins up
            # with commanded speed.
            self._rotor_angle += (90.0 + 25.0 * speed) * self._dt

    def _publish_joints(self) -> None:
        """Publish the animated joint state.

        Names and positions are accumulated in plain lists and assigned once:
        `JointState.position` is an `array.array` after assignment, and `+=`
        on it raises TypeError.
        """
        names: list[str] = []
        positions: list[float] = []

        if self.platform == Platform.QUADRUPED:
            activity = getattr(self, "_gait_activity", 0.0)
            for leg, phase in _LEG_PHASE.items():
                s = math.sin(self._gait_phase + phase)
                # The knee flexes on the swing half only, which is what reads
                # as a step rather than a paddle.
                swing = max(0.0, s)
                names += [f"{leg}_hip_roll", f"{leg}_hip_pitch", f"{leg}_knee"]
                positions += [
                    0.08 * activity * math.cos(self._gait_phase + phase),
                    _STANCE_HIP + activity * _HIP_AMPLITUDE * s,
                    _STANCE_KNEE - activity * _KNEE_AMPLITUDE * swing,
                ]
        elif self.platform == Platform.UGV:
            names = [
                "front_left_wheel_joint", "front_right_wheel_joint",
                "rear_left_wheel_joint", "rear_right_wheel_joint",
            ]
            positions = [self._wheel_angle] * 4
        else:
            for name in (
                "rotor_fl_rotor_joint", "rotor_fr_rotor_joint",
                "rotor_rl_rotor_joint", "rotor_rr_rotor_joint",
            ):
                names.append(name)
                # Adjacent rotors counter-rotate, as on a real X-quad.
                sign = 1.0 if name in (
                    "rotor_fl_rotor_joint", "rotor_rr_rotor_joint"
                ) else -1.0
                positions.append(sign * self._rotor_angle)
            # Slew the gimbal down as the aircraft descends to look at
            # something, and back toward level at cruise.
            target = 1.2 if self.z < 10.0 else 0.35
            self._gimbal_tilt += 0.08 * (target - self._gimbal_tilt)
            names += ["gimbal_pan_joint", "gimbal_tilt_joint"]
            positions += [0.0, self._gimbal_tilt]

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = names
        js.position = positions
        self.joint_pub.publish(js)

    # -------------------------------------------------------------- output
    def _publish_odom(self, vx: float, vy: float, wz: float) -> None:
        stamp = self.get_clock().now().to_msg()
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = self.z
        msg.pose.pose.orientation = quaternion_from_yaw(self.yaw)
        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.angular.z = wz
        self.odom_pub.publish(msg)
        self.truth_pub.publish(msg)

        # map -> base_footprint. robot_state_publisher takes it from there,
        # which is what makes the articulated model appear in RViz.
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "map"
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = self.z
        tf.transform.rotation = quaternion_from_yaw(self.yaw)
        self.tf_broadcaster.sendTransform(tf)

    def _publish_scan(self) -> None:
        """Ray-cast the sector geometry into a LaserScan.

        Vectorised slab test per box: entry and exit parameters along each ray
        are intersected per axis, and the nearest valid hit becomes the range.
        """
        samples = int(self.get_parameter("scan_samples").value)
        range_max = float(self.get_parameter("scan_range_max").value)

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.scan_frame
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = 2.0 * math.pi / samples
        msg.range_min = 0.12
        msg.range_max = range_max

        boxes = self._collision_boxes
        if self.platform == Platform.DRONE and len(boxes):
            # Only structures reaching the aircraft can be seen by a
            # horizontal scanner, and the controller must see the same world
            # the collision test enforces.
            boxes = boxes[boxes[:, 4] > self.z - 1.0]
        if not len(boxes):
            msg.ranges = [float("inf")] * samples
            self.scan_pub.publish(msg)
            return

        angles = msg.angle_min + msg.angle_increment * np.arange(samples) + self.yaw
        dx, dy = np.cos(angles), np.sin(angles)
        ranges = np.full(samples, np.inf)

        for bx, by, hx, hy, _height in boxes:
            with np.errstate(divide="ignore", invalid="ignore"):
                tx1 = (bx - hx - self.x) / dx
                tx2 = (bx + hx - self.x) / dx
                ty1 = (by - hy - self.y) / dy
                ty2 = (by + hy - self.y) / dy
            t_near = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            t_far = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
            hit = (
                (t_far >= np.maximum(t_near, 0.0))
                & np.isfinite(t_near)
                & (t_near > msg.range_min)
            )
            ranges = np.where(hit & (t_near < ranges), t_near, ranges)

        msg.ranges = [float(r) if r <= range_max else float("inf") for r in ranges]
        self.scan_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KinematicPlatformNode()
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
