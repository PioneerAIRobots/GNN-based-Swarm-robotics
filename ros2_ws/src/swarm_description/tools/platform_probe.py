#!/usr/bin/env python3
"""Open-loop motion check for one platform.

Drives a platform directly with a constant Twist, bypassing the whole
coordination layer, and reports what its odometry and the simulator's ground
truth say happened. This isolates a driver problem from a coordination problem. It is how the
UGV's original ride-height and traction bugs were found under physics: spawned
0.12 m into the ground it drifted 0.6 m/s with no command at all, which no
amount of staring at the allocator would have explained.

    # bring the mission up, then drive one platform by hand:
    ros2 launch swarm_bringup mission.launch.py rviz:=false &
    python3 tools/platform_probe.py ugv_1 --vx 1.5 --duration 8
    python3 tools/platform_probe.py dog_1 --wz 0.8 --duration 5
    python3 tools/platform_probe.py drone_1 --vx 2.0 --vz 1.0 --duration 6

Note that the agent controller is also publishing cmd_vel while the mission
runs, so the two commands fight. Use this to answer "does the driver respond
at all", and tools/navigation_check.py (no ROS, no coordination layer) to
answer "can the platform reach where the mission sends it".

Reports odom and ground truth separately on purpose: if they disagree, the
platform's odometry is spawn-relative and its controller needs the
`odom_offset` registration (see swarm_gnn's agent_controller).
"""
from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

# The drivers publish telemetry best-effort, so a reliable subscriber would
# never match them and would silently report nothing.
TELEMETRY_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


def yaw_of(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("namespace", help="e.g. ugv_1")
    ap.add_argument("--vx", type=float, default=0.0, help="forward speed, m/s")
    ap.add_argument("--vy", type=float, default=0.0, help="lateral speed, m/s")
    ap.add_argument("--vz", type=float, default=0.0, help="climb rate, m/s")
    ap.add_argument("--wz", type=float, default=0.0, help="yaw rate, rad/s")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--wait", type=float, default=15.0,
                    help="seconds to wait for the first telemetry message")
    args = ap.parse_args()

    rclpy.init()
    node = Node("platform_probe")
    seen: dict[str, tuple] = {}
    joints: dict[str, float] = {}

    def make_cb(key: str):
        def cb(msg: Odometry) -> None:
            p = msg.pose.pose.position
            seen[key] = (p.x, p.y, p.z, yaw_of(msg))
        return cb

    node.create_subscription(
        Odometry, f"/{args.namespace}/odom", make_cb("odom"), TELEMETRY_QOS)
    node.create_subscription(
        Odometry, f"/{args.namespace}/ground_truth", make_cb("truth"), TELEMETRY_QOS)

    def on_joints(msg: JointState) -> None:
        for name, vel in zip(msg.name, msg.velocity or [0.0] * len(msg.name)):
            joints[name] = vel

    node.create_subscription(
        JointState, f"/{args.namespace}/joint_states", on_joints, TELEMETRY_QOS)

    deadline = time.time() + args.wait
    while time.time() < deadline and len(seen) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not seen:
        print(f"no telemetry from /{args.namespace} after {args.wait:.0f} s "
              f"=== is the platform spawned?")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    start = dict(seen)
    for key in sorted(start):
        x, y, z, yaw = start[key]
        print(f"  start {key:6s} x={x:8.2f} y={y:8.2f} z={z:6.2f} "
              f"yaw={math.degrees(yaw):7.1f}")

    pub = node.create_publisher(Twist, f"/{args.namespace}/cmd_vel", 10)
    cmd = Twist()
    cmd.linear.x, cmd.linear.y, cmd.linear.z = args.vx, args.vy, args.vz
    cmd.angular.z = args.wz

    t0 = time.time()
    while time.time() - t0 < args.duration:
        pub.publish(cmd)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.04)
    # Always leave the platform stopped, and let the stop be observed.
    pub.publish(Twist())
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f"after {args.duration:.0f} s at vx={args.vx} vy={args.vy} "
          f"vz={args.vz} wz={args.wz}:")
    for key in sorted(seen):
        x, y, z, yaw = seen[key]
        print(f"  end   {key:6s} x={x:8.2f} y={y:8.2f} z={z:6.2f} "
              f"yaw={math.degrees(yaw):7.1f}")
        if key in start:
            dx, dy = x - start[key][0], y - start[key][1]
            dyaw = math.degrees(math.atan2(
                math.sin(yaw - start[key][3]), math.cos(yaw - start[key][3])))
            print(f"        moved {math.hypot(dx, dy):6.2f} m, "
                  f"turned {dyaw:7.1f} deg, dz {z - start[key][2]:+.2f} m")

    if joints:
        print("  joint velocities (rad/s):")
        for name in sorted(joints):
            print(f"      {name:28s} {joints[name]:+7.3f}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
