#!/usr/bin/env python3
"""Confirm RViz is actually being fed everything the display expects.

Run it while the mission is up:

    ros2 launch swarm_bringup mission.launch.py rviz:=false &
    python3 tools/rviz_inputs_check.py

Checks the marker namespaces, the belief grid, and each robot's description and
joint states, with the QoS each publisher actually uses. A silent display is
otherwise hard to diagnose: RViz shows an empty world whether a topic is
missing, mismatched on QoS, or published once to nobody.

It has caught two real problems: the sector was published a single time on a
volatile topic, so RViz showed robots floating in empty space whenever it
started after the visualiser; and the RF belief relied on RViz's Map display,
which cannot render an OccupancyGrid on OpenGL 2.1.
"""
from __future__ import annotations

import sys
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

AGENTS = ("drone_1", "ugv_1", "dog_1")
#: Namespaces the RViz display is built around.
EXPECTED_NAMESPACES = {
    "sector", "agents", "tasks", "threats", "assignments", "comms", "mission",
}
#: Joint counts each platform must animate, from its URDF.
EXPECTED_JOINTS = {"drone_1": 6, "ugv_1": 4, "dog_1": 12}

RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=2,
)
LATCHED = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=2,
)


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0

    rclpy.init()
    node = Node("rviz_inputs_check")
    namespaces: dict[str, int] = {}
    seen: dict[str, str] = {}

    def on_markers(msg: MarkerArray) -> None:
        for marker in msg.markers:
            namespaces[marker.ns] = namespaces.get(marker.ns, 0) + 1

    node.create_subscription(MarkerArray, "/swarm/markers", on_markers, RELIABLE)
    node.create_subscription(
        OccupancyGrid, "/swarm/belief_grid",
        lambda m: seen.__setitem__("belief_grid", f"{m.info.width}x{m.info.height}"),
        LATCHED,
    )
    for agent in AGENTS:
        node.create_subscription(
            JointState, f"/{agent}/joint_states",
            (lambda a: lambda m: seen.__setitem__(f"joints:{a}", str(len(m.name))))(agent),
            10,
        )
        node.create_subscription(
            String, f"/{agent}/robot_description",
            (lambda a: lambda m: seen.__setitem__(
                f"description:{a}", f"{len(m.data) // 1024} KB"))(agent),
            LATCHED,
        )

    deadline = time.time() + seconds
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    print(f"listened for {seconds:.0f} s\n")
    print("marker namespaces:")
    for name in sorted(namespaces):
        print(f"  {name:14s} {namespaces[name]} markers")

    print("\nother inputs:")
    for key in sorted(seen):
        print(f"  {key:22s} {seen[key]}")

    problems = []
    for name in sorted(EXPECTED_NAMESPACES - set(namespaces)):
        problems.append(f"no markers in namespace '{name}'")
    if "belief_grid" not in seen:
        problems.append("no belief grid")
    if "belief" not in namespaces:
        problems.append("belief not drawn as markers (invisible on OpenGL 2.1)")
    for agent, count in EXPECTED_JOINTS.items():
        key = f"joints:{agent}"
        if key not in seen:
            problems.append(f"{agent} publishes no joint states")
        elif int(seen[key]) != count:
            problems.append(f"{agent} animates {seen[key]} joints, expected {count}")
        if f"description:{agent}" not in seen:
            problems.append(f"{agent} publishes no robot_description")

    node.destroy_node()
    rclpy.shutdown()

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nall RViz inputs present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
