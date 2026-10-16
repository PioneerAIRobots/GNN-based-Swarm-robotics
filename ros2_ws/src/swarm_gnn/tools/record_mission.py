#!/usr/bin/env python3
"""Record a live mission to JSONL, for later rendering into figures.

    ros2 launch swarm_bringup mission.launch.py rviz:=false &
    python3 tools/record_mission.py --out /tmp/mission.jsonl --seconds 700

Subscribes to the same topics RViz does and appends one frame per sample. The
point of recording rather than screen-grabbing is that the figures in the
README are then generated from the system's actual state === poses, track
states, classifications, swept cells === instead of from whatever happened to
be on screen.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from swarm_interfaces.msg import (
    AgentState,
    AssignmentArray,
    MissionStatus,
    TaskArray,
    ThreatArray,
)

SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=5,
)
COORD = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, depth=1,
)


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self, path: str, period: float) -> None:
        super().__init__("mission_recorder")
        self.agents: dict[str, dict] = {}
        self.threats: dict[str, dict] = {}
        self.tasks: list[dict] = []
        self.assignments: dict[str, str] = {}
        self.status: dict | None = None
        self.frames = 0
        self._fh = open(path, "w")

        self.create_subscription(AgentState, "/swarm/agent_states", self._on_agent, SENSOR)
        self.create_subscription(ThreatArray, "/swarm/threats", self._on_threats, COORD)
        self.create_subscription(TaskArray, "/swarm/tasks", self._on_tasks, COORD)
        self.create_subscription(
            AssignmentArray, "/swarm/assignments", self._on_assign, COORD)
        self.create_subscription(
            MissionStatus, "/swarm/mission_status", self._on_status, COORD)
        self.create_timer(period, self._sample)

    def _on_agent(self, msg: AgentState) -> None:
        self.agents[msg.agent_id] = {
            "id": msg.agent_id,
            "platform": int(msg.platform),
            "x": round(msg.pose.position.x, 2),
            "y": round(msg.pose.position.y, 2),
            "z": round(msg.pose.position.z, 2),
            "yaw": round(yaw_of(msg.pose.orientation), 3),
            "battery": round(float(msg.battery), 3),
            "link": round(float(msg.comms_quality), 3),
            "task": msg.current_task_id,
            "mode": int(msg.agent_mode),
        }

    def _on_threats(self, msg: ThreatArray) -> None:
        self.threats = {
            t.threat_id: {
                "id": t.threat_id,
                "x": round(t.position.x, 2),
                "y": round(t.position.y, 2),
                "state": int(t.state),
                "cls": int(t.emitter_class),
                "sigma": round(
                    math.sqrt(max(float(t.covariance_xy[0]), 0.0)) + 1e-6, 2),
                "corroborations": int(t.confirm_count),
            }
            for t in msg.threats
        }

    def _on_tasks(self, msg: TaskArray) -> None:
        self.tasks = [
            {
                "id": t.task_id, "type": int(t.task_type),
                "x": round(t.position.x, 2), "y": round(t.position.y, 2),
                "priority": round(float(t.priority), 2),
            }
            for t in msg.tasks
        ]

    def _on_assign(self, msg: AssignmentArray) -> None:
        self.assignments = {a.agent_id: a.task_id for a in msg.assignments}

    def _on_status(self, msg: MissionStatus) -> None:
        self.status = {
            "phase": int(msg.phase),
            "phase_name": msg.phase_name,
            "t": round(float(msg.mission_time), 1),
            "coverage": round(float(msg.coverage_fraction), 3),
            "detected": int(msg.threats_detected),
            "confirmed": int(msg.threats_confirmed),
            "neutralized": int(msg.threats_neutralized),
            "open_tasks": int(msg.open_tasks),
            "narrative": msg.narrative,
        }

    def _sample(self) -> None:
        if self.status is None or not self.agents:
            return
        frame = {
            "status": self.status,
            "agents": list(self.agents.values()),
            "threats": list(self.threats.values()),
            "tasks": self.tasks,
            "assignments": self.assignments,
        }
        self._fh.write(json.dumps(frame) + "\n")
        self._fh.flush()
        self.frames += 1

    def close(self) -> None:
        self._fh.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mission.jsonl")
    ap.add_argument("--seconds", type=float, default=700.0)
    ap.add_argument("--period", type=float, default=1.0,
                    help="sampling period; 1 s gives a compact, smooth replay")
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.out, args.period)
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            # Stop as soon as the mission reports completion, plus a short tail.
            if node.status and node.status["phase_name"] == "COMPLETE":
                tail = time.time() + 6.0
                while time.time() < tail:
                    rclpy.spin_once(node, timeout_sec=0.2)
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"recorded {node.frames} frames -> {args.out}")
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
