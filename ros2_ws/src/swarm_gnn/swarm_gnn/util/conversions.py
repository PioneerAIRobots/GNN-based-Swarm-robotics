"""Conversions between ROS messages and the model layer's plain dataclasses.

The model package (swarm_gnn.model) imports no ROS. That is deliberate: the
allocator, the localiser and the coverage planner all train, profile and
unit-test without a ROS installation, and this module is the only seam where
the two worlds meet.
"""
from __future__ import annotations

import math

from geometry_msgs.msg import Point, Pose, Quaternion
from swarm_interfaces.msg import AgentState, TaskSpec, ThreatDetection, ThreatEstimate

from ..model.graph_builder import AgentView, TaskView
from ..model.rf_localizer import Detection, ThreatTrack
from ..model.spec import NUM_CAPABILITIES, Platform, TaskType, platform_capabilities


def yaw_from_quaternion(q: Quaternion) -> float:
    """Z-axis yaw from a quaternion, ignoring roll and pitch."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def agent_state_to_view(msg: AgentState, markers_remaining: int = 0) -> AgentView:
    caps = tuple(1.0 if b else 0.0 for b in msg.capabilities)
    if len(caps) != NUM_CAPABILITIES:
        # An agent that has not reported a mask yet gets its platform default,
        # which is the right fallback: capabilities are a property of the
        # airframe, not of the message.
        caps = tuple(platform_capabilities(Platform(msg.platform)))
    return AgentView(
        agent_id=msg.agent_id,
        platform=Platform(msg.platform),
        position=(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
        velocity=(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z),
        yaw=yaw_from_quaternion(msg.pose.orientation),
        battery=float(msg.battery),
        health=float(msg.health),
        comms_quality=float(msg.comms_quality),
        capabilities=caps,
        current_task_id=msg.current_task_id,
        markers_remaining=markers_remaining,
    )


def task_spec_to_view(msg: TaskSpec) -> TaskView:
    return TaskView(
        task_id=msg.task_id,
        task_type=TaskType(msg.task_type),
        position=(msg.position.x, msg.position.y, msg.position.z),
        radius=float(msg.radius),
        priority=float(msg.priority),
        risk=float(msg.risk),
        service_time=float(msg.service_time),
        deadline=float(msg.deadline),
        required_capabilities=tuple(
            1.0 if b else 0.0 for b in msg.required_capabilities
        ),
    )


def view_to_task_spec(task: TaskView) -> TaskSpec:
    msg = TaskSpec()
    msg.task_id = task.task_id
    msg.task_type = int(task.task_type)
    msg.position = Point(x=task.position[0], y=task.position[1], z=task.position[2])
    msg.radius = float(task.radius)
    msg.priority = float(task.priority)
    msg.risk = float(task.risk)
    msg.service_time = float(task.service_time)
    msg.deadline = float(task.deadline)
    msg.required_capabilities = [v > 0.5 for v in task.requirement_mask()]
    msg.complete = False
    return msg


def detection_msg_to_model(msg: ThreatDetection, stamp: float = 0.0) -> Detection:
    return Detection(
        observer_id=msg.observer_id,
        position=(
            msg.observer_position.x,
            msg.observer_position.y,
            msg.observer_position.z,
        ),
        rssi_dbm=float(msg.rssi_dbm),
        bearing_rad=float(msg.bearing_rad),
        bearing_sigma_rad=float(msg.bearing_sigma_rad),
        frequency_mhz=float(msg.frequency_mhz),
        stamp=stamp,
    )


def track_to_estimate(track: ThreatTrack) -> ThreatEstimate:
    msg = ThreatEstimate()
    msg.threat_id = track.threat_id
    msg.position = Point(x=track.position[0], y=track.position[1], z=0.0)
    cov = track.covariance
    msg.covariance_xy = [
        float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 0]), float(cov[1, 1])
    ]
    msg.confidence = float(track.confidence)
    msg.frequency_mhz = float(track.frequency_mhz)
    msg.emitter_class = int(track.emitter_class)
    msg.state = int(track.state)
    msg.confirm_count = int(track.confirm_count)
    return msg


def pose_from_xy(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> Pose:
    pose = Pose()
    pose.position = Point(x=float(x), y=float(y), z=float(z))
    pose.orientation = quaternion_from_yaw(yaw)
    return pose
