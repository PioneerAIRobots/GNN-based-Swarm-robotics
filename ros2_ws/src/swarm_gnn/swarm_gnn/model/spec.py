"""Shared vocabulary for the coordination layer.

Everything downstream === the graph encoder, the cost oracle, the ROS nodes and
the training script === agrees on the orderings defined here. Changing a feature
order without retraining will silently degrade the policy, so these tables are
deliberately in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Platform(IntEnum):
    """Must mirror swarm_interfaces/AgentState PLATFORM_* constants."""

    DRONE = 0
    UGV = 1
    QUADRUPED = 2


class TaskType(IntEnum):
    """Must mirror swarm_interfaces/TaskSpec TYPE_* constants."""

    SEARCH_CELL = 0
    RF_TRIANGULATE = 1
    VISUAL_CONFIRM = 2
    CBRN_SAMPLE = 3
    EMPLACE_JAMMER = 4
    COMMS_RELAY = 5
    BDA = 6


class Capability(IntEnum):
    """Capability mask indices, shared by AgentState and TaskSpec."""

    AERIAL_VIEW = 0        # can observe from altitude
    GROUND_MOBILITY = 1    # can traverse trafficable ground
    ROUGH_TERRAIN = 2      # stairs, rubble, kerbs
    RF_DIRECTION_FIND = 3  # carries a DF array
    THERMAL_IMAGING = 4    # carries an LWIR sensor
    CBRN_SAMPLING = 5      # carries a point sampler
    MARKER_EMPLACE = 6     # carries expendable jammer-markers
    HIGH_GAIN_RADIO = 7    # can serve as a comms relay


NUM_CAPABILITIES = len(Capability)
NUM_TASK_TYPES = len(TaskType)
NUM_PLATFORMS = len(Platform)


class EdgeType(IntEnum):
    """Must mirror swarm_interfaces/GraphDebug edge_type."""

    AGENT_AGENT = 0
    AGENT_TASK = 1
    TASK_TASK = 2


@dataclass(frozen=True)
class PlatformProfile:
    """Physical envelope and sensor fit of one platform class.

    These numbers are the contract between the URDFs and the coordination
    layer: `max_speed` matches the drive plugin's clamp, and `capabilities`
    matches the sensors actually bolted to that robot's xacro.
    """

    platform: Platform
    max_speed: float          # m/s, sustained
    capabilities: tuple[Capability, ...]
    # Fraction of the sector the platform can actually reach. The UGV is
    # road-bound and the rubble blocks the east end, so it cannot reach
    # everywhere the other two can.
    reachable_fraction: float
    # DF bearing standard deviation, radians. Follows from the array baseline
    # declared in each robot's rf_df_array macro.
    df_sigma: float
    # Relative exposure when operating in a contested cell. The drone is seen
    # from far away; the dog is the hardest to detect.
    signature: float
    # Energy drain per metre travelled, as a fraction of a full charge.
    energy_per_m: float


PROFILES: dict[Platform, PlatformProfile] = {
    Platform.DRONE: PlatformProfile(
        platform=Platform.DRONE,
        max_speed=8.0,
        capabilities=(
            Capability.AERIAL_VIEW,
            Capability.RF_DIRECTION_FIND,
            Capability.THERMAL_IMAGING,
            Capability.HIGH_GAIN_RADIO,
        ),
        reachable_fraction=1.0,
        # 85 mm baseline: poor resolution, but unobstructed look angles.
        df_sigma=0.21,
        signature=1.0,
        energy_per_m=7.5e-4,
    ),
    Platform.UGV: PlatformProfile(
        platform=Platform.UGV,
        max_speed=3.2,
        capabilities=(
            Capability.GROUND_MOBILITY,
            Capability.RF_DIRECTION_FIND,
            Capability.MARKER_EMPLACE,
            Capability.HIGH_GAIN_RADIO,
        ),
        reachable_fraction=0.62,
        # 300 mm baseline: the most accurate DF on the team.
        df_sigma=0.07,
        signature=0.55,
        energy_per_m=1.1e-4,
    ),
    Platform.QUADRUPED: PlatformProfile(
        platform=Platform.QUADRUPED,
        max_speed=1.5,
        capabilities=(
            Capability.GROUND_MOBILITY,
            Capability.ROUGH_TERRAIN,
            Capability.RF_DIRECTION_FIND,
            Capability.CBRN_SAMPLING,
        ),
        reachable_fraction=0.94,
        df_sigma=0.15,
        signature=0.28,
        energy_per_m=4.0e-4,
    ),
}


# Which capability a task type cannot be done without. A task may additionally
# carry its own required_capabilities mask; this is the type-level floor.
TASK_REQUIREMENTS: dict[TaskType, tuple[Capability, ...]] = {
    TaskType.SEARCH_CELL: (),
    TaskType.RF_TRIANGULATE: (Capability.RF_DIRECTION_FIND,),
    TaskType.VISUAL_CONFIRM: (Capability.THERMAL_IMAGING,),
    TaskType.CBRN_SAMPLE: (Capability.CBRN_SAMPLING,),
    TaskType.EMPLACE_JAMMER: (Capability.MARKER_EMPLACE,),
    TaskType.COMMS_RELAY: (Capability.HIGH_GAIN_RADIO,),
    TaskType.BDA: (Capability.THERMAL_IMAGING,),
}


def capability_vector(caps) -> list[float]:
    """One-hot-per-capability mask, in Capability order."""
    v = [0.0] * NUM_CAPABILITIES
    for c in caps:
        v[int(c)] = 1.0
    return v


def platform_capabilities(platform: Platform) -> list[float]:
    return capability_vector(PROFILES[Platform(platform)].capabilities)


def task_requirement_vector(task_type: TaskType) -> list[float]:
    return capability_vector(TASK_REQUIREMENTS[TaskType(task_type)])


# ============================ feature dimensions =============================
# Agent node features, in order:
#   x, y, z (scaled), vx, vy (scaled), sin(yaw), cos(yaw),
#   battery, health, max_speed (scaled), comms_quality, is_idle,
#   platform one-hot (3), capability mask (8)
AGENT_FEATURE_DIM = 12 + NUM_PLATFORMS + NUM_CAPABILITIES

# Task node features, in order:
#   x, y, z (scaled), priority, risk, service_time (scaled), radius (scaled),
#   urgency (deadline pressure), is_claimed,
#   task type one-hot (7), required capability mask (8)
TASK_FEATURE_DIM = 9 + NUM_TASK_TYPES + NUM_CAPABILITIES

# Edge features, in order: distance (scaled), 1/(1+d), capability match
# fraction, link-feasible flag, edge type one-hot (3)
EDGE_FEATURE_DIM = 4 + len(EdgeType)

# Scales that map world units into roughly [-1, 1] before the network sees them.
POSITION_SCALE = 60.0     # the sector half-width
SPEED_SCALE = 8.0         # the drone's envelope
TIME_SCALE = 60.0         # seconds
