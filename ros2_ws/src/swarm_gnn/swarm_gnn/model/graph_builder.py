"""Turn the swarm's belief state into the tensors the GNN consumes.

The graph is rebuilt from scratch every allocation tick (2 Hz). That is cheap at
these sizes and it is the right design: tasks appear and close continuously as
the Bayesian localiser promotes candidates, and the comms graph rewires as the
platforms move, so there is no stable structure worth caching.

Two gating rules matter, because they are what let the network exploit
structure instead of having to learn it:

  * An agent-task edge exists only if the agent is *capability-feasible* for
    that task. Infeasible pairings get no edge, so they receive no utility
    score at all, and the solver can never pick them. This is a hard constraint
    expressed in the topology rather than a soft penalty the network has to
    discover.

  * An agent-agent edge exists only within radio range. Messages therefore flow
    exactly along the real comms topology, which is what makes the learned
    policy decentralisation-compatible: an agent's embedding depends only on
    peers it could actually have heard from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .spec import (
    AGENT_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    NUM_CAPABILITIES,
    NUM_PLATFORMS,
    NUM_TASK_TYPES,
    POSITION_SCALE,
    PROFILES,
    SPEED_SCALE,
    TASK_FEATURE_DIM,
    TIME_SCALE,
    Capability,
    EdgeType,
    Platform,
    TaskType,
    task_requirement_vector,
)


@dataclass
class AgentView:
    """Plain-python mirror of swarm_interfaces/AgentState.

    The ROS nodes convert messages into these; the training script samples them
    directly. Keeping the model's input independent of ROS means the whole
    coordination layer trains and unit-tests without a ROS installation.
    """

    agent_id: str
    platform: Platform
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    battery: float = 1.0
    health: float = 1.0
    comms_quality: float = 1.0
    capabilities: tuple[float, ...] = ()
    current_task_id: str = ""
    # Consumable resources, e.g. jammer-markers remaining on the UGV.
    markers_remaining: int = 0

    @property
    def max_speed(self) -> float:
        return PROFILES[self.platform].max_speed

    def capability_mask(self) -> tuple[float, ...]:
        if self.capabilities:
            return self.capabilities
        from .spec import platform_capabilities

        return tuple(platform_capabilities(self.platform))


@dataclass
class TaskView:
    """Plain-python mirror of swarm_interfaces/TaskSpec."""

    task_id: str
    task_type: TaskType
    position: tuple[float, float, float]
    radius: float = 3.0
    priority: float = 0.5
    risk: float = 0.2
    service_time: float = 10.0
    deadline: float = -1.0
    required_capabilities: tuple[float, ...] = ()
    claimed_by: str = ""

    def requirement_mask(self) -> tuple[float, ...]:
        """Type-level floor, OR'd with any task-specific extras."""
        base = list(task_requirement_vector(self.task_type))
        for i, v in enumerate(self.required_capabilities[:NUM_CAPABILITIES]):
            if v:
                base[i] = 1.0
        return tuple(base)


@dataclass
class GraphConfig:
    """Topology gating thresholds.

    `comms_range` is the nominal radio range in open ground; the drone's
    altitude advantage is folded in by `aerial_range_bonus`. `task_link_range`
    bounds how far an agent will even consider bidding, which keeps the graph
    sparse in a large sector.
    """

    comms_range: float = 55.0
    aerial_range_bonus: float = 35.0
    # Must exceed the sector diagonal (156 m for a 110 m square), otherwise a
    # corner-to-corner pairing is silently unrepresentable in the graph while
    # still looking feasible to the analytic cost oracle.
    task_link_range: float = 200.0
    task_cluster_range: float = 22.0   # task-task edge threshold
    max_task_neighbours: int = 6
    # Below this capability-match fraction an agent-task edge is not created.
    # 1.0 means every required capability must be present.
    min_capability_match: float = 1.0


def _capability_match(agent_mask, req_mask) -> float:
    """Fraction of the task's required capabilities the agent actually has."""
    required = [i for i, v in enumerate(req_mask) if v > 0.5]
    if not required:
        return 1.0
    have = sum(1.0 for i in required if agent_mask[i] > 0.5)
    return have / len(required)


def _euclidean(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def agent_features(agent: AgentView, mission_time: float = 0.0) -> list[float]:
    x, y, z = agent.position
    vx, vy, _ = agent.velocity
    caps = agent.capability_mask()
    platform_onehot = [0.0] * NUM_PLATFORMS
    platform_onehot[int(agent.platform)] = 1.0
    return [
        x / POSITION_SCALE,
        y / POSITION_SCALE,
        z / POSITION_SCALE,
        vx / SPEED_SCALE,
        vy / SPEED_SCALE,
        math.sin(agent.yaw),
        math.cos(agent.yaw),
        agent.battery,
        agent.health,
        agent.max_speed / SPEED_SCALE,
        agent.comms_quality,
        1.0 if not agent.current_task_id else 0.0,
        *platform_onehot,
        *caps,
    ]


def task_features(task: TaskView, mission_time: float = 0.0) -> list[float]:
    x, y, z = task.position
    type_onehot = [0.0] * NUM_TASK_TYPES
    type_onehot[int(task.task_type)] = 1.0

    if task.deadline > 0.0:
        # Urgency rises as the deadline approaches and saturates at 1.
        slack = max(0.0, task.deadline - mission_time)
        urgency = 1.0 / (1.0 + slack / TIME_SCALE)
    else:
        urgency = 0.0

    return [
        x / POSITION_SCALE,
        y / POSITION_SCALE,
        z / POSITION_SCALE,
        task.priority,
        task.risk,
        task.service_time / TIME_SCALE,
        task.radius / POSITION_SCALE,
        urgency,
        1.0 if task.claimed_by else 0.0,
        *type_onehot,
        *task.requirement_mask(),
    ]


def _edge_attr(distance: float, cap_match: float, feasible: bool, etype: EdgeType) -> list[float]:
    onehot = [0.0] * len(EdgeType)
    onehot[int(etype)] = 1.0
    return [
        distance / POSITION_SCALE,
        1.0 / (1.0 + distance / POSITION_SCALE),
        cap_match,
        1.0 if feasible else 0.0,
        *onehot,
    ]


def comms_range_between(a: AgentView, b: AgentView, cfg: GraphConfig) -> float:
    """Effective link range for a pair, accounting for aerial line of sight."""
    rng = cfg.comms_range
    if a.platform == Platform.DRONE or b.platform == Platform.DRONE:
        rng += cfg.aerial_range_bonus
    return rng


def build_graph(
    agents: list[AgentView],
    tasks: list[TaskView],
    cfg: GraphConfig | None = None,
    mission_time: float = 0.0,
    device: torch.device | str = "cpu",
) -> dict:
    """Assemble the tensor dict consumed by SwarmAllocatorGNN.forward.

    The returned dict also carries `feasible` (a dense [A, T] bool mask) and the
    index lists, which the solver and the GraphDebug publisher both need.
    """
    cfg = cfg or GraphConfig()
    n_a, n_t = len(agents), len(tasks)

    agent_x = torch.tensor(
        [agent_features(a, mission_time) for a in agents], dtype=torch.float32, device=device
    ) if n_a else torch.zeros((0, AGENT_FEATURE_DIM), dtype=torch.float32, device=device)

    task_x = torch.tensor(
        [task_features(t, mission_time) for t in tasks], dtype=torch.float32, device=device
    ) if n_t else torch.zeros((0, TASK_FEATURE_DIM), dtype=torch.float32, device=device)

    # ========================= agent-agent (comms) ==========================
    aa_src: list[int] = []
    aa_dst: list[int] = []
    aa_attr: list[list[float]] = []
    for i, a in enumerate(agents):
        # Self-loop: keeps a node's own state in the aggregate even when it is
        # isolated, which otherwise zeroes the message and destabilises the
        # residual update.
        aa_src.append(i)
        aa_dst.append(i)
        aa_attr.append(_edge_attr(0.0, 1.0, True, EdgeType.AGENT_AGENT))
        for j, b in enumerate(agents):
            if i == j:
                continue
            d = _euclidean(a.position, b.position)
            if d <= comms_range_between(a, b, cfg):
                aa_src.append(j)   # message flows j -> i
                aa_dst.append(i)
                aa_attr.append(_edge_attr(d, 1.0, True, EdgeType.AGENT_AGENT))

    # ===================== agent-task (candidate bids) ======================
    at_src: list[int] = []
    at_dst: list[int] = []
    at_attr: list[list[float]] = []
    feasible = torch.zeros((n_a, n_t), dtype=torch.bool, device=device)
    for i, a in enumerate(agents):
        mask = a.capability_mask()
        for k, t in enumerate(tasks):
            match = _capability_match(mask, t.requirement_mask())
            if match < cfg.min_capability_match:
                continue
            # The UGV cannot emplace a marker it does not have left.
            if (
                t.task_type == TaskType.EMPLACE_JAMMER
                and Capability.MARKER_EMPLACE in PROFILES[a.platform].capabilities
                and a.markers_remaining <= 0
            ):
                continue
            d = _euclidean(a.position, t.position)
            if d > cfg.task_link_range:
                continue
            at_src.append(i)
            at_dst.append(k)
            at_attr.append(_edge_attr(d, match, True, EdgeType.AGENT_TASK))
            feasible[i, k] = True

    # ==================== task-task (spatial clustering) ====================
    tt_src: list[int] = []
    tt_dst: list[int] = []
    tt_attr: list[list[float]] = []
    for k, t in enumerate(tasks):
        # k-nearest within the cluster radius, so dense regions do not blow the
        # edge count up quadratically.
        neighbours = sorted(
            (
                (_euclidean(t.position, u.position), m)
                for m, u in enumerate(tasks)
                if m != k
            )
        )[: cfg.max_task_neighbours]
        for d, m in neighbours:
            if d <= cfg.task_cluster_range:
                tt_src.append(m)
                tt_dst.append(k)
                tt_attr.append(_edge_attr(d, 1.0, True, EdgeType.TASK_TASK))

    def _idx(src, dst):
        if not src:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    def _attr(rows):
        if not rows:
            return torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)
        return torch.tensor(rows, dtype=torch.float32, device=device)

    return {
        "agent_x": agent_x,
        "task_x": task_x,
        "aa_index": _idx(aa_src, aa_dst),
        "aa_attr": _attr(aa_attr),
        "at_index": _idx(at_src, at_dst),
        "at_attr": _attr(at_attr),
        "tt_index": _idx(tt_src, tt_dst),
        "tt_attr": _attr(tt_attr),
        "feasible": feasible,
        "num_agents": n_a,
        "num_tasks": n_t,
    }
