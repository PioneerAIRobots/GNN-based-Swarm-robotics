"""Random mission scenarios for training and evaluating the allocator.

The scenarios are not uniform noise. They are drawn from the *situations the
mission actually produces*, because a policy trained on uniform random task
sets learns nothing about the structure it will face:

  * agents start clustered near an insertion corner more often than scattered,
    since that is how a mission begins;
  * threat tasks arrive in correlated bundles (a confirm and a CBRN sample for
    the same track, two triangulation baselines either side of one candidate),
    which is exactly the coupling the task-task edges exist to represent;
  * batteries and comms quality are sometimes degraded, so the policy sees the
    states where the cost trade-offs actually bite;
  * occasionally a platform is near-dead or jammed, which is where a myopic
    per-agent rule fails and relational reasoning pays off.
"""
from __future__ import annotations

import math
import random

from swarm_gnn.model.graph_builder import AgentView, TaskView
from swarm_gnn.model.spec import Capability, Platform, TaskType

SECTOR = 55.0


def sample_agents(rng: random.Random) -> list[AgentView]:
    """Three platforms, one of each kind, in a plausible configuration."""
    clustered = rng.random() < 0.35
    if clustered:
        # Early mission: everyone near the insertion point.
        cx, cy = rng.uniform(-SECTOR, -SECTOR + 15), rng.uniform(-SECTOR, -SECTOR + 15)
        spread = 10.0
    else:
        cx, cy = rng.uniform(-30, 30), rng.uniform(-30, 30)
        spread = 45.0

    def pos(z: float) -> tuple[float, float, float]:
        return (
            max(-SECTOR, min(SECTOR, cx + rng.gauss(0, spread))),
            max(-SECTOR, min(SECTOR, cy + rng.gauss(0, spread))),
            z,
        )

    agents = [
        AgentView("drone_1", Platform.DRONE, pos(rng.uniform(8.0, 18.0))),
        AgentView("ugv_1", Platform.UGV, pos(0.25), markers_remaining=rng.randint(0, 4)),
        AgentView("dog_1", Platform.QUADRUPED, pos(0.36)),
    ]

    for a in agents:
        a.yaw = rng.uniform(-math.pi, math.pi)
        speed = rng.uniform(0.0, 1.0) * a.max_speed
        a.velocity = (speed * math.cos(a.yaw), speed * math.sin(a.yaw), 0.0)
        # Mostly healthy, with a long tail of degraded states.
        a.battery = rng.uniform(0.12, 1.0) if rng.random() < 0.3 else rng.uniform(0.6, 1.0)
        a.health = 1.0 if rng.random() < 0.85 else rng.uniform(0.3, 0.9)
        a.comms_quality = 1.0 if rng.random() < 0.6 else rng.uniform(0.08, 0.95)

    # One platform occasionally in real trouble: the case where who-does-what
    # has to change, not just by how much.
    if rng.random() < 0.15:
        victim = rng.choice(agents)
        victim.battery = rng.uniform(0.04, 0.18)
        victim.health = rng.uniform(0.15, 0.45)

    return agents


def sample_tasks(rng: random.Random, agents: list[AgentView]) -> list[TaskView]:
    """A task set with the correlation structure a real tick would have."""
    tasks: list[TaskView] = []
    tid = 0

    def add(ttype: TaskType, x: float, y: float, z: float = 0.0, **kw) -> None:
        nonlocal tid
        tasks.append(TaskView(f"t{tid}", ttype, (x, y, z), **kw))
        tid += 1

    # --- search backdrop: always present during the sweep -----------------
    for _ in range(rng.randint(3, 9)):
        add(
            TaskType.SEARCH_CELL,
            rng.uniform(-SECTOR, SECTOR), rng.uniform(-SECTOR, SECTOR),
            radius=8.0, priority=rng.uniform(0.08, 0.55),
            risk=rng.uniform(0.02, 0.2), service_time=6.0,
        )

    # --- threat bundles: the correlated part ------------------------------
    for _ in range(rng.randint(0, 3)):
        ex, ey = rng.uniform(-SECTOR, SECTOR), rng.uniform(-SECTOR, SECTOR)
        stage = rng.random()

        if stage < 0.40:
            # A candidate: two triangulation baselines, placed off to the side.
            axis = rng.uniform(-math.pi, math.pi)
            standoff = rng.uniform(14.0, 32.0)
            for sign in (1.0, -1.0):
                add(
                    TaskType.RF_TRIANGULATE,
                    max(-SECTOR, min(SECTOR, ex + sign * standoff * math.cos(axis))),
                    max(-SECTOR, min(SECTOR, ey + sign * standoff * math.sin(axis))),
                    radius=6.0, priority=rng.uniform(0.4, 0.85),
                    risk=rng.uniform(0.15, 0.4), service_time=10.0,
                )
        elif stage < 0.80:
            # Localised: a thermal confirm and a CBRN sample on the same track.
            # These two must go to *different* platforms, and only the GNN sees
            # that the choice of one constrains the other.
            add(TaskType.VISUAL_CONFIRM, ex, ey, z=9.0,
                radius=9.0, priority=rng.uniform(0.6, 0.9),
                risk=rng.uniform(0.3, 0.6), service_time=12.0)
            add(TaskType.CBRN_SAMPLE, ex, ey,
                radius=2.2, priority=rng.uniform(0.6, 0.85),
                risk=rng.uniform(0.6, 0.95), service_time=18.0)
        else:
            # Confirmed: a marking task, which only the UGV can take, plus the
            # BDA that follows it.
            add(TaskType.EMPLACE_JAMMER, ex, ey,
                radius=3.0, priority=rng.uniform(0.85, 1.0),
                risk=rng.uniform(0.5, 0.85), service_time=15.0)
            if rng.random() < 0.5:
                add(TaskType.BDA, ex, ey, z=7.0,
                    radius=9.0, priority=rng.uniform(0.4, 0.7),
                    risk=rng.uniform(0.1, 0.3), service_time=8.0)

    # --- relay: open when the team is strung out --------------------------
    if len(agents) >= 2:
        worst = max(
            (
                math.dist(a.position[:2], b.position[:2])
                for i, a in enumerate(agents) for b in agents[i + 1:]
            ),
            default=0.0,
        )
        if worst > 60.0 or rng.random() < 0.12:
            mid_x = sum(a.position[0] for a in agents) / len(agents)
            mid_y = sum(a.position[1] for a in agents) / len(agents)
            add(TaskType.COMMS_RELAY, mid_x, mid_y, z=14.0,
                radius=8.0, priority=rng.uniform(0.45, 0.9),
                risk=rng.uniform(0.1, 0.35), service_time=20.0)

    # Sticky assignment: some agents are already committed, which the oracle
    # rewards continuing. Without this the policy never learns the hysteresis.
    if tasks and rng.random() < 0.5:
        for a in agents:
            if rng.random() < 0.6:
                a.current_task_id = rng.choice(tasks).task_id

    return tasks


def sample_scenario(rng: random.Random) -> tuple[list[AgentView], list[TaskView], float]:
    """(agents, tasks, mission_time). Guaranteed non-empty task set."""
    agents = sample_agents(rng)
    for _ in range(8):
        tasks = sample_tasks(rng, agents)
        if tasks:
            return agents, tasks, rng.uniform(0.0, 600.0)
    # Fall back to a single search cell rather than returning an empty graph.
    return agents, [TaskView("t0", TaskType.SEARCH_CELL, (0.0, 0.0, 0.0))], 0.0
