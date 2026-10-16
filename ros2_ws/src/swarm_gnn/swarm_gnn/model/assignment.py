"""Assignment solvers and the analytic cost oracle.

Pipeline
========
The GNN produces a dense utility matrix over feasible (agent, task) pairs. That
matrix is *not* an assignment: nothing in it prevents two agents from scoring
the same task highest. A combinatorial solver turns scores into a conflict-free
matching, which is where the Hungarian algorithm comes in.

So the division of labour is:
  GNN        learns *what a pairing is worth*, using relational context that no
             hand-written cost function captures (who else is nearby, whether
             this agent is the only comms bridge, how tasks cluster).
  Hungarian  enforces the *combinatorial constraint* optimally, given those
             values.

This is deliberately not a monolithic learned assignment: the solver gives a
hard feasibility and one-task-per-agent guarantee that a softmax policy cannot,
and it is what makes the learned part supervisable (see `oracle_cost_matrix`).

scipy is not used: the installed scipy is ABI-incompatible with the installed
numpy, and more importantly a 100-line dependency-free O(n^3) implementation is
easier to trust in a flight-critical loop than a transitive dependency.
"""
from __future__ import annotations

import math

import numpy as np

from .graph_builder import AgentView, TaskView, _euclidean
from .spec import PROFILES, Capability, Platform, TaskType

# Sentinel for a forbidden pairing. Large but finite, so the solver stays
# numerically well behaved; any assignment landing on one is dropped afterwards.
INFEASIBLE = 1.0e6


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rectangular linear-sum assignment, minimising total cost.

    Jonker-Volgenant shortest augmenting path with column reduction, the same
    algorithm scipy.optimize.linear_sum_assignment uses. O(n^2 m) for an
    n x m matrix with n <= m.

    Returns (row_idx, col_idx) such that cost[row_idx, col_idx].sum() is
    minimal and each index appears at most once. Rows are matched exhaustively
    when n <= m.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be 2-D")
    if cost.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if not np.isfinite(cost).all():
        raise ValueError("cost contains non-finite entries; use INFEASIBLE instead")

    # The algorithm below assumes rows <= cols; transpose and swap back if not.
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    # col4row[j] = row matched to column j, or -1. row4col[i] = column for row i.
    col4row = np.full(m, -1, dtype=int)
    row4col = np.full(n, -1, dtype=int)
    # Dual variables: u on rows, v on columns.
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(m, dtype=np.float64)

    for cur_row in range(n):
        # Shortest augmenting path from `cur_row` in the reduced cost graph.
        dist = cost[cur_row] - u[cur_row] - v
        pred = np.full(m, cur_row, dtype=int)
        scanned = np.zeros(m, dtype=bool)
        sink = -1
        delta = 0.0
        path_row = cur_row

        while sink == -1:
            unscanned = np.flatnonzero(~scanned)
            # Closest unscanned column.
            j = unscanned[np.argmin(dist[unscanned])]
            delta = dist[j]
            scanned[j] = True
            if col4row[j] == -1:
                sink = j
                break
            # Column j is taken: continue the path through its current row.
            path_row = col4row[j]
            remaining = np.flatnonzero(~scanned)
            if remaining.size:
                reduced = delta + cost[path_row, remaining] - u[path_row] - v[remaining]
                better = reduced < dist[remaining]
                improved = remaining[better]
                dist[improved] = reduced[better]
                pred[improved] = path_row

        # Update the duals along the scanned set, then augment.
        u[cur_row] += delta
        for i in range(n):
            if row4col[i] != -1 and scanned[row4col[i]]:
                j = row4col[i]
                u[i] += delta - dist[j]
                v[j] -= delta - dist[j]

        j = sink
        while True:
            i = pred[j]
            col4row[j] = i
            prev_j = row4col[i]
            row4col[i] = j
            if i == cur_row:
                break
            j = prev_j

    rows = np.arange(n)
    cols = row4col
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: repeatedly take the globally cheapest free pairing.

    Kept so the mission log can report how much the optimal solver is actually
    buying over the obvious heuristic; the allocator node publishes both totals.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    work = cost.copy()
    rows, cols = [], []
    for _ in range(min(work.shape)):
        flat = int(np.argmin(work))
        i, j = divmod(flat, work.shape[1])
        if work[i, j] >= INFEASIBLE:
            break
        rows.append(i)
        cols.append(j)
        work[i, :] = np.inf
        work[:, j] = np.inf
        # np.inf would break the argmin after every row is consumed; cap it.
        work[~np.isfinite(work)] = INFEASIBLE * 10
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


# ============================== the cost oracle ==============================
def travel_time(agent: AgentView, task: TaskView) -> float:
    """Straight-line transit time, inflated for platforms that cannot fly it.

    A road-bound UGV does not travel in straight lines through a built-up block,
    so its distance is multiplied by a detour factor derived from its reachable
    fraction. This is a coarse model, but it captures the ordering that matters:
    across a block, the drone is fastest; down an alley, the dog wins.
    """
    profile = PROFILES[agent.platform]
    d = _euclidean(agent.position, task.position)
    # reachable_fraction 1.0 -> no detour; 0.62 -> ~1.6x path inflation.
    detour = 1.0 / max(0.25, profile.reachable_fraction)
    if agent.platform == Platform.DRONE:
        detour = 1.0
    return (d * detour) / max(0.1, profile.max_speed * max(0.2, agent.health))


def pairing_cost(agent: AgentView, task: TaskView, mission_time: float = 0.0) -> float:
    """Analytic cost of giving `task` to `agent`. Lower is better.

    This is the supervision signal for training (see training/train_allocator.py)
    and the fallback when no checkpoint is available. It is a *myopic* cost: it
    scores one pairing in isolation and knows nothing about the rest of the
    team. That limitation is exactly what the GNN is there to fix === the oracle
    teaches the network the basics of cost, and the relational structure lets it
    learn the coupling the oracle cannot express.
    """
    profile = PROFILES[agent.platform]
    mask = agent.capability_mask()
    for i, need in enumerate(task.requirement_mask()):
        if need > 0.5 and mask[i] < 0.5:
            return INFEASIBLE

    if task.task_type == TaskType.EMPLACE_JAMMER and agent.markers_remaining <= 0:
        return INFEASIBLE

    t_travel = travel_time(agent, task)
    cost = t_travel + task.service_time

    # Priority discount: a high-priority task is worth travelling further for.
    cost *= 1.0 - 0.55 * task.priority

    # Risk: scaled by how detectable this platform is at that location. The dog
    # is cheap to send into a hot cell; the drone is not.
    cost += 42.0 * task.risk * profile.signature

    # Energy: a nearly-flat battery makes every job more expensive, steeply.
    energy = profile.energy_per_m * _euclidean(agent.position, task.position)
    reserve = max(0.05, agent.battery)
    cost += 160.0 * energy / reserve

    # Comms: an agent on a weak link is a poor choice for anything except
    # holding as a relay, which it is uniquely well placed to do.
    if task.task_type == TaskType.COMMS_RELAY:
        cost -= 28.0 * (1.0 - agent.comms_quality)
    else:
        cost += 20.0 * (1.0 - agent.comms_quality)

    # Deadline: missing one is worse than a long drive.
    if task.deadline > 0.0:
        arrival = mission_time + t_travel
        if arrival > task.deadline:
            cost += 55.0 + 4.0 * (arrival - task.deadline)

    # Specialist bonus: steer the scarce sensor to the task that needs it rather
    # than letting it soak up generic search cells.
    specialist = {
        TaskType.CBRN_SAMPLE: Capability.CBRN_SAMPLING,
        TaskType.EMPLACE_JAMMER: Capability.MARKER_EMPLACE,
        TaskType.VISUAL_CONFIRM: Capability.THERMAL_IMAGING,
        TaskType.BDA: Capability.THERMAL_IMAGING,
    }.get(task.task_type)
    if specialist is not None and specialist in profile.capabilities:
        cost -= 12.0

    # A generic search cell should not consume a specialist platform while
    # specialist work is outstanding; mild penalty keeps them free.
    if task.task_type == TaskType.SEARCH_CELL and len(profile.capabilities) >= 4:
        cost += 5.0

    # Sticky assignment: re-tasking an agent mid-leg wastes the transit already
    # flown, so continuing costs a little less than switching.
    if agent.current_task_id == task.task_id:
        cost -= 9.0

    return float(max(0.0, cost))


def oracle_cost_matrix(
    agents: list[AgentView], tasks: list[TaskView], mission_time: float = 0.0
) -> np.ndarray:
    """[A, T] analytic cost matrix, INFEASIBLE where the pairing is illegal."""
    if not agents or not tasks:
        return np.zeros((len(agents), len(tasks)), dtype=np.float64)
    return np.array(
        [[pairing_cost(a, t, mission_time) for t in tasks] for a in agents],
        dtype=np.float64,
    )


def solve_from_utility(
    utility: np.ndarray, feasible: np.ndarray | None = None
) -> list[tuple[int, int, float]]:
    """Hungarian over negated GNN utility, dropping infeasible matches.

    Returns a list of (agent_index, task_index, utility) triples.
    """
    utility = np.asarray(utility, dtype=np.float64)
    if utility.size == 0:
        return []

    cost = -utility.copy()
    # -inf utility (no edge) becomes +inf cost; swap for the finite sentinel the
    # solver can handle.
    mask = ~np.isfinite(cost)
    if feasible is not None:
        mask |= ~np.asarray(feasible, dtype=bool)
    cost[mask] = INFEASIBLE

    rows, cols = hungarian(cost)
    out = []
    for i, j in zip(rows, cols):
        if j < 0 or i < 0:
            continue
        if cost[i, j] >= INFEASIBLE:
            # Padding match onto a forbidden cell: the agent simply goes unassigned.
            continue
        out.append((int(i), int(j), float(utility[i, j])))
    return out
