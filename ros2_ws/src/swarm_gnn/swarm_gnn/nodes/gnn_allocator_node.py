#!/usr/bin/env python3
"""The coordination brain: GNN utility + Hungarian matching, at 2 Hz.

Each tick:
  1. snapshot the agent states and the open task set
  2. build the heterogeneous graph (capability-gated agent-task edges,
     range-gated agent-agent edges)
  3. run the GAT to get a utility per feasible pairing, plus the per-agent
     relay and caution outputs
  4. solve the assignment optimally with Hungarian over the negated utilities
  5. publish the assignment, the per-agent caution scalar, and a flattened
     snapshot of the graph with its learned attention for visualisation

Fallback behaviour is explicit: with no checkpoint the node runs on the analytic
cost oracle instead of the network, logs that it is doing so, and keeps flying
the mission. That matters because it makes the GNN's contribution measurable ===
the same mission can be run with `use_gnn:=false` and the two logged utility
totals compared directly.

Churn control
=============
A fresh optimal matching every 0.5 s re-tasks agents constantly, and an agent
that turns around every tick arrives nowhere. Measured: the drone retasked 188
times in 270 s and the sector was never swept, because a search cell is only
marked swept once its assigned agent reaches it.

A hysteresis margin does not fix this === the flip magnitude scales with the
same utilities the margin is measured in, so raising it just moves the
threshold. What fixes it is commitment: an agent keeps its task until the task
is gone, has become infeasible for it, or an unassigned mission-critical task
has no other capable agent. See `_tick`.
"""
from __future__ import annotations

import os
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node

# One thread, deliberately. torch defaults to one thread per core, but a ROS
# stack runs many processes on the same cores: with eight nodes up, torch's
# own pools oversubscribe a 2-core machine badly and the 7 ms forward pass
# measured in isolation stretched past 1.9 s in a live run, overrunning the
# 500 ms budget at 2 Hz. The graphs here are tiny (3 agents, tens of tasks),
# so intra-op parallelism buys nothing; letting the OS schedule whole nodes
# is strictly better.
torch.set_num_threads(1)
from std_msgs.msg import Float32MultiArray, String
from swarm_interfaces.msg import (
    AgentState,
    Assignment,
    AssignmentArray,
    GraphDebug,
    TaskArray,
)

from ..model.assignment import (
    INFEASIBLE,
    greedy_assignment,
    hungarian,
    oracle_cost_matrix,
    solve_from_utility,
)
from ..model.graph_builder import GraphConfig, build_graph
from ..model.het_gat import SwarmAllocatorGNN
from ..model.spec import Capability, EdgeType, PROFILES, Platform
from ..util.conversions import agent_state_to_view, task_spec_to_view
from ..util.qos import COORDINATION_QOS, SENSOR_QOS


class GNNAllocatorNode(Node):
    def __init__(self) -> None:
        super().__init__("gnn_allocator")

        self.declare_parameter("update_rate", 2.0)
        self.declare_parameter("use_gnn", True)
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("comms_range", 55.0)
        self.declare_parameter("aerial_range_bonus", 35.0)
        self.declare_parameter("task_cluster_range", 22.0)
        # Tasks at or above this priority are mission-critical and are
        # assigned before anything else; see `_solve_tiered`.
        self.declare_parameter("urgent_priority", 0.6)
        # Agent state older than this is treated as stale and the agent is left
        # out of the assignment rather than tasked on a dead reckoning.
        self.declare_parameter("agent_timeout", 4.0)
        self.declare_parameter("publish_graph_debug", True)
        self.declare_parameter("log_every_n_ticks", 20)
        # Must match the UGV's marker_dispenser capacity in its xacro.
        self.declare_parameter("marker_capacity", 4)

        self.cfg = GraphConfig(
            comms_range=float(self.get_parameter("comms_range").value),
            aerial_range_bonus=float(self.get_parameter("aerial_range_bonus").value),
            task_cluster_range=float(self.get_parameter("task_cluster_range").value),
        )

        self.agents: dict[str, tuple[AgentState, float]] = {}
        self.tasks: list = []
        self.current: dict[str, str] = {}       # agent_id -> task_id
        self._current_utility: dict[str, float] = {}
        #: EMPLACE_JAMMER task ids already handed out, i.e. markers spent.
        self._marks_issued: set[str] = set()
        #: Tasks agents have reported finished. A commitment is only as good
        #: as its release condition: without this an agent that had done the
        #: work kept the assignment and never got new orders.
        self._completed: set[str] = set()
        self._tick_count = 0
        self._mission_start = self.get_clock().now().nanoseconds * 1e-9

        self.model: SwarmAllocatorGNN | None = None
        self.model_source = "oracle"
        if bool(self.get_parameter("use_gnn").value):
            self._load_model()
        else:
            self.get_logger().warn(
                "use_gnn:=false === running on the analytic cost oracle, "
                "no learned coordination"
            )

        self.create_subscription(AgentState, "/swarm/agent_states", self._on_agent, SENSOR_QOS)
        self.create_subscription(TaskArray, "/swarm/tasks", self._on_tasks, COORDINATION_QOS)
        self.create_subscription(
            String, "/swarm/task_complete", self._on_task_complete, COORDINATION_QOS
        )

        self.assignment_pub = self.create_publisher(
            AssignmentArray, "/swarm/assignments", COORDINATION_QOS
        )
        self.graph_pub = self.create_publisher(GraphDebug, "/swarm/graph_debug", SENSOR_QOS)
        # Per-agent caution scalars, in the order the agents were solved; the
        # controllers read their own entry by matching /swarm/caution_order.
        self.caution_pub = self.create_publisher(
            Float32MultiArray, "/swarm/caution", SENSOR_QOS
        )
        self.caution_order_pub = self.create_publisher(
            String, "/swarm/caution_order", SENSOR_QOS
        )

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(0.5, rate), self._tick)

        self.get_logger().info(
            f"gnn_allocator up: solver=hungarian, utility={self.model_source}, "
            f"{rate:.1f} Hz"
        )

    # ------------------------------------------------------------------ setup
    def _load_model(self) -> None:
        path = str(self.get_parameter("checkpoint").value)
        if not path:
            self.get_logger().warn(
                "no checkpoint parameter set === falling back to the analytic cost "
                "oracle. Train one with: ros2 run swarm_gnn train_allocator"
            )
            return
        if not os.path.isfile(path):
            self.get_logger().error(
                f"checkpoint '{path}' not found === falling back to the analytic "
                "cost oracle"
            )
            return
        try:
            self.model, extra = SwarmAllocatorGNN.load(path)
            self.model_source = f"gnn({os.path.basename(path)})"
            self.get_logger().info(
                f"loaded {self.model_source}: "
                f"{sum(p.numel() for p in self.model.parameters())} parameters, "
                f"trained {extra.get('epochs', '?')} epochs, "
                f"val top-1 {extra.get('val_top1', float('nan')):.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - must not take the swarm down
            self.get_logger().error(
                f"failed to load checkpoint '{path}': {exc} === falling back to the "
                "analytic cost oracle"
            )
            self.model = None

    # ------------------------------------------------------------- callbacks
    def _on_agent(self, msg: AgentState) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.agents[msg.agent_id] = (msg, now)

    def _on_tasks(self, msg: TaskArray) -> None:
        self.tasks = list(msg.tasks)
        # Forget completions for tasks the executive has withdrawn, so the set
        # does not grow without bound over a long mission.
        live = {t.task_id for t in msg.tasks}
        self._completed &= live

    def _on_task_complete(self, msg: String) -> None:
        parts = msg.data.split(":", 1)
        if len(parts) != 2:
            self.get_logger().warn(f"malformed completion: '{msg.data}'")
            return
        agent_id, task_id = parts
        self._completed.add(task_id)
        if self.current.get(agent_id) == task_id:
            # Drop it now rather than waiting for the next solve, so the agent
            # is free in this tick.
            self.current.pop(agent_id, None)

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        self._tick_count += 1
        t_start = time.perf_counter()

        now = self.get_clock().now().nanoseconds * 1e-9
        timeout = float(self.get_parameter("agent_timeout").value)
        live = [
            (aid, state) for aid, (state, stamp) in sorted(self.agents.items())
            if now - stamp <= timeout
        ]
        if not live or not self.tasks:
            return

        agent_ids = [aid for aid, _ in live]
        views = []
        for aid, state in live:
            # Markers remaining is a resource the graph gates on; the UGV
            # reports it in its agent state's battery-adjacent fields, so it is
            # reconstructed here from the platform profile and what has been used.
            markers = self._markers_remaining(state)
            view = agent_state_to_view(state, markers_remaining=markers)
            # Use the allocator's own view of who holds what, which is
            # authoritative; an agent's self-report can lag by a tick.
            view.current_task_id = self.current.get(aid, "")
            views.append(view)

        task_views = [task_spec_to_view(t) for t in self.tasks]
        mission_time = now - self._mission_start

        graph = build_graph(views, task_views, self.cfg, mission_time)

        # ======================= utility =====================================
        if self.model is not None:
            with torch.no_grad():
                out = self.model(graph)
            utility = out["utility"].numpy()
            relay = out["relay"].numpy()
            caution = out["caution"].numpy()
            attention = out["attention"]
            solver_name = "gnn+hungarian"
        else:
            # Oracle fallback: utility is negated cost, put on a comparable
            # scale so the switch margin means roughly the same thing either way.
            cost = oracle_cost_matrix(views, task_views, mission_time)
            utility = -cost / 20.0
            utility[~graph["feasible"].numpy()] = -np.inf
            relay = np.full(len(views), 0.5, dtype=np.float32)
            caution = np.full(len(views), 0.85, dtype=np.float32)
            attention = {}
            solver_name = "oracle+hungarian"

        # ====================== commitment ===================================
        # An agent keeps the task it is working on. This is not a tie-breaker,
        # it is the fix for a real failure: with the assignment re-auctioned
        # from scratch every tick, the drone retasked 188 times in 270 s. Its
        # utilities are position-sensitive and it covers 4 m per tick, so the
        # argmax kept flipping; and because a search cell is only marked swept
        # once its assigned agent reaches it, nothing was ever swept and the
        # same cells kept coming back. Raising the hysteresis margin did not
        # help, because the flip magnitude scales with the same utilities the
        # margin is measured in.
        #
        # So a commitment is only broken when it stops being valid: the task
        # has gone (completed, or the executive withdrew it), it has become
        # infeasible for that agent, or the agent is the only one capable of an
        # unassigned mission-critical task. Everything else is auctioned
        # normally, which is still where the GNN does its work.
        task_index = {t.task_id: k for k, t in enumerate(task_views)}
        threshold = float(self.get_parameter("urgent_priority").value)

        committed: dict[int, int] = {}
        for i, aid in enumerate(agent_ids):
            held = self.current.get(aid, "")
            k = task_index.get(held)
            if (
                k is not None
                and held not in self._completed
                and graph["feasible"][i, k]
                and np.isfinite(utility[i, k])
            ):
                committed[i] = k

        # Preemption: an unassigned urgent task whose only capable agents are
        # all committed to routine work takes one of them.
        urgent_unassigned = [
            k for k, t in enumerate(task_views)
            if t.priority >= threshold and k not in committed.values()
        ]
        for k in urgent_unassigned:
            capable = [
                i for i in range(len(agent_ids))
                if graph["feasible"][i, k] and np.isfinite(utility[i, k])
            ]
            if not capable:
                continue
            if any(i not in committed for i in capable):
                continue  # a free agent can take it in the solve below
            # All capable agents are busy: preempt the one on the least
            # important work, by that task's priority.
            victim = min(
                capable, key=lambda i: task_views[committed[i]].priority
            )
            if task_views[committed[victim]].priority < threshold:
                self.get_logger().info(
                    f"{agent_ids[victim]} preempted off "
                    f"{task_views[committed[victim]].task_id} for urgent "
                    f"{task_views[k].task_id}"
                )
                del committed[victim]

        matches: list[tuple[int, int, float]] = [
            (i, k, float(utility[i, k])) for i, k in committed.items()
        ]

        # ====================== solve the remainder ==========================
        free_agents = [i for i in range(len(agent_ids)) if i not in committed]
        held_tasks = set(committed.values())
        open_tasks = [
            k for k in range(len(task_views))
            if k not in held_tasks and task_views[k].task_id not in self._completed
        ]
        if free_agents and open_tasks:
            sub_u = utility[np.ix_(free_agents, open_tasks)]
            sub_f = graph["feasible"].numpy()[np.ix_(free_agents, open_tasks)]
            sub_tasks = [task_views[k] for k in open_tasks]
            for li, lk, util in self._solve_tiered(sub_u, sub_f, sub_tasks):
                matches.append((free_agents[li], open_tasks[lk], util))

        # ================== comparison baselines =============================
        # Logged, not acted on: this is how the mission report can say what the
        # optimal solver and the learned utility are each worth.
        cost_for_baseline = -np.where(np.isfinite(utility), utility, -INFEASIBLE)
        cost_for_baseline[~np.isfinite(utility)] = INFEASIBLE
        g_rows, g_cols = greedy_assignment(cost_for_baseline)
        greedy_total = float(
            sum(utility[i, j] for i, j in zip(g_rows, g_cols) if np.isfinite(utility[i, j]))
        )
        total_utility = float(sum(u for _, _, u in matches))

        # ========================= publish ===================================
        assignments = []
        new_current: dict[str, str] = {}
        for i, k, util in matches:
            aid = agent_ids[i]
            task = task_views[k]
            new_current[aid] = task.task_id
            if task.task_id.endswith("_mark"):
                self._marks_issued.add(task.task_id)
            a = Assignment()
            a.agent_id = aid
            a.task_id = task.task_id
            a.utility = float(util)
            from ..model.assignment import pairing_cost

            a.oracle_cost = float(pairing_cost(views[i], task, mission_time))
            assignments.append(a)

        changed = {
            aid for aid in set(new_current) | set(self.current)
            if new_current.get(aid) != self.current.get(aid)
        }
        self.current = new_current

        msg = AssignmentArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.assignments = assignments
        msg.solver = solver_name
        msg.solve_time_ms = float((time.perf_counter() - t_start) * 1000.0)
        msg.total_utility = total_utility
        self.assignment_pub.publish(msg)

        self.caution_order_pub.publish(String(data=",".join(agent_ids)))
        self.caution_pub.publish(
            Float32MultiArray(data=[float(c) for c in caution])
        )

        if bool(self.get_parameter("publish_graph_debug").value):
            self._publish_graph_debug(agent_ids, task_views, graph, attention)

        if changed:
            detail = ", ".join(
                f"{aid}->{new_current.get(aid, 'idle')}" for aid in sorted(changed)
            )
            self.get_logger().info(
                f"[t={mission_time:6.1f}s] retask: {detail} "
                f"({len(task_views)} open, U={total_utility:.2f}, "
                f"{msg.solve_time_ms:.1f} ms)"
            )

        every = int(self.get_parameter("log_every_n_ticks").value)
        if every > 0 and self._tick_count % every == 0:
            self.get_logger().info(
                f"[t={mission_time:6.1f}s] {len(agent_ids)}a/{len(task_views)}t  "
                f"edges aa={graph['aa_index'].shape[1]} "
                f"at={graph['at_index'].shape[1]} tt={graph['tt_index'].shape[1]}  "
                f"U_hungarian={total_utility:.2f} U_greedy={greedy_total:.2f}  "
                f"relay={np.round(relay, 2).tolist()} "
                f"caution={np.round(caution, 2).tolist()}"
            )

    def _solve_tiered(
        self, utility: np.ndarray, feasible: np.ndarray, task_views: list
    ) -> list[tuple[int, int, float]]:
        """Assign mission-critical tasks first, then the rest.

        Hungarian maximises total utility over the whole matrix, which is the
        right objective *within* a class of comparable work. It is the wrong
        objective across classes: three cheap, nearby search cells can
        out-score one distant thermal confirm, and the matching then leaves a
        localised emitter uncorroborated while the team tidies up coverage.
        Observed exactly that === a confirm task sat unassigned for 165 s while
        the drone worked search cells.

        Priority is a mission statement, not a preference to be traded off, so
        it is enforced structurally: solve the urgent tier against every
        capable agent, then solve whoever is left against the routine tier.
        Each stage is still an optimal assignment, and the GNN's utilities
        still decide *who* takes *which* urgent task === which is the
        relational judgement it is actually good at.
        """
        threshold = float(self.get_parameter("urgent_priority").value)
        urgent = [k for k, t in enumerate(task_views) if t.priority >= threshold]
        routine = [k for k, t in enumerate(task_views) if t.priority < threshold]

        if not urgent or not routine:
            return solve_from_utility(utility, feasible)

        matches: list[tuple[int, int, float]] = []
        n_agents = utility.shape[0]

        stage1 = solve_from_utility(utility[:, urgent], feasible[:, urgent])
        taken_agents = set()
        for i, local_k, util in stage1:
            matches.append((i, urgent[local_k], util))
            taken_agents.add(i)

        free_agents = [i for i in range(n_agents) if i not in taken_agents]
        if free_agents and routine:
            sub_u = utility[np.ix_(free_agents, routine)]
            sub_f = feasible[np.ix_(free_agents, routine)]
            for local_i, local_k, util in solve_from_utility(sub_u, sub_f):
                matches.append((free_agents[local_i], routine[local_k], util))

        return matches

    def _markers_remaining(self, state: AgentState) -> int:
        """How many jammer-markers this platform still carries.

        The dispenser capacity comes from the URDF's marker_dispenser macro;
        expenditure is tracked in `self._marks_issued`, which this node updates
        as it hands out EMPLACE_JAMMER tasks. Once the count hits capacity the
        graph builder stops creating marking edges for this platform at all, so
        the executive's marking tasks simply go unassigned rather than being
        promised to a robot with an empty dispenser.
        """
        if Capability.MARKER_EMPLACE not in PROFILES[Platform(state.platform)].capabilities:
            return 0
        capacity = int(self.get_parameter("marker_capacity").value)
        return max(0, capacity - len(self._marks_issued))

    # -------------------------------------------------------- graph debug
    def _publish_graph_debug(self, agent_ids, task_views, graph, attention) -> None:
        """Flatten the graph and its attention for the visualiser.

        Node indices are agents first, then tasks offset by num_agent_nodes,
        which is the convention GraphDebug documents.
        """
        msg = GraphDebug()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.agent_ids = list(agent_ids)
        msg.task_ids = [t.task_id for t in task_views]
        msg.num_agent_nodes = len(agent_ids)

        src: list[int] = []
        dst: list[int] = []
        weight: list[float] = []
        etype: list[int] = []
        att: list[float] = []
        offset = len(agent_ids)

        def mean_attention(key: str, count: int) -> list[float]:
            """Attention averaged over heads, or 1.0 when running on the oracle."""
            tensor = attention.get(key) if attention else None
            if tensor is None or tensor.numel() == 0 or tensor.shape[0] != count:
                return [1.0] * count
            return tensor.mean(dim=-1).tolist()

        for key, index_key, attr_key, edge_type, off_src, off_dst in (
            ("aa", "aa_index", "aa_attr", EdgeType.AGENT_AGENT, 0, 0),
            ("ta", "at_index", "at_attr", EdgeType.AGENT_TASK, 0, offset),
            ("tt", "tt_index", "tt_attr", EdgeType.TASK_TASK, offset, offset),
        ):
            idx = graph[index_key]
            n = idx.shape[1]
            if n == 0:
                continue
            attrs = graph[attr_key]
            means = mean_attention(key, n)
            for e in range(n):
                src.append(int(idx[0, e]) + off_src)
                dst.append(int(idx[1, e]) + off_dst)
                # Edge weight is the inverse-distance feature, which is what a
                # viewer wants to render as link strength.
                weight.append(float(attrs[e, 1]))
                etype.append(int(edge_type))
                att.append(float(means[e]))

        msg.edge_src = src
        msg.edge_dst = dst
        msg.edge_weight = weight
        msg.edge_type = etype
        msg.attention = att
        self.graph_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GNNAllocatorNode()
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
