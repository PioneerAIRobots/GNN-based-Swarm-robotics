#!/usr/bin/env python3
"""OPERATION SENTINEL GRID === the mission executive.

Owns the mission phase machine and, from it, the *task set*. It does not decide
who does what === that is the allocator's job. The division is the point: the
executive answers "what does the mission need right now", the GNN answers "who
should do it", and the two are coupled only through the /swarm/tasks topic. That
is what lets the allocation policy be retrained without touching mission logic.

Phases
======
  INIT        wait for all platforms to report in
  INGRESS     move off the insertion point onto the corridor
  SEARCH      sweep the sector; Lloyd partition seeds the cell task set
  LOCALIZE    a candidate exists but is not tight enough; spend triangulation
              baselines on it (this phase overlaps SEARCH and is re-entrant)
  CONFIRM     a localized emitter needs two independent sensor corroborations
  NEUTRALIZE  a confirmed hostile emitter gets a marker emplaced
  BDA         verify the emitter went silent
  EGRESS      recover to the rally point
  COMPLETE

The phases are not a strict sequence. SEARCH, LOCALIZE and CONFIRM run
concurrently whenever the threat picture warrants it, because waiting for a
clean phase boundary would waste the two platforms not involved in the current
step. The reported phase is the *most advanced* activity in progress, which is
what an operator wants to see.

Task generation rules
=====================
Every task the executive emits is traceable to a rule here, which is what makes
the system auditable:

  SEARCH_CELL     one per unswept cell, priority from the cell's prior weight
  RF_TRIANGULATE  for a candidate whose baseline spread is too small to resolve
                  range; placed *off to the side* of the current cut geometry,
                  not at the emitter, because what is needed is a new angle
  VISUAL_CONFIRM  for a localized track, at thermal standoff
  CBRN_SAMPLE     for a localized track suspected of being a trigger
  EMPLACE_JAMMER  for a confirmed hostile track, UGV only
  COMMS_RELAY     when the comms graph is about to partition
  BDA             after a marker is emplaced
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import String
from swarm_interfaces.msg import (
    AgentState,
    AssignmentArray,
    MissionStatus,
    TaskArray,
    TaskSpec,
    ThreatArray,
    ThreatEstimate,
)

from ..model.coverage import (
    SearchCell,
    build_search_cells,
    coverage_fraction,
    lloyd_partition,
)
from ..model import sector
from ..model.graph_builder import GraphConfig, comms_range_between
from ..model.spec import Capability, Platform, TaskType
from ..util.conversions import agent_state_to_view
from ..util.qos import COORDINATION_QOS, SENSOR_QOS, WORLD_MODEL_QOS

_PHASE_NAMES = {
    MissionStatus.PHASE_INIT: "INIT",
    MissionStatus.PHASE_INGRESS: "INGRESS",
    MissionStatus.PHASE_SEARCH: "SEARCH",
    MissionStatus.PHASE_LOCALIZE: "LOCALIZE",
    MissionStatus.PHASE_CONFIRM: "CONFIRM",
    MissionStatus.PHASE_NEUTRALIZE: "NEUTRALIZE",
    MissionStatus.PHASE_BDA: "BDA",
    MissionStatus.PHASE_EGRESS: "EGRESS",
    MissionStatus.PHASE_COMPLETE: "COMPLETE",
}


class MissionExecutiveNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_executive")

        self.declare_parameter("update_rate", 2.0)
        self.declare_parameter("expected_agents", ["drone_1", "ugv_1", "dog_1"])
        self.declare_parameter("sector_half_extent", 55.0)
        self.declare_parameter("search_cell_size", 11.0)
        self.declare_parameter("corridor_half_width", 7.0)
        # (x, y, radius) triples the UGV cannot reach, flattened.
        self.declare_parameter("ugv_exclusions", [44.0, 1.5, 9.0, 47.5, -2.0, 9.0, -2.0, -27.0, 8.0])
        self.declare_parameter("rally_point", [-52.0, 52.0])
        self.declare_parameter("ingress_waypoint", [-40.0, -40.0])
        # A cell counts as swept when an agent assigned to it gets this close.
        self.declare_parameter("sweep_radius", 8.0)
        self.declare_parameter("cruise_altitude", 14.0)
        self.declare_parameter("confirm_standoff", 9.0)
        self.declare_parameter("sample_standoff", 2.2)
        self.declare_parameter("max_concurrent_search_tasks", 9)
        self.declare_parameter("coverage_target", 1.0)
        # Classes the team is authorised to mark. The civilian transmitter is
        # not on this list, and that exclusion is the mission's rules of
        # engagement expressed in one line.
        self.declare_parameter("engageable_classes", [1, 2])
        # Seconds between operator progress lines. Without these the only
        # visible output is a stream of retask messages, from which it is
        # impossible to tell whether the mission is advancing or stuck.
        self.declare_parameter("progress_log_period", 15.0)
        # Safety valve. The sweep objective is a hard one, so there has to be
        # a stated limit after which the team egresses and reports what it did
        # not reach, rather than staying in the sector indefinitely.
        self.declare_parameter("max_mission_time", 1200.0)

        half = float(self.get_parameter("sector_half_extent").value)
        # Rubble exclusions come from the shared layout rather than a
        # hand-maintained parameter list that could disagree with it.
        exclusions = sector.UGV_EXCLUSIONS
        self.cells: list[SearchCell] = build_search_cells(
            half_extent=half,
            cell_size=float(self.get_parameter("search_cell_size").value),
            corridor_half_width=float(self.get_parameter("corridor_half_width").value),
            rubble_exclusions=exclusions,
        )
        # A cell centre that lands inside a building can never be reached, so
        # the cell would sit in the task set forever and hold the coverage
        # fraction permanently below the completion target. Relocating the
        # observation point to reachable ground *within the same cell* is
        # better than discarding the cell: 27 of 49 centres fall in blocks on
        # this layout, and dropping them would collapse coverage onto the two
        # routes and leave most of the sector unswept.
        self.cells, dropped = self._make_cells_reachable(self.cells)

        self.agents: dict[str, AgentState] = {}
        self.threats: dict[str, ThreatEstimate] = {}
        self.assignments: dict[str, str] = {}      # agent_id -> task_id
        self.regions: dict[str, list[SearchCell]] = {}
        self.phase = MissionStatus.PHASE_INIT
        self.narrative = "awaiting platform check-in"
        self._start_time: float | None = None
        self._marked: set[str] = set()
        #: Latched so the relay task does not flap; see `_relay_tasks`.
        self._relay_open = False
        #: Tasks the team has finished. Triangulation and relay tasks are
        #: regenerated from world state every tick, so without this they would
        #: reappear the instant they were serviced and the agent would be sent
        #: straight back to work it had already done.
        self._completed_tasks: set[str] = set()
        self._bda_done: set[str] = set()
        self._tick_count = 0

        self.create_subscription(AgentState, "/swarm/agent_states", self._on_agent, SENSOR_QOS)
        self.create_subscription(ThreatArray, "/swarm/threats", self._on_threats, WORLD_MODEL_QOS)
        self.create_subscription(
            AssignmentArray, "/swarm/assignments", self._on_assignments, COORDINATION_QOS
        )
        self.create_subscription(
            String, "/swarm/task_complete", self._on_task_complete, COORDINATION_QOS
        )

        self.task_pub = self.create_publisher(TaskArray, "/swarm/tasks", COORDINATION_QOS)
        self.status_pub = self.create_publisher(
            MissionStatus, "/swarm/mission_status", COORDINATION_QOS
        )
        self.neutralize_pub = self.create_publisher(
            String, "/swarm/neutralize", WORLD_MODEL_QOS
        )

        rate = float(self.get_parameter("update_rate").value)
        self.create_timer(1.0 / max(0.5, rate), self._tick)

        self._mission_report_emitted = False
        self._last_progress_log = 0.0

        self.get_logger().info(
            f"mission_executive up: OPERATION SENTINEL GRID, "
            f"{len(self.cells)} reachable search cells "
            f"({dropped} dropped as unreachable), awaiting "
            f"{list(self.get_parameter('expected_agents').value)}"
        )

    @staticmethod
    def _make_cells_reachable(
        cells: list[SearchCell],
    ) -> tuple[list[SearchCell], int]:
        """Move each blocked cell centre to free ground inside its own cell.

        Returns (cells, number dropped). A cell is only dropped when it is
        entirely built over, which genuinely cannot be swept by anything.
        """
        import dataclasses

        out: list[SearchCell] = []
        dropped = 0
        for cell in cells:
            if not sector.blocked(*cell.centre, 1.0):
                out.append(cell)
                continue
            # Stay inside the cell: a point outside it does not observe it.
            free = sector.nearest_free(
                cell.centre[0], cell.centre[1], margin=1.0,
                step=1.0, max_radius=cell.size * 0.6,
            )
            if sector.blocked(*free, 1.0):
                dropped += 1
                continue
            out.append(dataclasses.replace(cell, centre=free))
        return out, dropped

    # ------------------------------------------------------------- callbacks
    def _on_agent(self, msg: AgentState) -> None:
        self.agents[msg.agent_id] = msg

    def _on_threats(self, msg: ThreatArray) -> None:
        for est in msg.threats:
            self.threats[est.threat_id] = est

    def _on_assignments(self, msg: AssignmentArray) -> None:
        self.assignments = {a.agent_id: a.task_id for a in msg.assignments}

    def _on_task_complete(self, msg: String) -> None:
        parts = msg.data.split(":", 1)
        if len(parts) != 2:
            return
        agent_id, task_id = parts
        self._completed_tasks.add(task_id)
        # A completed search cell is swept, whoever reported it.
        for cell in self.cells:
            if cell.cell_id == task_id and not cell.swept:
                cell.swept = True
                cell.swept_by = agent_id

    # ------------------------------------------------------------- mission
    @property
    def mission_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return self.get_clock().now().nanoseconds * 1e-9 - self._start_time

    def _tick(self) -> None:
        self._tick_count += 1
        expected = list(self.get_parameter("expected_agents").value)

        if self.phase == MissionStatus.PHASE_INIT:
            if all(a in self.agents for a in expected):
                self._start_time = self.get_clock().now().nanoseconds * 1e-9
                self.phase = MissionStatus.PHASE_INGRESS
                self.narrative = "all platforms up; moving off the insertion point"
                self.get_logger().info(
                    f"all {len(expected)} platforms checked in === INGRESS"
                )
            else:
                missing = [a for a in expected if a not in self.agents]
                self.narrative = f"waiting on {', '.join(missing)}"
                self._publish_status(open_tasks=0)
                return

        self._mark_swept_cells()
        self._repartition()
        tasks = self._generate_tasks()
        self._advance_phase(tasks)
        self._emit_marker_orders()

        self._publish_tasks(tasks)
        self._publish_status(open_tasks=len(tasks))
        self._log_progress(tasks)

    # ------------------------------------------------------ world bookkeeping
    def _mark_swept_cells(self) -> None:
        """Retire cells an assigned agent has actually reached.

        Retiring on *proximity of the assigned agent* rather than on anyone
        wandering past is what keeps coverage honest: a drone transiting over a
        cell at 14 m has not searched it.
        """
        radius = float(self.get_parameter("sweep_radius").value)
        by_id = {c.cell_id: c for c in self.cells}

        for agent_id, task_id in self.assignments.items():
            if not task_id.startswith("cell_"):
                continue
            cell = by_id.get(task_id)
            state = self.agents.get(agent_id)
            if cell is None or state is None or cell.swept:
                continue
            d = math.dist(
                (state.pose.position.x, state.pose.position.y), cell.centre
            )
            if d <= radius:
                cell.swept = True
                cell.swept_by = agent_id

    def _repartition(self) -> None:
        """Re-run Lloyd every few seconds, not every tick.

        The partition is a long-horizon split; recomputing it at 2 Hz would
        make regions flicker and the allocator would chase the churn. Every
        5 s is fast enough to react to a platform going down.
        """
        if self._tick_count % 10 != 1:
            return
        agents = []
        for agent_id, state in self.agents.items():
            agents.append(
                (
                    agent_id,
                    Platform(state.platform),
                    (state.pose.position.x, state.pose.position.y),
                )
            )
        if agents:
            self.regions, _ = lloyd_partition(self.cells, agents)

    # ------------------------------------------------------ task generation
    def _generate_tasks(self) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        tasks += self._threat_tasks()
        tasks += self._relay_tasks()
        # Search is the lowest-priority backdrop: it fills whatever capacity the
        # threat work leaves over. Capping the count keeps the assignment
        # problem small and stops the allocator from committing a platform to a
        # distant cell when a threat task is about to open.
        cap = int(self.get_parameter("max_concurrent_search_tasks").value)
        tasks += self._search_tasks(cap)
        # A serviced task must not be re-offered. Triangulation and relay
        # tasks are derived from world state each tick, so they would
        # otherwise reappear the moment they were completed and the agent
        # would be sent straight back.
        return [t for t in tasks if t.task_id not in self._completed_tasks]

    def _egress_tasks(self) -> list[TaskSpec]:
        """One rally task per agent, so every platform actually recovers.

        Without these the task set empties at EGRESS, the allocator has
        nothing to assign, and the agents sit where they finished === the
        mission then never reaches COMPLETE. One task per agent rather than a
        shared one, because Hungarian gives each task to a single bidder.
        """
        rally = tuple(float(v) for v in self.get_parameter("rally_point").value)
        out = []
        for i, agent_id in enumerate(sorted(self.agents)):
            # Fan the rally positions out so three platforms do not converge
            # on one point and deconflict against each other forever.
            angle = 2.0 * math.pi * i / max(1, len(self.agents))
            out.append(
                self._make_task(
                    f"rally_{agent_id}", TaskType.SEARCH_CELL,
                    rally[0] + 5.0 * math.cos(angle),
                    rally[1] + 5.0 * math.sin(angle),
                    radius=6.0, priority=1.0, risk=0.0, service_time=2.0,
                )
            )
        return out

    def _search_tasks(self, cap: int) -> list[TaskSpec]:
        """Nearest few unswept cells per agent region, as biddable tasks."""
        if self.phase in (MissionStatus.PHASE_EGRESS, MissionStatus.PHASE_COMPLETE):
            return self._egress_tasks()

        out: list[TaskSpec] = []
        seen: set[str] = set()
        per_agent = max(1, cap // max(1, len(self.agents)))

        for agent_id, region in self.regions.items():
            state = self.agents.get(agent_id)
            if state is None:
                continue
            pos = (state.pose.position.x, state.pose.position.y)
            candidates = sorted(
                (c for c in region if not c.swept),
                key=lambda c: math.dist(pos, c.centre) / max(0.05, c.weight),
            )[:per_agent]
            for cell in candidates:
                if cell.cell_id in seen:
                    continue
                seen.add(cell.cell_id)
                task = TaskSpec()
                task.task_id = cell.cell_id
                task.task_type = TaskType.SEARCH_CELL
                task.position = Point(x=cell.centre[0], y=cell.centre[1], z=0.0)
                task.radius = float(self.get_parameter("sweep_radius").value)
                # The cell's prior weight *is* its priority; a cell in the
                # overwatch band is worth more than a far corner.
                task.priority = float(min(0.55, 0.18 * cell.weight))
                task.risk = 0.10
                task.service_time = 6.0
                task.deadline = -1.0
                task.required_capabilities = [False] * len(Capability)
                task.complete = False
                out.append(task)
        return out

    def _threat_tasks(self) -> list[TaskSpec]:
        """Triangulation, confirmation, sampling, marking and BDA.

        During egress only *unfinished business* is emitted: marking an
        already-confirmed hostile, and battle damage assessment. New contacts
        are recorded and reported but not pursued.

        That is a deliberate stopping rule, not a shortcut. The sweep has met
        its coverage target, so a fresh unconfirmed candidate is a residual
        detection rather than a reason to re-enter the sector; without the
        rule the team was pulled back repeatedly by new candidates and the
        mission could not terminate.
        """
        egressing = self.phase in (
            MissionStatus.PHASE_EGRESS, MissionStatus.PHASE_COMPLETE
        )
        out: list[TaskSpec] = []
        engageable = set(int(v) for v in self.get_parameter("engageable_classes").value)
        confirm_standoff = float(self.get_parameter("confirm_standoff").value)
        sample_standoff = float(self.get_parameter("sample_standoff").value)

        for tid, est in self.threats.items():
            tx, ty = est.position.x, est.position.y
            sigma = self._sigma_of(est)

            if est.state == ThreatEstimate.STATE_NEUTRALIZED:
                # One BDA pass per neutralised emitter, then let it go.
                if tid not in self._bda_done:
                    out.append(
                        self._make_task(
                            f"{tid}_bda", TaskType.BDA, tx, ty,
                            z=float(self.get_parameter("cruise_altitude").value) * 0.5,
                            radius=confirm_standoff, priority=0.60, risk=0.15,
                            service_time=8.0,
                            caps=(Capability.THERMAL_IMAGING,),
                        )
                    )
                continue

            if egressing and est.state < ThreatEstimate.STATE_CONFIRMED:
                continue

            if est.state == ThreatEstimate.STATE_CANDIDATE:
                # Not tight enough to act on. Buy *geometry*, not more samples
                # from where we already stood.
                for idx, (bx, by) in enumerate(self._baseline_waypoints(tx, ty, sigma)):
                    out.append(
                        self._make_task(
                            f"{tid}_tri{idx}", TaskType.RF_TRIANGULATE, bx, by,
                            radius=6.0,
                            # Urgency scales with how uncertain the fix is.
                            priority=float(min(0.85, 0.40 + 0.02 * sigma)),
                            risk=0.25, service_time=10.0,
                            caps=(Capability.RF_DIRECTION_FIND,),
                        )
                    )
                continue

            if est.state == ThreatEstimate.STATE_LOCALIZED:
                # Two independent modalities are required before engagement.
                out.append(
                    self._make_task(
                        f"{tid}_confirm", TaskType.VISUAL_CONFIRM, tx, ty,
                        z=confirm_standoff,
                        radius=confirm_standoff, priority=0.80, risk=0.45,
                        service_time=12.0,
                        caps=(Capability.THERMAL_IMAGING,),
                    )
                )
                # The chemical signature is what separates a command trigger
                # from a civilian transmitter, so the sample is not optional.
                out.append(
                    self._make_task(
                        f"{tid}_cbrn", TaskType.CBRN_SAMPLE, tx, ty,
                        radius=sample_standoff,
                        priority=0.75,
                        # Standing next to a live trigger is the most dangerous
                        # thing anyone on the team does.
                        risk=0.80, service_time=18.0,
                        caps=(Capability.CBRN_SAMPLING,),
                    )
                )
                continue

            if est.state == ThreatEstimate.STATE_CONFIRMED:
                if int(est.emitter_class) not in engageable:
                    # Confirmed, and confirmed to be something we must not
                    # touch. No task is generated and the track stays on the
                    # map as a no-strike entry.
                    continue
                if tid in self._marked:
                    continue
                out.append(
                    self._make_task(
                        f"{tid}_mark", TaskType.EMPLACE_JAMMER, tx, ty,
                        radius=3.0, priority=1.0, risk=0.70, service_time=15.0,
                        caps=(Capability.MARKER_EMPLACE,),
                    )
                )
        return out

    def _relay_tasks(self) -> list[TaskSpec]:
        """Open a relay station when the comms graph is about to partition.

        The test is on the *graph*, not on any one link: a relay is only worth a
        platform when losing one link would actually disconnect the team. The
        station is placed at the midpoint of the longest link, which is where
        it does the most good.
        """
        if len(self.agents) < 3:
            return []

        cfg = GraphConfig()
        views = {aid: agent_state_to_view(st) for aid, st in self.agents.items()}
        ids = list(views)

        worst_pair, worst_ratio = None, 0.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = views[ids[i]], views[ids[j]]
                d = math.dist(a.position[:2], b.position[:2])
                ratio = d / max(1.0, comms_range_between(a, b, cfg))
                if ratio > worst_ratio:
                    worst_pair, worst_ratio = (a, b), ratio

        # Hysteresis on *opening and closing the task*, not on assigning it.
        # Without it the relay oscillated: the task opened when the team was
        # stretched, a platform was preempted onto it, flying there restored
        # the margin, the task closed, the platform returned to its cell, the
        # margin lapsed again. Observed as the drone being preempted onto
        # `relay_bridge` nine times in a row without ever holding the station.
        open_at = 0.85
        close_at = 0.70
        trigger = close_at if self._relay_open else open_at
        if worst_pair is None or worst_ratio < trigger:
            self._relay_open = False
            return []
        self._relay_open = True

        a, b = worst_pair
        mid_x = (a.position[0] + b.position[0]) / 2.0
        mid_y = (a.position[1] + b.position[1]) / 2.0
        return [
            self._make_task(
                "relay_bridge", TaskType.COMMS_RELAY, mid_x, mid_y,
                z=float(self.get_parameter("cruise_altitude").value),
                radius=8.0,
                priority=float(min(0.9, 0.45 + 0.45 * (worst_ratio - 0.85) / 0.15)),
                risk=0.30, service_time=20.0,
                caps=(Capability.HIGH_GAIN_RADIO,),
            )
        ]

    def _baseline_waypoints(
        self, tx: float, ty: float, sigma: float
    ) -> list[tuple[float, float]]:
        """Two stand-off points on opposite sides of the current cut geometry.

        A DF fix needs *angular diversity*. Taking more cuts from the same side
        tightens the bearing but leaves range unobservable (see
        rf_localizer.multilaterate on GDOP), so the waypoints are placed
        perpendicular to the mean observer direction, at a stand-off that
        scales with the current uncertainty.
        """
        standoff = float(max(14.0, min(34.0, 2.2 * sigma + 12.0)))
        # Mean direction from the current agents to the candidate: the new cuts
        # should be across that, not along it.
        if self.agents:
            mx = sum(s.pose.position.x for s in self.agents.values()) / len(self.agents)
            my = sum(s.pose.position.y for s in self.agents.values()) / len(self.agents)
            axis = math.atan2(ty - my, tx - mx)
        else:
            axis = 0.0
        perp = axis + math.pi / 2.0

        half = float(self.get_parameter("sector_half_extent").value)
        points = []
        for sign in (1.0, -1.0):
            px = tx + sign * standoff * math.cos(perp)
            py = ty + sign * standoff * math.sin(perp)
            # Keep the waypoint inside the sector.
            px = max(-half, min(half, px))
            py = max(-half, min(half, py))
            points.append((px, py))
        return points

    def _make_task(
        self, task_id: str, task_type: TaskType, x: float, y: float,
        z: float = 0.0, radius: float = 4.0, priority: float = 0.5,
        risk: float = 0.2, service_time: float = 10.0,
        caps: tuple[Capability, ...] = (), deadline: float = -1.0,
    ) -> TaskSpec:
        task = TaskSpec()
        task.task_id = task_id
        task.task_type = int(task_type)
        task.position = Point(x=float(x), y=float(y), z=float(z))
        task.radius = float(radius)
        task.priority = float(priority)
        task.risk = float(risk)
        task.service_time = float(service_time)
        task.deadline = float(deadline)
        mask = [False] * len(Capability)
        for c in caps:
            mask[int(c)] = True
        task.required_capabilities = mask
        task.complete = False
        return task

    @staticmethod
    def _sigma_of(est: ThreatEstimate) -> float:
        """Scalar 1-sigma radius from the flattened 2x2 covariance."""
        cxx, cxy, _, cyy = (float(v) for v in est.covariance_xy)
        # Larger eigenvalue of [[cxx, cxy], [cxy, cyy]].
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = max(0.0, (tr / 2.0) ** 2 - det)
        return math.sqrt(max(0.0, tr / 2.0 + math.sqrt(disc)))

    # --------------------------------------------------------- phase + orders
    def _advance_phase(self, tasks: list[TaskSpec]) -> None:
        """Report the most advanced activity actually in progress."""
        types = {t.task_type for t in tasks}
        states = [e.state for e in self.threats.values()]
        coverage = coverage_fraction(self.cells)
        target = float(self.get_parameter("coverage_target").value)

        live_threats = [
            e for e in self.threats.values()
            if e.state != ThreatEstimate.STATE_NEUTRALIZED
        ]
        engageable = set(int(v) for v in self.get_parameter("engageable_classes").value)

        # What still *requires action* before the team can leave.
        #
        # A bare candidate does not. It is a mode in the belief grid that two
        # platforms have taken a bearing on and nothing more, and residual
        # candidates keep appearing from leftover detections for as long as
        # the mission runs. Counting them blocked egress permanently: with the
        # sector 100% swept and all three hostiles neutralised, one unconfirmed
        # candidate held the team in the sector with nothing left to do.
        #
        # A localised track does require action: it is tight enough to be
        # worth corroborating. A confirmed hostile requires action until a
        # marker has been sent at it. Anything already marked is finished from
        # the team's side, whether or not the marker achieved its effect ===
        # that outcome belongs in the report, not in a loop.
        outstanding = []
        for threat_id, e in self.threats.items():
            if e.state == ThreatEstimate.STATE_NEUTRALIZED:
                continue
            if e.state == ThreatEstimate.STATE_LOCALIZED:
                outstanding.append(e)
            elif (
                e.state == ThreatEstimate.STATE_CONFIRMED
                and int(e.emitter_class) in engageable
                and threat_id not in self._marked
            ):
                outstanding.append(e)

        if self.phase == MissionStatus.PHASE_INGRESS:
            waypoint = list(self.get_parameter("ingress_waypoint").value)
            if any(
                math.dist((s.pose.position.x, s.pose.position.y), tuple(waypoint)) < 25.0
                for s in self.agents.values()
            ):
                self.phase = MissionStatus.PHASE_SEARCH
                self.narrative = "on the corridor; beginning the sector sweep"
            return

        if self.phase in (MissionStatus.PHASE_EGRESS, MissionStatus.PHASE_COMPLETE):
            rally = tuple(float(v) for v in self.get_parameter("rally_point").value)
            if self.agents and all(
                math.dist((s.pose.position.x, s.pose.position.y), rally) < 14.0
                for s in self.agents.values()
            ):
                if self.phase != MissionStatus.PHASE_COMPLETE:
                    self.get_logger().info("all platforms at the rally point === MISSION COMPLETE")
                self.phase = MissionStatus.PHASE_COMPLETE
                self.narrative = "all platforms recovered"
                self._emit_mission_report()
            else:
                self.narrative = "recovering to the rally point"
            return

        # Egress when the sweep objective is met and nothing is left open.
        #
        # The gate is the *fraction of cells swept*, not the weighted coverage
        # figure. Those differ, and the difference decided a mission: at a 0.92
        # weighted target the team egressed with 42 of 49 cells swept === 93% by
        # weight === and the seven it skipped happened to contain two of the
        # three hostile emitters, which were never detected at all. Detection
        # here is a consequence of somebody taking DF cuts nearby, so the
        # search guarantee has to be stated in cells, not in a weighted score
        # that low-prior cells can satisfy on someone else's behalf.
        swept = sum(1 for c in self.cells if c.swept)
        sweep_fraction = swept / max(1, len(self.cells))
        sweep_done = sweep_fraction >= target

        if self.mission_time > float(self.get_parameter("max_mission_time").value):
            if not sweep_done:
                self.get_logger().warn(
                    f"mission time limit reached with {len(self.cells) - swept} "
                    f"cells unswept; egressing"
                )
            self.phase = MissionStatus.PHASE_EGRESS
            self.narrative = "time limit reached; egress"
            return

        if sweep_done and not outstanding:
            self.phase = MissionStatus.PHASE_EGRESS
            self.narrative = (
                f"sector swept ({swept}/{len(self.cells)} cells), "
                f"no outstanding threats; egress"
            )
            return

        if TaskType.EMPLACE_JAMMER in types:
            self.phase = MissionStatus.PHASE_NEUTRALIZE
            self.narrative = "confirmed emitter; emplacing a marker"
        elif TaskType.BDA in types:
            self.phase = MissionStatus.PHASE_BDA
            self.narrative = "marker emplaced; verifying the emitter is silent"
        elif TaskType.VISUAL_CONFIRM in types or TaskType.CBRN_SAMPLE in types:
            self.phase = MissionStatus.PHASE_CONFIRM
            self.narrative = "emitter localised; corroborating with two sensors"
        elif TaskType.RF_TRIANGULATE in types:
            self.phase = MissionStatus.PHASE_LOCALIZE
            self.narrative = "contact; spreading the DF baseline to resolve range"
        else:
            self.phase = MissionStatus.PHASE_SEARCH
            self.narrative = f"sweeping the sector, {coverage*100:.0f}% covered"

    def _emit_marker_orders(self) -> None:
        """Fire the neutralise order when the UGV is on top of its mark.

        The order carries the *estimated* position, which is the only one the
        team has. Whether that is close enough to the real emitter is exactly
        what the simulator checks === and is how marking the wrong thing shows
        up in the log rather than being silently forgiven.
        """
        for agent_id, task_id in self.assignments.items():
            if not task_id.endswith("_mark"):
                continue
            threat_id = task_id[: -len("_mark")]
            est = self.threats.get(threat_id)
            state = self.agents.get(agent_id)
            if est is None or state is None or threat_id in self._marked:
                continue
            d = math.dist(
                (state.pose.position.x, state.pose.position.y),
                (est.position.x, est.position.y),
            )
            if d <= 3.5:
                self._marked.add(threat_id)
                self.neutralize_pub.publish(
                    String(data=f"{est.position.x:.3f},{est.position.y:.3f}")
                )
                self.get_logger().info(
                    f"{agent_id} emplaced a marker on {threat_id} at "
                    f"({est.position.x:.1f}, {est.position.y:.1f})"
                )

        # BDA completion: the drone has flown the post-strike look.
        for agent_id, task_id in self.assignments.items():
            if not task_id.endswith("_bda"):
                continue
            threat_id = task_id[: -len("_bda")]
            est = self.threats.get(threat_id)
            state = self.agents.get(agent_id)
            if est is None or state is None:
                continue
            d = math.dist(
                (state.pose.position.x, state.pose.position.y),
                (est.position.x, est.position.y),
            )
            if d <= float(self.get_parameter("confirm_standoff").value):
                if threat_id not in self._bda_done:
                    self._bda_done.add(threat_id)
                    self.get_logger().info(f"{agent_id} completed BDA on {threat_id}")

    def _log_progress(self, tasks: list[TaskSpec]) -> None:
        """Periodic one-line-per-agent progress, so a stall is visible.

        Reports each platform's assigned task and its distance to it. A robot
        whose distance is not falling is stuck, which is exactly the condition
        that previously had to be inferred from a wall of retask messages.
        """
        period = float(self.get_parameter("progress_log_period").value)
        if period <= 0.0 or self.mission_time - self._last_progress_log < period:
            return
        if self.phase == MissionStatus.PHASE_COMPLETE and self._mission_report_emitted:
            # The report has been printed; repeating the progress line adds
            # nothing and buries it.
            return
        self._last_progress_log = self.mission_time

        by_id = {t.task_id: t for t in tasks}
        parts = []
        for agent_id in sorted(self.agents):
            state = self.agents[agent_id]
            task_id = self.assignments.get(agent_id, "")
            task = by_id.get(task_id)
            if task is None:
                parts.append(f"{agent_id}=idle")
                continue
            d = math.dist(
                (state.pose.position.x, state.pose.position.y),
                (task.position.x, task.position.y),
            )
            parts.append(f"{agent_id}={task_id}@{d:.0f}m")

        swept = sum(1 for c in self.cells if c.swept)
        threats = len(self.threats)
        confirmed = sum(
            1 for e in self.threats.values()
            if e.state >= ThreatEstimate.STATE_CONFIRMED
        )
        self.get_logger().info(
            f"[t={self.mission_time:5.0f}s] {_PHASE_NAMES.get(self.phase, '?'):10s} "
            f"cells {swept}/{len(self.cells)} "
            f"({coverage_fraction(self.cells) * 100:.0f}%)  "
            f"threats {threats} ({confirmed} confirmed)  "
            f"tasks {len(tasks)}  " + "  ".join(parts)
        )

    # --------------------------------------------------------------- report
    def _emit_mission_report(self) -> None:
        """Print the mission outcome once, when it completes.

        Scored against `sector.EMITTERS`, which is ground truth. Reading it
        here is legitimate and is confined to this function: it is scoring,
        not sensing, and nothing in the coordination path touches it. Without
        this the operator sees a stream of retask lines and no statement of
        what the mission actually achieved.
        """
        if self._mission_report_emitted:
            return
        self._mission_report_emitted = True

        log = self.get_logger()
        coverage = coverage_fraction(self.cells)
        engageable = set(int(v) for v in self.get_parameter("engageable_classes").value)
        class_names = {0: "unknown", 1: "JAMMER", 2: "IED_TRIGGER", 3: "civilian"}

        log.info("=" * 72)
        log.info("OPERATION SENTINEL GRID === MISSION REPORT")
        log.info("=" * 72)
        log.info(
            f"  outcome        {_PHASE_NAMES.get(self.phase, '?')} "
            f"after {self.mission_time:.0f} s"
        )
        swept = sum(1 for c in self.cells if c.swept)
        log.info(
            f"  sector swept   {swept}/{len(self.cells)} cells "
            f"({swept / max(1, len(self.cells)) * 100:.0f}%), "
            f"{coverage * 100:.0f}% by prior weight"
        )
        unswept = [c for c in self.cells if not c.swept]
        if unswept:
            # An undetected emitter is almost always in here, so name them.
            log.info(
                "    not swept: "
                + ", ".join(f"({c.centre[0]:.0f},{c.centre[1]:.0f})" for c in unswept[:8])
                + (" ..." if len(unswept) > 8 else "")
            )

        # Each ground-truth emitter, and what the team made of it.
        log.info("  emitters:")
        for truth in sector.EMITTERS:
            best, best_d = None, float("inf")
            for est in self.threats.values():
                d = math.dist((est.position.x, est.position.y), (truth.x, truth.y))
                if d < best_d:
                    best, best_d = est, d
            if best is None or best_d > 25.0:
                log.info(
                    f"    {truth.name:12s} {truth.kind:12s} "
                    f"({truth.x:6.1f},{truth.y:6.1f})  NOT DETECTED"
                )
                continue
            state = {0: "candidate", 1: "localized", 2: "CONFIRMED",
                     3: "NEUTRALIZED"}.get(int(best.state), "?")
            correct = class_names.get(int(best.emitter_class), "?")
            expected = {"jammer": "JAMMER", "ied_trigger": "IED_TRIGGER",
                        "civilian": "civilian"}[truth.kind]
            verdict = "correct" if correct == expected else f"MISCLASSIFIED (want {expected})"
            log.info(
                f"    {truth.name:12s} {truth.kind:12s} "
                f"({truth.x:6.1f},{truth.y:6.1f})  fix error {best_d:5.1f} m  "
                f"{state:11s} class {correct:11s} {verdict}"
            )

        # The engagement decisions, which is where the ROE rule shows up.
        marked_ok, marked_bad, missed = [], [], []
        for truth in sector.EMITTERS:
            engaged = any(
                math.dist((e.position.x, e.position.y), (truth.x, truth.y)) <= 8.0
                and int(e.state) == ThreatEstimate.STATE_NEUTRALIZED
                for e in self.threats.values()
            )
            should = truth.kind in ("jammer", "ied_trigger")
            if engaged and should:
                marked_ok.append(truth.name)
            elif engaged and not should:
                marked_bad.append(truth.name)
            elif should:
                missed.append(truth.name)

        log.info("  engagement:")
        log.info(f"    neutralised correctly  {len(marked_ok)}: {marked_ok or '-'}")
        log.info(f"    hostiles left active   {len(missed)}: {missed or '-'}")
        if marked_bad:
            log.error(f"    FRATRICIDE             {len(marked_bad)}: {marked_bad}")
        else:
            log.info("    fratricide             0  (civilian transmitter respected)")

        residual = [
            (tid, e) for tid, e in self.threats.items()
            if e.state == ThreatEstimate.STATE_CANDIDATE
        ]
        if residual:
            log.info(
                f"  residual contacts  {len(residual)} unconfirmed candidate(s), "
                f"reported not pursued:"
            )
            for tid, e in residual:
                log.info(
                    f"    {tid:10s} ({e.position.x:6.1f},{e.position.y:6.1f})  "
                    f"confidence {e.confidence:.2f}"
                )

        log.info("  platforms:")
        for agent_id, state in sorted(self.agents.items()):
            log.info(
                f"    {agent_id:9s} battery {state.battery * 100:5.1f}%  "
                f"link {state.comms_quality:4.2f}  "
                f"last task {state.current_task_id or '-'}"
            )
        log.info("=" * 72)

    # -------------------------------------------------------------- publish
    def _publish_tasks(self, tasks: list[TaskSpec]) -> None:
        msg = TaskArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.tasks = tasks
        self.task_pub.publish(msg)

    def _publish_status(self, open_tasks: int) -> None:
        msg = MissionStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.phase = self.phase
        msg.phase_name = _PHASE_NAMES.get(self.phase, "UNKNOWN")
        msg.mission_time = float(self.mission_time)
        msg.coverage_fraction = float(coverage_fraction(self.cells))
        msg.threats_detected = len(self.threats)
        msg.threats_confirmed = sum(
            1 for e in self.threats.values()
            if e.state >= ThreatEstimate.STATE_CONFIRMED
        )
        msg.threats_neutralized = sum(
            1 for e in self.threats.values()
            if e.state == ThreatEstimate.STATE_NEUTRALIZED
        )
        msg.open_tasks = int(open_tasks)
        msg.narrative = self.narrative
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionExecutiveNode()
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
