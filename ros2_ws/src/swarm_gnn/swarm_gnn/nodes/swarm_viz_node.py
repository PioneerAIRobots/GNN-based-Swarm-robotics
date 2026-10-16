#!/usr/bin/env python3
"""Renders the swarm's internal state as RViz markers.

Everything drawn here is something the coordination layer actually believes ===
there are no decorations. In particular the graph edges are the real edges the
GNN consumed, with their learned attention mapped to line width, which makes it
possible to watch the network decide: when the UGV commits to a marking task
the attention on its link to the relaying drone visibly thickens.

Namespaces (toggle them individually in RViz):
    agents        platform position, heading and id
    comms         agent-agent edges, width by attention
    bids          agent-task edges, width by attention
    assignments   the committed matching, as arrows
    tasks         open tasks, coloured by type
    threats       fused tracks with their 2-sigma uncertainty ellipse
    mission       phase and coverage readout
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import ColorRGBA
from swarm_interfaces.msg import (
    AgentState,
    AssignmentArray,
    GraphDebug,
    MissionStatus,
    TaskArray,
    ThreatArray,
    ThreatEstimate,
)
from visualization_msgs.msg import Marker, MarkerArray

from ..model import sector
from ..model.spec import TaskType
from ..util.qos import COORDINATION_QOS, SENSOR_QOS, WORLD_MODEL_QOS


def _c(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


PLATFORM_COLOR = {
    0: _c(0.25, 0.70, 1.00),   # drone, cyan
    1: _c(0.95, 0.75, 0.15),   # UGV, amber
    2: _c(0.45, 0.90, 0.45),   # quadruped, green
}

TASK_COLOR = {
    TaskType.SEARCH_CELL: _c(0.55, 0.55, 0.60, 0.55),
    TaskType.RF_TRIANGULATE: _c(0.40, 0.65, 1.00, 0.85),
    TaskType.VISUAL_CONFIRM: _c(1.00, 0.60, 0.15, 0.90),
    TaskType.CBRN_SAMPLE: _c(0.85, 0.85, 0.20, 0.90),
    TaskType.EMPLACE_JAMMER: _c(1.00, 0.20, 0.20, 0.95),
    TaskType.COMMS_RELAY: _c(0.75, 0.40, 1.00, 0.85),
    TaskType.BDA: _c(0.30, 1.00, 0.80, 0.90),
}

THREAT_COLOR = {
    ThreatEstimate.STATE_CANDIDATE: _c(0.90, 0.90, 0.30, 0.50),
    ThreatEstimate.STATE_LOCALIZED: _c(1.00, 0.55, 0.10, 0.70),
    ThreatEstimate.STATE_CONFIRMED: _c(1.00, 0.15, 0.15, 0.85),
    ThreatEstimate.STATE_NEUTRALIZED: _c(0.35, 0.35, 0.40, 0.55),
}

# GraphDebug.edge_type
EDGE_AGENT_AGENT, EDGE_AGENT_TASK, EDGE_TASK_TASK = 0, 1, 2


class SwarmVizNode(Node):
    def __init__(self) -> None:
        super().__init__("swarm_viz")

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate", 4.0)
        self.declare_parameter("draw_search_tasks", True)
        # Attention below this is not drawn: a fully-connected bid graph with
        # 30 tasks is unreadable, and the whole point is to see what the network
        # is actually weighting.
        self.declare_parameter("attention_floor", 0.08)
        self.declare_parameter("draw_sector", True)
        # The sector is static, so it is re-sent slowly rather than every
        # cycle. It cannot be sent only once: the marker topic is volatile, so
        # any subscriber that joins later === RViz started after this node, or
        # reopened === would show an empty world with robots floating in it.
        self.declare_parameter("sector_repeat_period", 4.0)
        # The RF belief is also drawn as markers, not only published as an
        # OccupancyGrid. RViz's Map display renders every colour scheme through
        # its indexed-8-bit image shader, which fails to link on OpenGL 2.1 /
        # GLSL 1.2 ("active samplers with a different type refer to the same
        # texture image unit") and leaves the grid invisible on that hardware.
        # Markers use the ordinary pipeline and render anywhere, and the belief
        # is the clearest view of why the team spends triangulation tasks.
        self.declare_parameter("draw_belief", True)
        # Only cells above this fraction of the peak are drawn, which keeps a
        # 6400-cell grid down to a few hundred markers.
        self.declare_parameter("belief_threshold", 0.25)

        self.frame = str(self.get_parameter("frame_id").value)
        self.agents: dict[str, AgentState] = {}
        self.tasks: dict[str, object] = {}
        self.threats: dict[str, ThreatEstimate] = {}
        self.assignments: list = []
        self.graph: GraphDebug | None = None
        self.status: MissionStatus | None = None
        self.belief: OccupancyGrid | None = None

        self.create_subscription(AgentState, "/swarm/agent_states", self._on_agent, SENSOR_QOS)
        self.create_subscription(TaskArray, "/swarm/tasks", self._on_tasks, COORDINATION_QOS)
        self.create_subscription(ThreatArray, "/swarm/threats", self._on_threats, WORLD_MODEL_QOS)
        self.create_subscription(
            AssignmentArray, "/swarm/assignments", self._on_assign, COORDINATION_QOS
        )
        self.create_subscription(GraphDebug, "/swarm/graph_debug", self._on_graph, SENSOR_QOS)
        self.create_subscription(
            MissionStatus, "/swarm/mission_status", self._on_status, COORDINATION_QOS
        )
        self.create_subscription(
            OccupancyGrid, "/swarm/belief_grid", self._on_belief, WORLD_MODEL_QOS
        )

        self.marker_pub = self.create_publisher(MarkerArray, "/swarm/markers", 1)
        self.create_timer(
            1.0 / max(1.0, float(self.get_parameter("publish_rate").value)), self._publish
        )
        self.get_logger().info("swarm_viz up: publishing /swarm/markers")

    # ------------------------------------------------------------- callbacks
    def _on_agent(self, msg: AgentState) -> None:
        self.agents[msg.agent_id] = msg

    def _on_tasks(self, msg: TaskArray) -> None:
        self.tasks = {t.task_id: t for t in msg.tasks}

    def _on_threats(self, msg: ThreatArray) -> None:
        self.threats = {t.threat_id: t for t in msg.threats}

    def _on_assign(self, msg: AssignmentArray) -> None:
        self.assignments = list(msg.assignments)

    def _on_graph(self, msg: GraphDebug) -> None:
        self.graph = msg

    def _on_status(self, msg: MissionStatus) -> None:
        self.status = msg

    def _on_belief(self, msg: OccupancyGrid) -> None:
        self.belief = msg

    # --------------------------------------------------------------- helpers
    def _base(self, ns: str, mid: int, mtype: int, scale: Vector3,
              color: ColorRGBA) -> Marker:
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.scale = scale
        m.color = color
        m.pose.orientation.w = 1.0
        # Markers are refreshed every cycle; a lifetime slightly longer than
        # the period means a stalled publisher visibly clears instead of
        # leaving a stale picture on screen.
        m.lifetime.sec = 2
        return m

    def _agent_point(self, agent_id: str) -> Point | None:
        state = self.agents.get(agent_id)
        if state is None:
            return None
        return Point(
            x=state.pose.position.x, y=state.pose.position.y, z=state.pose.position.z
        )

    # --------------------------------------------------------------- publish
    def _publish(self) -> None:
        markers: list[Marker] = []
        markers += self._agent_markers()
        markers += self._task_markers()
        markers += self._threat_markers()
        markers += self._assignment_markers()
        markers += self._graph_markers()
        markers += self._status_markers()
        markers += self._sector_markers()
        markers += self._belief_markers()
        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))

    def _agent_markers(self) -> list[Marker]:
        out = []
        for i, (aid, state) in enumerate(sorted(self.agents.items())):
            color = PLATFORM_COLOR.get(int(state.platform), _c(1, 1, 1))
            body = self._base(
                "agents", i, Marker.SPHERE, Vector3(x=2.2, y=2.2, z=2.2), color
            )
            body.pose = state.pose
            out.append(body)

            # Heading spike, so orientation is readable from the operator view.
            arrow = self._base(
                "agents", 1000 + i, Marker.ARROW, Vector3(x=4.5, y=0.6, z=0.6), color
            )
            arrow.pose = state.pose
            out.append(arrow)

            label = self._base(
                "agents", 2000 + i, Marker.TEXT_VIEW_FACING,
                Vector3(x=0.0, y=0.0, z=2.4), _c(1, 1, 1, 0.95)
            )
            label.pose = state.pose
            label.pose.position.z += 3.2
            task = state.current_task_id or "idle"
            label.text = (
                f"{aid}\n{task}\nbatt {state.battery*100:.0f}%  "
                f"link {state.comms_quality*100:.0f}%"
            )
            out.append(label)
        return out

    def _task_markers(self) -> list[Marker]:
        draw_search = bool(self.get_parameter("draw_search_tasks").value)
        out = []
        for i, task in enumerate(self.tasks.values()):
            ttype = TaskType(task.task_type)
            if ttype == TaskType.SEARCH_CELL and not draw_search:
                continue
            color = TASK_COLOR.get(ttype, _c(0.7, 0.7, 0.7, 0.6))

            if ttype == TaskType.SEARCH_CELL:
                # Flat footprint: this is an area to sweep, not a point to visit.
                m = self._base(
                    "tasks", i, Marker.CYLINDER,
                    Vector3(x=task.radius * 2, y=task.radius * 2, z=0.25), color
                )
            else:
                m = self._base(
                    "tasks", i, Marker.CUBE, Vector3(x=2.4, y=2.4, z=2.4), color
                )
            m.pose.position = task.position
            m.pose.position.z = max(0.2, task.position.z)
            out.append(m)
        return out

    def _threat_markers(self) -> list[Marker]:
        out = []
        for i, est in enumerate(self.threats.values()):
            color = THREAT_COLOR.get(int(est.state), _c(1, 1, 1, 0.6))

            # 2-sigma ellipse from the covariance: the operator needs to see
            # *how well* it is localised, not just where the estimate is.
            cxx, cxy, _, cyy = (float(v) for v in est.covariance_xy)
            tr, det = cxx + cyy, cxx * cyy - cxy * cxy
            disc = max(0.0, (tr / 2.0) ** 2 - det)
            major = math.sqrt(max(1e-3, tr / 2.0 + math.sqrt(disc)))
            minor = math.sqrt(max(1e-3, tr / 2.0 - math.sqrt(disc)))
            angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)

            ellipse = self._base(
                "threats", i, Marker.CYLINDER,
                Vector3(x=min(80.0, 4.0 * major), y=min(80.0, 4.0 * minor), z=0.15),
                ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.25),
            )
            ellipse.pose.position = est.position
            ellipse.pose.position.z = 0.1
            ellipse.pose.orientation.z = math.sin(angle / 2.0)
            ellipse.pose.orientation.w = math.cos(angle / 2.0)
            out.append(ellipse)

            mast = self._base(
                "threats", 1000 + i, Marker.CYLINDER,
                Vector3(x=1.2, y=1.2, z=6.0), color
            )
            mast.pose.position = est.position
            mast.pose.position.z = 3.0
            out.append(mast)

            state_name = {
                0: "CANDIDATE", 1: "LOCALIZED", 2: "CONFIRMED", 3: "NEUTRALIZED"
            }.get(int(est.state), "?")
            class_name = {
                0: "unknown", 1: "JAMMER", 2: "IED TRIGGER", 3: "civilian"
            }.get(int(est.emitter_class), "?")
            label = self._base(
                "threats", 2000 + i, Marker.TEXT_VIEW_FACING,
                Vector3(z=2.0), _c(1, 1, 1, 0.95)
            )
            label.pose.position = est.position
            label.pose.position.z = 8.0
            label.text = (
                f"{est.threat_id}\n{state_name} / {class_name}\n"
                f"sigma {major:.1f} m  conf {est.confidence:.2f}  "
                f"corrob {est.confirm_count}/2"
            )
            out.append(label)
        return out

    def _assignment_markers(self) -> list[Marker]:
        out = []
        for i, a in enumerate(self.assignments):
            start = self._agent_point(a.agent_id)
            task = self.tasks.get(a.task_id)
            if start is None or task is None:
                continue
            state = self.agents.get(a.agent_id)
            color = PLATFORM_COLOR.get(
                int(state.platform) if state else -1, _c(1, 1, 1)
            )
            m = self._base(
                "assignments", i, Marker.ARROW,
                Vector3(x=0.55, y=1.3, z=1.3),
                ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.95),
            )
            end = Point(x=task.position.x, y=task.position.y, z=max(0.4, task.position.z))
            m.points = [start, end]
            out.append(m)
        return out

    def _graph_markers(self) -> list[Marker]:
        """The actual GNN input graph, with attention as line width."""
        if self.graph is None:
            return []
        g = self.graph
        floor = float(self.get_parameter("attention_floor").value)
        n_agents = g.num_agent_nodes

        def node_point(idx: int) -> Point | None:
            if idx < n_agents:
                return self._agent_point(g.agent_ids[idx]) if idx < len(g.agent_ids) else None
            t_idx = idx - n_agents
            if t_idx >= len(g.task_ids):
                return None
            task = self.tasks.get(g.task_ids[t_idx])
            if task is None:
                return None
            return Point(x=task.position.x, y=task.position.y, z=max(0.4, task.position.z))

        comms: list[Marker] = []
        bids: list[Marker] = []
        for e in range(len(g.edge_src)):
            if g.edge_type[e] == EDGE_TASK_TASK:
                # Task-task edges are dense and tell the operator little.
                continue
            att = g.attention[e] if e < len(g.attention) else 1.0
            if att < floor:
                continue
            a = node_point(g.edge_src[e])
            b = node_point(g.edge_dst[e])
            if a is None or b is None:
                continue
            if g.edge_src[e] == g.edge_dst[e]:
                continue  # self-loop

            is_comms = g.edge_type[e] == EDGE_AGENT_AGENT
            # Width carries attention: a thick line is an edge the network is
            # actually leaning on this tick.
            width = 0.12 + 1.5 * min(1.0, att)
            target = comms if is_comms else bids
            m = self._base(
                "comms" if is_comms else "bids",
                len(target), Marker.LINE_LIST,
                Vector3(x=width),
                _c(0.30, 0.95, 0.85, 0.75) if is_comms else _c(0.85, 0.85, 0.35, 0.35),
            )
            m.points = [a, b]
            target.append(m)
        return comms + bids

    def _sector_markers(self) -> list[Marker]:
        """The buildings, rubble, routes and rally points.

        Published from `model/sector.py`, the same geometry the planner
        searches and the platforms collide against, so what the operator sees
        is exactly what the robots are navigating. Without Gazebo there is no
        other source of the scene.

        Re-sent every few seconds rather than on every cycle: 67 cubes at
        4 Hz is pure overhead, but sending them once is not enough either,
        because a late subscriber on a volatile topic would never receive
        them and would show robots floating in an empty world.
        """
        if not bool(self.get_parameter("draw_sector").value):
            return []
        now = self.get_clock().now().nanoseconds * 1e-9
        period = float(self.get_parameter("sector_repeat_period").value)
        if now - getattr(self, "_sector_sent", -1e9) < period:
            return []
        self._sector_sent = now

        out: list[Marker] = []
        for i, box in enumerate(sector.BUILDINGS):
            m = self._base(
                "sector", i, Marker.CUBE,
                Vector3(x=box.half_x * 2, y=box.half_y * 2, z=box.height),
                _c(0.42, 0.44, 0.48, 0.85),
            )
            m.pose.position = Point(x=box.x, y=box.y, z=box.height / 2.0)
            m.lifetime.sec = 0
            out.append(m)

        for i, box in enumerate(sector.RUBBLE):
            m = self._base(
                "sector", 1000 + i, Marker.CUBE,
                Vector3(x=box.half_x * 2, y=box.half_y * 2, z=box.height),
                _c(0.55, 0.40, 0.25, 0.9),
            )
            m.pose.position = Point(x=box.x, y=box.y, z=box.height / 2.0)
            m.lifetime.sec = 0
            out.append(m)

        # The two open routes, as flat strips.
        for i, (sx, sy) in enumerate(
            ((2 * sector.SEARCH_HALF, 2 * sector.ROUTE_HALF_WIDTH),
             (2 * sector.ROUTE_HALF_WIDTH, 2 * sector.SEARCH_HALF))
        ):
            m = self._base(
                "sector", 2000 + i, Marker.CUBE, Vector3(x=sx, y=sy, z=0.05),
                _c(0.22, 0.23, 0.26, 0.9),
            )
            m.pose.position = Point(x=0.0, y=0.0, z=0.01)
            m.lifetime.sec = 0
            out.append(m)

        for i, (label, (px, py), colour) in enumerate((
            ("INSERTION", sector.INSERTION, _c(0.25, 0.85, 0.35, 0.8)),
            ("RALLY", sector.RALLY, _c(0.35, 0.55, 1.0, 0.8)),
        )):
            disc = self._base(
                "sector", 3000 + i, Marker.CYLINDER,
                Vector3(x=6.0, y=6.0, z=0.08), colour,
            )
            disc.pose.position = Point(x=px, y=py, z=0.04)
            disc.lifetime.sec = 0
            out.append(disc)

            text = self._base(
                "sector", 3100 + i, Marker.TEXT_VIEW_FACING,
                Vector3(z=3.0), _c(1, 1, 1, 0.9),
            )
            text.pose.position = Point(x=px, y=py, z=3.0)
            text.text = label
            text.lifetime.sec = 0
            out.append(text)

        return out

    def _belief_markers(self) -> list[Marker]:
        """The RF posterior as a single CUBE_LIST, one cube per likely cell.

        Drawn rather than left to RViz's Map display, which cannot render an
        OccupancyGrid at all on OpenGL 2.1. One marker holding many cubes
        keeps this to a single message regardless of how much of the grid is
        above threshold.
        """
        if not bool(self.get_parameter("draw_belief").value) or self.belief is None:
            return []

        grid = self.belief
        data = grid.data
        peak = max(data) if data else 0
        if peak <= 0:
            return []
        floor = peak * float(self.get_parameter("belief_threshold").value)

        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y

        m = self._base(
            "belief", 0, Marker.CUBE_LIST,
            Vector3(x=res, y=res, z=0.1), _c(1, 1, 1, 1),
        )
        m.pose.position = Point(x=0.0, y=0.0, z=0.0)
        for index, value in enumerate(data):
            if value < floor:
                continue
            i = index % grid.info.width
            j = index // grid.info.width
            m.points.append(Point(
                x=ox + (i + 0.5) * res, y=oy + (j + 0.5) * res, z=0.05,
            ))
            # Yellow where the posterior is weak, red where it is strong, so
            # the eye goes to the cells the team is about to act on.
            t = min(1.0, (value - floor) / max(1.0, peak - floor))
            m.colors.append(ColorRGBA(
                r=1.0, g=float(0.85 - 0.75 * t), b=0.10, a=float(0.25 + 0.5 * t),
            ))
        return [m] if m.points else []

    def _status_markers(self) -> list[Marker]:
        if self.status is None:
            return []
        s = self.status
        m = self._base(
            "mission", 0, Marker.TEXT_VIEW_FACING, Vector3(z=4.0), _c(1, 1, 1, 0.95)
        )
        m.pose.position = Point(x=0.0, y=0.0, z=34.0)
        m.text = (
            f"OPERATION SENTINEL GRID === {s.phase_name}\n"
            f"t+{s.mission_time:.0f}s   coverage {s.coverage_fraction*100:.0f}%   "
            f"tasks {s.open_tasks}\n"
            f"threats: {s.threats_detected} detected / {s.threats_confirmed} confirmed "
            f"/ {s.threats_neutralized} neutralised\n"
            f"{s.narrative}"
        )
        return [m]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SwarmVizNode()
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
