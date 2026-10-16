"""Global grid planner, so a platform routes around a block instead of into it.

Why this exists
===============
The local controller is a potential field, which is the right thing for
re-deciding a velocity every tick against teammates and live LiDAR returns.
What it cannot do is get out of a concave trap, and a street grid of 64
buildings is nothing but concave traps. Observed without a global plan: the
quadruped covered 4 m in 70 s because its goal lay beyond a building and the
field held it against the wall, and the UGV arced several metres wide of the
route it was trying to follow.

So the division is the usual one, and each layer does what it is good at:

  * A* over an inflated occupancy grid decides *which way round* the blocks,
    using the known sector geometry. Global, combinatorial, replanned only on
    demand.
  * The potential field tracks the next waypoint and handles what the map does
    not know about === the other two robots, and anything the scanner sees.

Per-platform grids
==================
Each platform gets its own grid, because "passable" is platform-specific:
  drone       nothing is an obstacle above building height; it flies direct
  UGV         wide inflation, and the rubble is impassable
  quadruped   narrow inflation, and it picks through rubble
That is the same capability asymmetry the allocator reasons about, expressed
in the geometry the planner searches.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from . import sector
from .spec import Platform

#: Grid resolution, metres. The narrowest gap the platforms must use is a 4 m
#: alley, and a cell straddling a wall is treated as blocked, so the effective
#: inflation is `INFLATION + resolution / 2`. At 1.0 m that half-cell term ate
#: enough of the alleys to make the UGV unable to reach any courtyard.
DEFAULT_RESOLUTION = 0.75

#: Obstacle inflation per platform: body half-width plus a working margin.
#: The UGV is 0.58 m wide and the quadruped 0.19 m, so these leave the UGV
#: about 0.8 m of clearance either side in a 4 m alley and the quadruped
#: about 1.2 m, which is what lets both reach the block courtyards where the
#: emitters sit. Raising the UGV's value past ~1.0 closes the alleys to it
#: entirely and no marker can ever be emplaced.
INFLATION = {
    Platform.DRONE: 0.0,
    Platform.UGV: 0.8,
    Platform.QUADRUPED: 0.45,
}

#: Whether rubble blocks this platform.
RUBBLE_BLOCKS = {
    Platform.DRONE: False,
    Platform.UGV: True,
    Platform.QUADRUPED: False,
}

_SQRT2 = math.sqrt(2.0)
#: 8-connected moves as (di, dj, cost).
_MOVES = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
)


class OccupancyGrid:
    """Inflated binary obstacle grid over the sector, in the map frame."""

    def __init__(
        self,
        inflation: float = 0.0,
        include_rubble: bool = True,
        resolution: float = DEFAULT_RESOLUTION,
        half_extent: float = sector.SECTOR_HALF,
    ) -> None:
        self.resolution = float(resolution)
        self.half_extent = float(half_extent)
        self.n = int(round(2 * half_extent / resolution))
        self.inflation = float(inflation)

        # Mark a cell blocked if its centre lies within an obstacle inflated by
        # `inflation`, plus half a cell so a cell straddling a wall counts.
        margin = inflation + resolution * 0.5
        self.blocked = np.zeros((self.n, self.n), dtype=bool)
        axis = -half_extent + resolution * (np.arange(self.n) + 0.5)
        xs, ys = np.meshgrid(axis, axis, indexing="ij")
        for box in sector.obstacles(include_rubble=include_rubble):
            self.blocked |= (
                (np.abs(xs - box.x) <= box.half_x + margin)
                & (np.abs(ys - box.y) <= box.half_y + margin)
            )
        self._axis = axis

    # ------------------------------------------------------------- indexing
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        i = int((x + self.half_extent) / self.resolution)
        j = int((y + self.half_extent) / self.resolution)
        return (min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1))

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        return (float(self._axis[i]), float(self._axis[j]))

    def is_free(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n and not self.blocked[i, j]

    def nearest_free_cell(self, i: int, j: int, max_radius: int = 24) -> tuple[int, int] | None:
        """Spiral out to a free cell.

        Both ends of a plan routinely land in an inflated obstacle: a goal from
        the belief grid can sit inside a building, and a robot hugging a wall
        can be inside its own inflation. Snapping to the nearest free cell is
        what keeps those cases planable instead of returning failure and
        leaving the platform parked.
        """
        if self.is_free(i, j):
            return (i, j)
        for r in range(1, max_radius + 1):
            best, best_d = None, None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    # Ring only, not the filled square.
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ci, cj = i + di, j + dj
                    if not self.is_free(ci, cj):
                        continue
                    d = di * di + dj * dj
                    if best_d is None or d < best_d:
                        best, best_d = (ci, cj), d
            if best is not None:
                return best
        return None

    def line_of_sight(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Supercover line walk: true if no blocked cell lies between a and b."""
        (i0, j0), (i1, j1) = a, b
        di, dj = abs(i1 - i0), abs(j1 - j0)
        si = 1 if i1 >= i0 else -1
        sj = 1 if j1 >= j0 else -1
        i, j = i0, j0
        err = di - dj
        while True:
            if not self.is_free(i, j):
                return False
            if (i, j) == (i1, j1):
                return True
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj


@dataclass
class PlanResult:
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    cost: float = 0.0
    #: True when no path to the goal cell existed and the plan heads to the
    #: closest cell the search did reach.
    truncated: bool = False
    #: True when the requested goal lies inside an obstacle (or its inflation)
    #: and the plan ends at the nearest free cell instead.
    #:
    #: This has to be surfaced rather than hidden. The first version snapped
    #: the goal to free space to find a path, then overwrote the last waypoint
    #: with the *original* goal === which pointed the platform straight at the
    #: wall it had just routed around, and reported a path cost shorter than
    #: the straight-line distance, which is impossible for a shortest path and
    #: is what gave the bug away.
    goal_adjusted: bool = False

    @property
    def reaches_goal(self) -> bool:
        """False if the endpoint is only an approach position."""
        return not (self.truncated or self.goal_adjusted)

    @property
    def endpoint(self) -> tuple[float, float] | None:
        return self.waypoints[-1] if self.waypoints else None

    def __bool__(self) -> bool:
        return bool(self.waypoints)


class GridPlanner:
    """A* over a per-platform inflated grid, with path shortcutting."""

    def __init__(self, resolution: float = DEFAULT_RESOLUTION) -> None:
        self.resolution = resolution
        self._grids: dict[Platform, OccupancyGrid] = {}

    def grid_for(self, platform: Platform) -> OccupancyGrid:
        if platform not in self._grids:
            self._grids[platform] = OccupancyGrid(
                inflation=INFLATION[platform],
                include_rubble=RUBBLE_BLOCKS[platform],
                resolution=self.resolution,
            )
        return self._grids[platform]

    def plan(
        self,
        platform: Platform,
        start: tuple[float, float],
        goal: tuple[float, float],
        max_expansions: int = 120_000,
    ) -> PlanResult:
        """Shortest 8-connected path from start to goal, as sparse waypoints.

        The drone short-circuits to a straight line: it flies above the
        buildings, so searching a grid for it would be wasted work and would
        also wrongly forbid flying over a block.
        """
        if platform == Platform.DRONE:
            return PlanResult(waypoints=[goal], cost=math.dist(start, goal))

        grid = self.grid_for(platform)
        s = grid.nearest_free_cell(*grid.to_cell(*start))
        goal_cell = grid.to_cell(*goal)
        g = grid.nearest_free_cell(*goal_cell)
        if s is None or g is None:
            return PlanResult()
        goal_adjusted = g != goal_cell
        if s == g:
            end_xy = grid.to_world(*g) if goal_adjusted else goal
            return PlanResult(
                waypoints=[end_xy], cost=0.0, goal_adjusted=goal_adjusted
            )

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {s: 0.0}
        # Octile heuristic, which is the admissible one for 8-connected grids.
        def h(c):
            dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
            return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)

        open_heap = [(h(s), 0.0, s)]
        closed: set[tuple[int, int]] = set()
        expansions = 0
        best_cell, best_h = s, h(s)

        while open_heap:
            _f, cost, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            expansions += 1

            hc = h(cur)
            if hc < best_h:
                best_cell, best_h = cur, hc

            if cur == g:
                break
            if expansions > max_expansions:
                break

            ci, cj = cur
            for di, dj, step in _MOVES:
                ni, nj = ci + di, cj + dj
                if not grid.is_free(ni, nj):
                    continue
                # No corner cutting: a diagonal needs both orthogonals free,
                # otherwise the path clips a building corner the body cannot.
                if di and dj and not (grid.is_free(ci + di, cj) and grid.is_free(ci, cj + dj)):
                    continue
                ng = cost + step
                if ng < g_score.get((ni, nj), math.inf):
                    g_score[(ni, nj)] = ng
                    came_from[(ni, nj)] = cur
                    heapq.heappush(open_heap, (ng + h((ni, nj)), ng, (ni, nj)))

        truncated = g not in came_from and g != s
        end = g if not truncated else best_cell
        if end != s and end not in came_from:
            return PlanResult()

        # Reconstruct, then shortcut.
        cells = [end]
        while cells[-1] != s:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        cells = self._shortcut(grid, cells)

        waypoints = [grid.to_world(i, j) for i, j in cells[1:]]
        if not truncated and not goal_adjusted:
            # Finish on the exact goal rather than on its cell centre. Only
            # legitimate when the goal cell itself was free and reached.
            if waypoints:
                waypoints[-1] = goal
            else:
                waypoints = [goal]
        if not waypoints:
            waypoints = [grid.to_world(*end)]
        return PlanResult(
            waypoints=waypoints,
            cost=g_score.get(end, math.inf) * grid.resolution,
            truncated=truncated,
            goal_adjusted=goal_adjusted,
        )

    @staticmethod
    def _shortcut(grid: OccupancyGrid, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedy line-of-sight shortcutting.

        A* on a grid returns a staircase of hundreds of cells. Keeping only the
        corners the path actually needs turns it into a handful of waypoints,
        which is what the velocity controller can follow smoothly instead of
        chattering between adjacent cells.
        """
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not grid.line_of_sight(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out


@dataclass
class PathFollower:
    """Consumes a plan and hands the local controller one waypoint at a time."""

    waypoints: list[tuple[float, float]] = field(default_factory=list)
    index: int = 0
    #: Distance at which an intermediate waypoint counts as reached. Generous
    #: on purpose: these are routing hints, not positions to hold.
    waypoint_tolerance: float = 2.5

    def set_plan(self, waypoints: list[tuple[float, float]]) -> None:
        self.waypoints = list(waypoints)
        self.index = 0

    def clear(self) -> None:
        self.waypoints = []
        self.index = 0

    @property
    def active(self) -> bool:
        return self.index < len(self.waypoints)

    @property
    def on_final_leg(self) -> bool:
        return self.index >= len(self.waypoints) - 1

    def current(self, position: tuple[float, float]) -> tuple[float, float] | None:
        """Advance past reached waypoints and return the one to steer at."""
        while self.index < len(self.waypoints) - 1:
            if math.dist(position, self.waypoints[self.index]) <= self.waypoint_tolerance:
                self.index += 1
            else:
                break
        if self.index < len(self.waypoints):
            return self.waypoints[self.index]
        return None
