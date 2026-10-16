"""Search-cell generation and Lloyd/Voronoi coverage partitioning.

The search phase has to answer two questions: *where should we look*, and *who
looks where*. They are different problems and get different algorithms.

  * Where: the sector is tiled into search cells, each weighted by how likely
    an emitter is to be there. The weight is not uniform === a command emitter
    sits where it has line of sight to the route it covers, so cells near the
    main corridor and near building edges are weighted up. Cells are retired as
    they are swept, which is what makes `coverage_fraction` meaningful.

  * Who: the cells are partitioned between platforms by weighted Lloyd descent
    on a Voronoi diagram, using each platform's *own* travel metric. The drone's
    metric is Euclidean, the UGV's is inflated off-road, the dog's is nearly
    isotropic but slow. Descending on those metrics produces regions that match
    what each platform can actually reach, with no hand-drawn sector boundaries.

Lloyd gives the long-horizon split; the GNN allocator then picks the specific
next cell for each robot inside its region, which is where the relational
reasoning (comms coverage, who is already committed) comes in. Lloyd alone
would happily send the UGV into the rubble; the allocator alone would thrash
between distant cells with no spatial coherence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .spec import PROFILES, Platform


@dataclass
class SearchCell:
    """One rectangular patch of the sector to be swept."""

    cell_id: str
    centre: tuple[float, float]
    size: float
    #: Prior probability weight that an emitter is in this cell.
    weight: float = 1.0
    swept_by: str = ""
    swept: bool = False
    #: Platforms that physically cannot work this cell (e.g. UGV past rubble).
    excluded: tuple[Platform, ...] = ()

    @property
    def area(self) -> float:
        return self.size * self.size


def travel_metric(platform: Platform, a, b) -> float:
    """Platform-specific distance, in seconds of travel.

    This is the metric Lloyd descends on. Using time rather than metres is what
    makes the partition fair between a 8 m/s drone and a 1.5 m/s quadruped:
    equal-time regions, not equal-area ones.
    """
    profile = PROFILES[platform]
    d = math.dist(a[:2], b[:2])
    if platform == Platform.UGV:
        # Road-bound: an L1-ish path through a street grid, not a straight line.
        detour = abs(a[0] - b[0]) + abs(a[1] - b[1])
        d = 0.35 * d + 0.65 * detour
    return d / max(0.1, profile.max_speed)


def build_search_cells(
    half_extent: float = 55.0,
    cell_size: float = 11.0,
    corridor_half_width: float = 7.0,
    rubble_exclusions: tuple[tuple[float, float, float], ...] = (),
) -> list[SearchCell]:
    """Tile the sector, weighting cells by emitter-placement likelihood.

    `rubble_exclusions` is a tuple of (x, y, radius) the UGV cannot reach;
    cells whose centre falls inside one are marked UGV-excluded, so the
    partition and the allocator both route around them instead of discovering
    the blockage by driving into it.
    """
    cells: list[SearchCell] = []
    n = int(round(2 * half_extent / cell_size))
    start = -half_extent + cell_size / 2.0

    for i in range(n):
        for j in range(n):
            cx = start + i * cell_size
            cy = start + j * cell_size

            # Emitters are placed to cover the routes they target, so proximity
            # to the main corridor and the cross street raises the prior.
            d_corridor = min(abs(cy), abs(cx))
            if d_corridor < corridor_half_width:
                # Directly on the route: an emitter here would be seen and
                # removed by the dismounts, so it is actually less likely.
                weight = 0.45
            elif d_corridor < corridor_half_width + cell_size:
                # The overwatch band: best line of sight to the route.
                weight = 1.9
            else:
                # Deep inside a block: concealed but poor coverage of the route.
                weight = 0.85

            # The sector edges are the insertion and egress routes; sweeping
            # them matters more than the far corners.
            if max(abs(cx), abs(cy)) > half_extent - cell_size:
                weight *= 0.7

            excluded: list[Platform] = []
            for rx, ry, rr in rubble_exclusions:
                if math.dist((cx, cy), (rx, ry)) <= rr:
                    excluded.append(Platform.UGV)
                    break

            cells.append(
                SearchCell(
                    cell_id=f"cell_{i:02d}_{j:02d}",
                    centre=(cx, cy),
                    size=cell_size,
                    weight=weight,
                    excluded=tuple(excluded),
                )
            )
    return cells


def fair_shares(agents: list[tuple[str, Platform, tuple[float, float]]],
                exponent: float = 0.5) -> dict[str, float]:
    """Target fraction of the total search weight each agent should sweep.

    Proportional to speed^exponent, not to speed. With exponent 1.0 an 8 m/s
    drone would be given five times the quadruped's work and the ground
    platforms would sit idle for most of the phase; measured on this sector, a
    pure equal-time partition hands the drone 92 of 100 cells. The sub-linear
    exponent keeps the fast platform doing more without making the slow ones
    useless, which matters because the ground platforms are the only ones whose
    DF cuts come from inside the clutter.
    """
    raw = {aid: PROFILES[plat].max_speed ** exponent for aid, plat, _ in agents}
    total = sum(raw.values()) or 1.0
    return {aid: v / total for aid, v in raw.items()}


def lloyd_partition(
    cells: list[SearchCell],
    agents: list[tuple[str, Platform, tuple[float, float]]],
    iterations: int = 24,
    tolerance: float = 0.25,
    balance_gain: float = 0.6,
    share_exponent: float = 0.5,
) -> tuple[dict[str, list[SearchCell]], dict[str, tuple[float, float]]]:
    """Capacity-constrained weighted Lloyd descent over the search cells.

    Each iteration:
      (a) assign every unswept cell to the agent minimising
              travel_metric(seed_i, cell) / cell.weight + bias_i
      (b) move each seed to the weight-weighted centroid of its region
      (c) nudge bias_i to drive the agent's actual share of the swept weight
          toward its `fair_shares` target

    Step (a) with an additive per-agent bias is a *power diagram* rather than a
    plain Voronoi diagram, and (c) is the standard way to make one
    capacity-constrained: the bias is a Lagrange multiplier on the load
    constraint, and raising it shrinks that agent's region. Without (c) the
    fastest platform absorbs nearly the whole sector (see `fair_shares`).

    Steps (a)+(b) alone are Lloyd's algorithm on a multiplicatively-weighted
    Voronoi diagram, the classic coverage-control result (Cortes et al., 2004);
    the platform-specific metric and the load balancing are what adapt it to a
    heterogeneous team.

    Returns (regions keyed by agent id, converged seed per agent).
    """
    live = [c for c in cells if not c.swept]
    seeds = {aid: (pos[0], pos[1]) for aid, _, pos in agents}
    platforms = {aid: plat for aid, plat, _ in agents}
    regions: dict[str, list[SearchCell]] = {aid: [] for aid, _, _ in agents}
    if not live or not agents:
        return regions, seeds

    targets = fair_shares(agents, share_exponent)
    bias = {aid: 0.0 for aid in seeds}
    total_weight = sum(c.weight for c in live) or 1.0
    # Bias is in the same units as the normalised metric, so scale the nudge by
    # a typical cell cost to keep the gain dimensionless.
    cost_scale = max(
        1e-3,
        np.median([
            travel_metric(platforms[aid], seeds[aid], c.centre) / max(0.05, c.weight)
            for aid in seeds for c in live[:: max(1, len(live) // 20)]
        ]),
    )

    for _ in range(iterations):
        regions = {aid: [] for aid in seeds}
        for cell in live:
            best_agent, best_cost = None, float("inf")
            for aid, seed in seeds.items():
                if platforms[aid] in cell.excluded:
                    continue
                cost = (
                    travel_metric(platforms[aid], seed, cell.centre) / max(0.05, cell.weight)
                    + bias[aid]
                )
                if cost < best_cost:
                    best_agent, best_cost = aid, cost
            if best_agent is not None:
                regions[best_agent].append(cell)

        shift = 0.0
        for aid, region in regions.items():
            if region:
                w = np.array([c.weight for c in region], dtype=np.float64)
                pts = np.array([c.centre for c in region], dtype=np.float64)
                centroid = (pts * w[:, None]).sum(axis=0) / w.sum()
                shift = max(shift, float(np.linalg.norm(np.array(seeds[aid]) - centroid)))
                seeds[aid] = (float(centroid[0]), float(centroid[1]))

        # Load balancing: overloaded agents get a higher bias, shrinking them.
        max_imbalance = 0.0
        for aid in seeds:
            share = sum(c.weight for c in regions[aid]) / total_weight
            error = share - targets[aid]
            max_imbalance = max(max_imbalance, abs(error))
            bias[aid] += balance_gain * cost_scale * error

        if shift < tolerance and max_imbalance < 0.05:
            break

    return regions, seeds


def next_cell_for_agent(
    region: list[SearchCell],
    position: tuple[float, float],
    platform: Platform,
) -> SearchCell | None:
    """Cheapest unswept cell in a region, by weighted travel metric.

    Used as the fallback sweep order and as the candidate set the allocator
    bids over; the GNN may still prefer a different cell once it accounts for
    the rest of the team.
    """
    best, best_cost = None, float("inf")
    for cell in region:
        if cell.swept or platform in cell.excluded:
            continue
        cost = travel_metric(platform, position, cell.centre) / max(0.05, cell.weight)
        if cost < best_cost:
            best, best_cost = cell, cost
    return best


def coverage_fraction(cells: list[SearchCell]) -> float:
    """Weight-weighted fraction of the sector swept.

    Weighted, not plain count: clearing the high-prior overwatch band matters
    more than ticking off a far corner, and the mission report should say so.
    """
    total = sum(c.weight for c in cells)
    if total <= 0.0:
        return 1.0
    return sum(c.weight for c in cells if c.swept) / total
