#!/usr/bin/env python3
"""Regression suite for the coordination layer. No ROS, no simulator.

    python3 tools/selfcheck.py

Every check here corresponds to a bug that actually occurred and cost real
debugging time. They run in a couple of seconds against the model package
directly, which is the point: the mission log cannot distinguish an allocation
fault from an arrival-test fault from a controller fault, and these can.
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np

sys.path.insert(0, ".")

from swarm_gnn.model import sector
from swarm_gnn.model.assignment import INFEASIBLE, hungarian, solve_from_utility
from swarm_gnn.model.coverage import build_search_cells, fair_shares, lloyd_partition
from swarm_gnn.model.planner import GridPlanner
from swarm_gnn.model.rf_localizer import (
    BayesianRFGrid,
    Detection,
    SHADOWING_SIGMA_DB,
    ThreatTrack,
    expected_rssi,
    multilaterate,
    observer_baseline,
    wrap_angle,
)
from swarm_gnn.model.spec import Platform

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            _results.append((name, True, detail))
        except AssertionError as exc:
            _results.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn
    return wrap


# ============================== sector layout ===============================
@check("layout: nothing spawns or hides inside a building")
def _layout():
    problems = sector.validate()
    assert not problems, "; ".join(problems)
    return f"{len(sector.BUILDINGS)} buildings, {len(sector.EMITTERS)} emitters"


@check("layout: both routes are genuinely clear")
def _routes():
    half = sector.ROUTE_HALF_WIDTH
    blocked = [
        (x, y)
        for x in np.arange(-55, 56, 1.0)
        for y in np.arange(-half + 0.6, half - 0.5, 1.0)
        if sector.blocked(x, y, 0.0, include_rubble=False)
    ]
    assert not blocked, f"{len(blocked)} blocked samples on the main route"
    return f"{2 * half:.0f} m wide, clear end to end"


@check("layout: the drone's transit altitude clears every structure")
def _transit():
    cruise = sector.MAX_STRUCTURE_HEIGHT + 5.0
    worst = max(b.height for b in sector.obstacles())
    assert cruise > worst, f"cruise {cruise} m below a {worst} m structure"
    # And the height query must distinguish the two cases.
    tall = max(sector.BUILDINGS, key=lambda b: b.height)
    assert sector.blocked_at_height(tall.x, tall.y, 7.0)
    assert not sector.blocked_at_height(tall.x, tall.y, cruise)
    return f"tallest {worst:.0f} m, transit {cruise:.0f} m"


# ============================== assignment ==================================
@check("hungarian: optimal against brute force")
def _hungarian():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        m = int(rng.integers(n, 7))
        c = rng.random((n, m)) * 10
        r, col = hungarian(c)
        best = min(
            sum(c[i, p[i]] for i in range(n))
            for p in itertools.permutations(range(m), n)
        )
        assert abs(c[r, col].sum() - best) < 1e-9, "suboptimal matching"
        assert len(set(col.tolist())) == len(col), "duplicate column"
    return "200 random matrices, all optimal"


@check("assignment: an urgent task is never left for a cheap one")
def _tiering():
    # One urgent task worth little, three routine tasks worth a lot.
    u = np.array([
        [1.0, 9.0, 9.0, 9.0],
        [-np.inf, 8.0, 8.0, 8.0],
        [-np.inf, 7.0, 7.0, 7.0],
    ])
    feasible = np.isfinite(u)
    plain = solve_from_utility(u, feasible)
    assert not any(k == 0 for _, k, _ in plain), "expected plain Hungarian to skip it"

    urgent, routine = [0], [1, 2, 3]
    stage1 = solve_from_utility(u[:, urgent], feasible[:, urgent])
    taken = {i for i, _, _ in stage1}
    assert any(urgent[k] == 0 for _, k, _ in stage1), "tiering failed to assign it"
    free = [i for i in range(3) if i not in taken]
    sub = solve_from_utility(
        u[np.ix_(free, routine)], feasible[np.ix_(free, routine)]
    )
    assert len(sub) == min(len(free), len(routine))
    return "plain solver skips it, tiered solver assigns it"


# ============================== localisation ================================
@check("localiser: a single observer does not invent an emitter beneath itself")
def _ghost():
    obs = (-52.0, 0.0, 0.3)
    truth = (20.0, 46.0)
    rng = np.random.default_rng(0)
    grid = BayesianRFGrid(60.0, 1.5, tx_power_prior_dbm=22.0)
    for _ in range(40):
        slant = math.hypot(math.dist(obs[:2], truth), obs[2])
        bearing = math.atan2(truth[1] - obs[1], truth[0] - obs[0])
        grid.update(Detection(
            "ugv_1", obs,
            expected_rssi(30.0, slant) + rng.normal(0, SHADOWING_SIGMA_DB),
            wrap_angle(bearing + rng.normal(0, 0.07)), 0.07, 433.0,
        ))
    (px, py), _ = grid.peak()
    d = math.dist((px, py), obs[:2])
    assert d > 20.0, f"peak only {d:.1f} m from the observer"
    return f"peak {d:.0f} m away, bearing error {math.degrees(abs(wrap_angle(math.atan2(py - obs[1], px - obs[0]) - math.atan2(truth[1] - obs[1], truth[0] - obs[0])))):.1f} deg"


@check("localiser: a degenerate DF baseline is refused, not answered")
def _gdop():
    truth = (20.0, 46.0)
    rng = np.random.default_rng(0)
    cuts = []
    for pos in ((-10.0, -5.0, 0.3), (-9.0, -4.0, 0.4)):
        for _ in range(4):
            bearing = math.atan2(truth[1] - pos[1], truth[0] - pos[0])
            cuts.append(Detection(
                "x", pos, -60.0, wrap_angle(bearing + rng.normal(0, 0.07)),
                0.07, 433.0,
            ))
    assert observer_baseline(cuts) < 2.0
    assert multilaterate(cuts, (20.0, 40.0)) is None, "accepted a 1.4 m baseline"
    return "1.4 m baseline rejected"


@check("localiser: a boundary peak is treated as truncation, not an emitter")
def _boundary():
    grid = BayesianRFGrid(60.0, 1.5)
    grid.log_belief[:] = -40.0
    grid.log_belief[-1, -1] = 0.0      # the corner
    grid.log_belief[65, 49] = -0.5     # a genuine interior peak
    assert max(abs(v) for v in grid.peak()[0]) > 56.0, "fixture did not peak at the edge"
    modes = grid.modes(min_separation=10.0, max_modes=4,
                       mass_threshold=1e-6, edge_margin=6.0)
    assert modes, "the interior peak was discarded too"
    assert all(max(abs(x), abs(y)) <= 56.0 for (x, y), _ in modes), "boundary mode kept"
    return "corner peak rejected, interior peak retained"


@check("localiser: class is derived from evidence, so the jammer is engageable")
def _classify():
    cases = {
        1: {"thermal_hot", "jam_detected", "cbrn_neg"},   # jammer
        2: {"thermal_hot", "cbrn_pos"},                   # IED trigger
        3: {"thermal_hot", "no_jam", "cbrn_neg"},         # civilian, no-strike
        0: {"thermal_hot"},                               # not yet decidable
    }
    for expected, evidence in cases.items():
        track = ThreatTrack("t", (0.0, 0.0))
        track.evidence = set(evidence)
        got = track.classify()
        assert got == expected, f"{sorted(evidence)} -> {got}, want {expected}"
    return "jammer=1, trigger=2, civilian=3, partial=unknown"


# ================================ coverage ==================================
@check("coverage: the sweep is shared, not monopolised by the fastest platform")
def _lloyd():
    cells = build_search_cells(
        sector.SEARCH_HALF, 16.0, sector.ROUTE_HALF_WIDTH, sector.UGV_EXCLUSIONS
    )
    agents = [
        ("drone_1", Platform.DRONE, sector.INSERTION),
        ("ugv_1", Platform.UGV, sector.INSERTION),
        ("dog_1", Platform.QUADRUPED, sector.INSERTION),
    ]
    regions, _ = lloyd_partition(cells, agents)
    assert sum(len(r) for r in regions.values()) == len(cells), "cells lost"
    targets = fair_shares(agents)
    total = sum(c.weight for c in cells)
    for agent_id, region in regions.items():
        share = sum(c.weight for c in region) / total
        assert abs(share - targets[agent_id]) < 0.08, (
            f"{agent_id} got {share:.0%}, target {targets[agent_id]:.0%}"
        )
    assert all(
        Platform.UGV not in c.excluded for c in regions["ugv_1"]
    ), "UGV routed into rubble"
    return "  ".join(
        f"{a}={sum(c.weight for c in r) / total:.0%}" for a, r in regions.items()
    )


# ================================ planning ==================================
@check("planner: every mission destination is reachable by the platform sent")
def _planner():
    planner = GridPlanner()
    destinations = [(e.name, (e.x, e.y)) for e in sector.EMITTERS]
    destinations.append(("rally", sector.RALLY))
    lines = []
    for platform in (Platform.UGV, Platform.QUADRUPED):
        for name, goal in destinations:
            result = planner.plan(platform, sector.INSERTION, goal)
            assert result, f"{platform.name} has no plan to {name}"
            assert result.reaches_goal, f"{platform.name} cannot reach {name}"
            straight = math.dist(sector.INSERTION, goal)
            assert result.cost >= straight - 1e-6, (
                f"{platform.name}->{name} cost {result.cost:.1f} m beats the "
                f"straight line {straight:.1f} m, which is impossible"
            )
            lines.append(result.cost / straight)
    return f"{len(lines)} routes, worst detour {max(lines):.2f}x direct"


def main() -> int:
    width = max(len(name) for name, _, _ in _results)
    failed = 0
    for name, ok, detail in _results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
