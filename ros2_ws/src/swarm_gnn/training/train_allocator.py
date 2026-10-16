#!/usr/bin/env python3
"""Train the allocator GNN by imitating an optimal Hungarian oracle.

Why imitation and not RL
========================
The assignment subproblem has an exact solver. Given a cost matrix, Hungarian
is optimal === there is nothing to learn about the combinatorics. What is *not*
available at run time is a good cost matrix: the analytic cost in
`model/assignment.pairing_cost` is myopic, scoring one pairing in isolation with
no knowledge of what the rest of the team is doing.

So the training signal is: run the oracle on the analytic cost, take its
optimal matching as the label, and teach the network to reproduce that matching
*from the graph*. The network starts by learning the oracle's own trade-offs
(distance, capability, risk, energy, deadlines). Where it goes beyond the oracle
is the part the oracle cannot see: because its embeddings pass messages along
agent-agent and task-task edges, it conditions each pairing's value on team
geometry and task clustering, which no per-pair cost function can express.

Reinforcement learning would be the way to push past the oracle's *policy*
rather than its representation, and the reward would have to come from full
mission rollouts. That is a much larger undertaking and is not what this
checkpoint claims to be: this is a learned, relational approximation of a
known-good cost, trained to be computed in one forward pass over a graph of any
size.

Losses
======
  assignment CE   per agent, cross-entropy over its feasible tasks toward the
                  oracle's choice. The primary signal.
  listwise rank   soft-ranking loss against the normalised oracle costs, so the
                  whole utility row is calibrated, not just its argmax. This is
                  what makes the Hungarian solve downstream meaningful === it
                  consumes the magnitudes, not only the ordering.
  margin          hinge pushing the chosen pairing's utility above the best
                  alternative by a margin, which stabilises the matching
                  against the hysteresis bias applied at run time.

Usage
=====
    python3 -m training.train_allocator --epochs 40 --out checkpoints/allocator.pt
    ros2 run swarm_gnn train_allocator -- --epochs 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from swarm_gnn.model.assignment import INFEASIBLE, hungarian, oracle_cost_matrix
from swarm_gnn.model.graph_builder import GraphConfig, build_graph
from swarm_gnn.model.het_gat import SwarmAllocatorGNN, count_parameters

try:
    from training.scenario_gen import sample_scenario
except ImportError:  # running as a script from inside training/
    from scenario_gen import sample_scenario


def oracle_labels(cost: np.ndarray) -> np.ndarray:
    """Optimal task index per agent, -1 for an agent left unassigned."""
    labels = np.full(cost.shape[0], -1, dtype=np.int64)
    if cost.size == 0:
        return labels
    rows, cols = hungarian(cost)
    for i, j in zip(rows, cols):
        if cost[i, j] < INFEASIBLE:
            labels[i] = j
    return labels


def make_example(rng: random.Random, cfg: GraphConfig) -> dict | None:
    """One training example: a graph, oracle labels and the oracle costs."""
    agents, tasks, mission_time = sample_scenario(rng)
    graph = build_graph(agents, tasks, cfg, mission_time)
    if graph["at_index"].numel() == 0:
        return None

    cost = oracle_cost_matrix(agents, tasks, mission_time)
    # The oracle and the graph must agree on what is legal. They derive
    # feasibility independently (one from `pairing_cost`, one from the edge
    # gating), so any divergence === a range gate, a resource rule === would let
    # the oracle label a pairing the network has no edge for, whose utility is
    # -inf, making the cross-entropy infinite. Masking the cost by the graph's
    # own feasibility makes that impossible by construction.
    feasible = graph["feasible"].numpy()
    cost = np.where(feasible, cost, INFEASIBLE)

    labels = oracle_labels(cost)
    if (labels < 0).all():
        # Nothing feasible: no signal in this scenario.
        return None

    return {
        "graph": graph,
        "labels": torch.from_numpy(labels),
        "cost": torch.from_numpy(cost).float(),
        "feasible": graph["feasible"],
    }


def losses_for(
    out: dict, example: dict, rank_weight: float, margin_weight: float, margin: float
) -> tuple[torch.Tensor, dict]:
    """Compute the three losses and the diagnostics for one example."""
    utility = out["utility"]
    labels = example["labels"]
    feasible = example["feasible"]
    cost = example["cost"]

    valid = labels >= 0
    if not bool(valid.any()):
        return utility.sum() * 0.0, {}

    # ------------------------------ assignment CE -------------------------
    # -inf on infeasible entries is exactly what cross_entropy needs to
    # exclude them, but a row with no finite entry would produce NaN; those
    # rows are filtered by `valid` because the oracle could not place them.
    logits = utility[valid]
    target = labels[valid]
    ce = F.cross_entropy(logits, target)

    # ------------------------------ listwise rank -------------------------
    # Turn the oracle's costs into a target distribution (low cost -> high
    # mass) and match the utility row to it in KL. Infeasible entries get zero
    # target mass and are masked out of the prediction.
    with torch.no_grad():
        masked_cost = cost[valid].clone()
        masked_cost[~feasible[valid]] = float("inf")
        # Temperature in cost units: 15 s of travel time is a meaningful gap.
        target_dist = torch.softmax(-masked_cost / 15.0, dim=-1)
        target_dist = torch.nan_to_num(target_dist, nan=0.0)
    log_pred = torch.log_softmax(logits, dim=-1)
    # Infeasible entries are -inf in `logits`, hence -inf in log_pred, and they
    # carry zero target mass. 0 * -inf is NaN, so the -inf is floored to a
    # finite, very negative value first; the zero target then kills the term
    # cleanly and no NaN reaches the backward pass.
    log_pred = torch.where(
        torch.isfinite(log_pred), log_pred, torch.full_like(log_pred, -60.0)
    )
    rank = -(target_dist * log_pred).sum(dim=-1).mean()

    # --------------------------------- margin -----------------------------
    chosen = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    # Best feasible alternative per row.
    alt = logits.clone()
    alt.scatter_(1, target.unsqueeze(1), float("-inf"))
    best_alt = alt.max(dim=-1).values
    # Rows whose only feasible task is the chosen one have no alternative.
    has_alt = torch.isfinite(best_alt)
    if bool(has_alt.any()):
        hinge = F.relu(margin - (chosen[has_alt] - best_alt[has_alt])).mean()
    else:
        hinge = utility.sum() * 0.0

    total = ce + rank_weight * rank + margin_weight * hinge

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        top1 = float((pred == target).float().mean())
        # Cost regret: the analytic cost of the network's own argmax minus the
        # cost of the task the oracle assigned to that agent, in seconds.
        #
        # Read this alongside top-1, not instead of it, and note that it can go
        # *negative* without the policy being better. The oracle optimises the
        # joint matching, so it will hand one agent a locally worse task in
        # order to free a better one for a teammate; a per-row comparison
        # against that choice rewards exactly the myopic behaviour the joint
        # solve was avoiding. Negative regret therefore means "more greedy than
        # the oracle", not "better than the oracle". Top-1 agreement is the
        # headline number; regret is useful for catching the opposite failure,
        # where the policy is confidently assigning something very expensive.
        picked_cost = cost[valid].gather(1, pred.unsqueeze(1)).squeeze(1)
        best_cost = cost[valid].gather(1, target.unsqueeze(1)).squeeze(1)
        finite = torch.isfinite(picked_cost) & (picked_cost < INFEASIBLE)
        regret = float((picked_cost[finite] - best_cost[finite]).mean()) if bool(finite.any()) else 0.0
        infeasible_picks = int((~finite).sum())

    return total, {
        "ce": float(ce.detach()),
        "rank": float(rank.detach()),
        "hinge": float(hinge.detach()),
        "top1": top1,
        "regret": regret,
        "infeasible_picks": infeasible_picks,
        "rows": int(valid.sum()),
    }


def evaluate(model: SwarmAllocatorGNN, examples: list[dict], args) -> dict:
    model.eval()
    agg = {"top1": 0.0, "regret": 0.0, "rows": 0, "infeasible_picks": 0, "n": 0}
    with torch.no_grad():
        for ex in examples:
            out = model(ex["graph"])
            _, stats = losses_for(
                out, ex, args.rank_weight, args.margin_weight, args.margin
            )
            if not stats:
                continue
            agg["top1"] += stats["top1"]
            agg["regret"] += stats["regret"]
            agg["rows"] += stats["rows"]
            agg["infeasible_picks"] += stats["infeasible_picks"]
            agg["n"] += 1
    n = max(1, agg["n"])
    return {
        "top1": agg["top1"] / n,
        "regret": agg["regret"] / n,
        "infeasible_picks": agg["infeasible_picks"],
        "rows": agg["rows"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-scenarios", type=int, default=1200)
    ap.add_argument("--val-scenarios", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="graphs accumulated per optimiser step")
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rank-weight", type=float, default=0.6)
    ap.add_argument("--margin-weight", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/allocator.pt")
    ap.add_argument("--log-json", type=str, default="",
                    help="also write the training curve as JSON here")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = GraphConfig()

    print("generating scenarios...", flush=True)
    train_set: list[dict] = []
    while len(train_set) < args.train_scenarios:
        ex = make_example(rng, cfg)
        if ex is not None:
            train_set.append(ex)
    # Validation comes from a disjoint seed stream, so a scenario never leaks.
    val_rng = random.Random(args.seed + 99991)
    val_set: list[dict] = []
    while len(val_set) < args.val_scenarios:
        ex = make_example(val_rng, cfg)
        if ex is not None:
            val_set.append(ex)

    model = SwarmAllocatorGNN(
        hidden=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_set) // args.batch_size),
        pct_start=0.25,
    )

    print(
        f"model: {count_parameters(model)} parameters "
        f"(hidden={args.hidden}, layers={args.layers}, heads={args.heads})\n"
        f"data:  {len(train_set)} train / {len(val_set)} val graphs\n"
    )

    baseline = evaluate(model, val_set, args)
    print(
        f"untrained baseline: top-1 {baseline['top1']:.3f}  "
        f"regret {baseline['regret']:.2f} s  "
        f"infeasible picks {baseline['infeasible_picks']}"
    )

    history = []
    best_top1 = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_set)
        epoch_stats = {"ce": 0.0, "rank": 0.0, "hinge": 0.0, "top1": 0.0, "n": 0}

        opt.zero_grad(set_to_none=True)
        in_batch = 0
        for ex in train_set:
            out = model(ex["graph"])
            loss, stats = losses_for(
                out, ex, args.rank_weight, args.margin_weight, args.margin
            )
            if not stats:
                continue
            # Graphs have different numbers of agents, so average over the
            # batch rather than summing: otherwise a 3-agent graph dominates.
            (loss / args.batch_size).backward()
            in_batch += 1
            for k in ("ce", "rank", "hinge", "top1"):
                epoch_stats[k] += stats[k]
            epoch_stats["n"] += 1

            if in_batch >= args.batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                in_batch = 0

        if in_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        n = max(1, epoch_stats["n"])
        val = evaluate(model, val_set, args)
        row = {
            "epoch": epoch,
            "train_ce": epoch_stats["ce"] / n,
            "train_rank": epoch_stats["rank"] / n,
            "train_hinge": epoch_stats["hinge"] / n,
            "train_top1": epoch_stats["top1"] / n,
            "val_top1": val["top1"],
            "val_regret": val["regret"],
            "val_infeasible": val["infeasible_picks"],
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"ce {row['train_ce']:.4f}  rank {row['train_rank']:.4f}  "
            f"hinge {row['train_hinge']:.4f}  "
            f"train top-1 {row['train_top1']:.3f}  "
            f"val top-1 {row['val_top1']:.3f}  "
            f"val regret {row['val_regret']:5.2f} s",
            flush=True,
        )

        if val["top1"] > best_top1:
            best_top1 = val["top1"]
            model.save(
                args.out,
                extra={
                    "epochs": epoch,
                    "val_top1": val["top1"],
                    "val_regret": val["regret"],
                    "val_infeasible_picks": val["infeasible_picks"],
                    "untrained_top1": baseline["top1"],
                    "untrained_regret": baseline["regret"],
                    "args": vars(args),
                    "spec_version": 1,
                },
            )

    elapsed = time.time() - t0
    print(
        f"\ndone in {elapsed:.0f} s. best val top-1 {best_top1:.3f} "
        f"(untrained {baseline['top1']:.3f}), checkpoint -> {args.out}"
    )
    # An infeasible pick would mean the capability gating leaked, which would be
    # a correctness bug rather than a quality one; call it out explicitly.
    final = evaluate(model, val_set, args)
    print(
        f"final: val regret {final['regret']:.2f} s over {final['rows']} assignment "
        f"decisions, infeasible picks {final['infeasible_picks']} "
        f"(must be 0 === the graph gates them)"
    )

    if args.log_json:
        with open(args.log_json, "w") as fh:
            json.dump({"history": history, "args": vars(args)}, fh, indent=2)
        print(f"curve -> {args.log_json}")


if __name__ == "__main__":
    main()
