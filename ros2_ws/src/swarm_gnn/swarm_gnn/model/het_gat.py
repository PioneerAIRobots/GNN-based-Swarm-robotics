"""Heterogeneous graph attention network for swarm task allocation.

Why a GNN at all
================
Task allocation in a heterogeneous team is not a per-agent decision. Whether the
dog should take a CBRN sample depends on where the UGV is (it may be closer to
the next marker-emplacement), on whether the drone is currently the only comms
bridge, and on what *other* tasks are clustered near that sample. That is a
relational problem over a graph whose size changes every tick as tasks open and
close, and whose agents only see neighbours inside radio range.

A GNN is the natural function class: it is permutation-equivariant over agents
and tasks, handles a variable node count with one set of weights, and respects
the communication topology because messages only flow along edges that exist.
Attention, specifically, lets an agent learn *which* neighbour matters === the
drone attends strongly to a ground unit it is relaying for, weakly to one on the
far side of the sector.

Architecture
============
  Bipartite-plus graph, three edge types:
      agent-agent  communication links, gated by range and line of sight
      agent-task   candidate assignments, gated by capability feasibility
      task-task    spatial coupling, so a cluster of tasks is seen as a cluster

  Per layer, per edge type: an independent multi-head attention operator with
  edge features folded into the attention logits (an edge-conditioned GATv2).
  The three type-specific messages are summed, passed through a residual MLP
  and layer-normed. Agent and task nodes have separate encoders and separate
  per-type weights at every layer: this is a genuinely heterogeneous network,
  not one GAT over a homogeneous relabelled graph.

  Heads on top of the final embeddings:
      utility head   scores every agent-task edge  -> the assignment matrix
      relay head     per agent, P(this agent should hold as a comms relay)
      caution head   per agent, a speed scalar in [0,1] for risk-aware motion

Implementation note
===================
torch_geometric is not a dependency. The message passing below is written
directly against dense per-edge index tensors with `index_add_` /
segment-softmax, which is all PyG's GATConv does under the hood. For the graph
sizes here (3 agents, tens of tasks) this is faster than the generic library
path and keeps the package installable with nothing but torch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import (
    AGENT_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    TASK_FEATURE_DIM,
    EdgeType,
)


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax over the groups defined by `index`.

    `logits` is [E, H], `index` is [E] giving each edge's destination node.
    Normalisation runs per destination node, per head, which is what makes this
    attention rather than a plain weighted sum. Returns [E, H].
    """
    if logits.numel() == 0:
        return logits

    heads = logits.shape[1]
    # Subtract the per-segment max for numerical stability.
    max_per_seg = logits.new_full((num_segments, heads), float("-inf"))
    max_per_seg.scatter_reduce_(
        0, index.unsqueeze(-1).expand(-1, heads), logits, reduce="amax", include_self=True
    )
    # Destinations with no incoming edge keep -inf; clamp so they do not poison
    # the exp below (their rows are never read, but NaNs propagate in autograd).
    max_per_seg = torch.nan_to_num(max_per_seg, neginf=0.0)

    centred = logits - max_per_seg[index]
    exp = centred.exp()

    denom = logits.new_zeros((num_segments, heads))
    denom.index_add_(0, index, exp)
    return exp / (denom[index] + 1e-16)


class EdgeTypeAttention(nn.Module):
    """Edge-conditioned multi-head attention for a single relation.

    Follows GATv2 (Brody et al., 2022): the attention score is computed
    *after* the nonlinearity, which makes the attention dynamic rather than
    a fixed per-node ranking. Edge features are concatenated into the scoring
    input, so distance and feasibility modulate attention directly.
    """

    def __init__(
        self,
        src_dim: int,
        dst_dim: int,
        edge_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError(f"out_dim {out_dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_src = nn.Linear(src_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(dst_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)
        # Scoring vector applied per head after the nonlinearity.
        self.att = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.lin_src, self.lin_dst, self.lin_edge):
            nn.init.xavier_uniform_(lin.weight, gain=1.0)
        nn.init.xavier_uniform_(self.att.view(1, -1), gain=1.0)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x_src: torch.Tensor,     # [N_src, src_dim]
        x_dst: torch.Tensor,     # [N_dst, dst_dim]
        edge_index: torch.Tensor,  # [2, E] rows (src, dst)
        edge_attr: torch.Tensor,   # [E, edge_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (aggregated [N_dst, out_dim], attention [E, heads])."""
        n_dst = x_dst.shape[0]
        if edge_index.numel() == 0:
            return (
                x_dst.new_zeros((n_dst, self.out_dim)),
                x_dst.new_zeros((0, self.heads)),
            )

        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = self.lin_src(x_src).view(-1, self.heads, self.head_dim)
        h_dst = self.lin_dst(x_dst).view(-1, self.heads, self.head_dim)
        h_edge = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)

        # GATv2: combine, activate, then score.
        combined = h_src[src_idx] + h_dst[dst_idx] + h_edge
        scores = F.leaky_relu(combined, self.negative_slope)
        logits = (scores * self.att.unsqueeze(0)).sum(dim=-1)  # [E, heads]

        alpha = segment_softmax(logits, dst_idx, n_dst)
        if self.training and self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Messages carry the edge features too, not just the source state.
        messages = (h_src[src_idx] + h_edge) * alpha.unsqueeze(-1)
        out = x_dst.new_zeros((n_dst, self.heads, self.head_dim))
        out.index_add_(0, dst_idx, messages)
        return out.reshape(n_dst, self.out_dim) + self.bias, alpha


class HeteroGATLayer(nn.Module):
    """One round of message passing over all three relations.

    Agent nodes receive from other agents (the comms graph) and from tasks
    (what is on offer nearby). Task nodes receive from agents (who is competing
    for me) and from neighbouring tasks (am I part of a cluster). Both node
    types then go through their own residual update.
    """

    def __init__(self, hidden: int, edge_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.agent_from_agent = EdgeTypeAttention(hidden, hidden, edge_dim, hidden, heads, dropout)
        self.agent_from_task = EdgeTypeAttention(hidden, hidden, edge_dim, hidden, heads, dropout)
        self.task_from_agent = EdgeTypeAttention(hidden, hidden, edge_dim, hidden, heads, dropout)
        self.task_from_task = EdgeTypeAttention(hidden, hidden, edge_dim, hidden, heads, dropout)

        self.agent_update = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.task_update = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.agent_norm = nn.LayerNorm(hidden)
        self.task_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_agent: torch.Tensor,
        h_task: torch.Tensor,
        edges: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        aa, att_aa = self.agent_from_agent(
            h_agent, h_agent, edges["aa_index"], edges["aa_attr"]
        )
        # agent-task edges are stored once, src=agent dst=task, and used in
        # both directions by swapping the index rows.
        at_index = edges["at_index"]
        at_attr = edges["at_attr"]
        ta_index = at_index.flip(0) if at_index.numel() else at_index

        at, att_at = self.agent_from_task(h_task, h_agent, ta_index, at_attr)
        ta, att_ta = self.task_from_agent(h_agent, h_task, at_index, at_attr)
        tt, att_tt = self.task_from_task(h_task, h_task, edges["tt_index"], edges["tt_attr"])

        h_agent = self.agent_norm(
            h_agent + self.dropout(self.agent_update(torch.cat([h_agent, aa, at], dim=-1)))
        )
        h_task = self.task_norm(
            h_task + self.dropout(self.task_update(torch.cat([h_task, ta, tt], dim=-1)))
        )
        attention = {"aa": att_aa, "at": att_at, "ta": att_ta, "tt": att_tt}
        return h_agent, h_task, attention


class SwarmAllocatorGNN(nn.Module):
    """The policy network: graph in, utility matrix and per-agent heads out."""

    def __init__(
        self,
        hidden: int = 96,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        agent_dim: int = AGENT_FEATURE_DIM,
        task_dim: int = TASK_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.config = {
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "dropout": dropout,
            "agent_dim": agent_dim,
            "task_dim": task_dim,
            "edge_dim": edge_dim,
        }

        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        self.gnn_layers = nn.ModuleList(
            [HeteroGATLayer(hidden, edge_dim, heads, dropout) for _ in range(layers)]
        )

        # Utility head: scores a (agent, task) pair from both embeddings, the
        # edge features, and their elementwise product (a cheap bilinear term
        # that lets the head express "this agent matches this task" directly).
        self.utility_head = nn.Sequential(
            nn.Linear(hidden * 3 + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # A learned global context token, mean-pooled over the graph, gives each
        # per-agent head a view of the whole mission state.
        self.relay_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )
        self.caution_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """`graph` is the dict produced by graph_builder.build_graph.

        Returns:
            utility   [N_agents, N_tasks]  scores; -inf where infeasible
            relay     [N_agents]           sigmoid, relay desirability
            caution   [N_agents]           sigmoid, speed scalar
            attention dict of per-edge attention, for the GraphDebug topic
        """
        h_agent = self.agent_encoder(graph["agent_x"])
        h_task = self.task_encoder(graph["task_x"])

        attention: dict[str, torch.Tensor] = {}
        for layer in self.gnn_layers:
            h_agent, h_task, attention = layer(h_agent, h_task, graph)

        n_agents = h_agent.shape[0]
        n_tasks = h_task.shape[0]

        # Global context: the mean of every node embedding in the graph.
        if n_tasks > 0:
            context = torch.cat([h_agent, h_task], dim=0).mean(dim=0, keepdim=True)
        else:
            context = h_agent.mean(dim=0, keepdim=True)
        context = context.expand(n_agents, -1)

        relay = torch.sigmoid(self.relay_head(torch.cat([h_agent, context], -1))).squeeze(-1)
        caution = torch.sigmoid(self.caution_head(torch.cat([h_agent, context], -1))).squeeze(-1)

        utility = h_agent.new_full((n_agents, n_tasks), float("-inf"))
        at_index = graph["at_index"]
        if at_index.numel() > 0 and n_tasks > 0:
            a_idx, t_idx = at_index[0], at_index[1]
            ha = h_agent[a_idx]
            ht = h_task[t_idx]
            pair = torch.cat([ha, ht, ha * ht, graph["at_attr"]], dim=-1)
            scores = self.utility_head(pair).squeeze(-1)
            # Only edges that exist get a score; everything else stays -inf,
            # which is how infeasible pairings are excluded from the solver.
            utility = utility.index_put((a_idx, t_idx), scores)

        return {
            "utility": utility,
            "relay": relay,
            "caution": caution,
            "attention": attention,
        }

    # ------------------------------------------------------------------ io
    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "format_version": 1,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> tuple["SwarmAllocatorGNN", dict]:
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model, blob.get("extra", {})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
