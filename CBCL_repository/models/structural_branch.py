from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RelationAwareLayer(nn.Module):
    """RGCN-style relation aggregation with a parallel meta-path structural channel."""

    def __init__(self, hidden_dim: int, n_relations: int, n_metapaths: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_linears = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_relations))
        self.metapath_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.metapath_logits = nn.Parameter(torch.zeros(n_metapaths))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _aggregate(h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_zeros(h.shape)
        src, dst = edge_index
        messages = linear(h[src]) * edge_weight.unsqueeze(-1)
        out = h.new_zeros(h.shape)
        out.index_add_(0, dst, messages)
        return out

    def forward(
        self,
        h: torch.Tensor,
        relation_edges: list[tuple[str, torch.Tensor, torch.Tensor]],
        metapath_edges: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        out = self.self_linear(h)
        for linear, (_, edge, weight) in zip(self.relation_linears, relation_edges):
            out = out + self._aggregate(h, edge, weight, linear)
        gates = torch.softmax(self.metapath_logits, dim=0)
        for gate, (edge, weight) in zip(gates, metapath_edges):
            out = out + gate * self._aggregate(h, edge, weight, self.metapath_linear)
        return self.dropout(F.relu(self.norm(out + self.bias)))


class StructuralBranch(nn.Module):
    def __init__(self, hidden_dim: int, n_relations: int, n_metapaths: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            RelationAwareLayer(hidden_dim, n_relations, n_metapaths, dropout)
            for _ in range(layers)
        )

    def forward(
        self,
        h: torch.Tensor,
        relation_edges: list[tuple[str, torch.Tensor, torch.Tensor]],
        metapath_edges: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, relation_edges, metapath_edges)
        return h
