from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import MetaPathTransformer, PathMLP, SparseGraphAttention


class SemanticBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_metapaths: int,
        gat_heads: int,
        transformer_heads: int,
        dropout: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.gats = nn.ModuleList(SparseGraphAttention(hidden_dim, gat_heads, dropout) for _ in range(n_metapaths))
        self.mlps = nn.ModuleList(PathMLP(hidden_dim, dropout) for _ in range(n_metapaths))
        self.transformer = MetaPathTransformer(hidden_dim, transformer_heads, dropout, beta)

    def forward(self, target_h: torch.Tensor, metapath_edges: list[torch.Tensor]) -> torch.Tensor:
        if len(metapath_edges) != len(self.gats):
            raise ValueError("Meta-path edge count does not match the semantic encoders")
        path_tokens = [
            mlp(F.elu(gat(target_h, edge)))
            for gat, mlp, edge in zip(self.gats, self.mlps, metapath_edges)
        ]
        tokens = torch.stack(path_tokens, dim=1)
        fused = self.transformer(tokens)
        return fused.flatten(start_dim=1)
