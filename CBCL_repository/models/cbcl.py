from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from models.contrastive import multi_positive_infonce
from models.semantic_branch import SemanticBranch
from models.structural_branch import StructuralBranch
from preprocessing.feature_transform import TypeFeatureTransform
from preprocessing.metapath import PositiveIndex, PreparedGraph


@dataclass
class CBCLOutput:
    structural: torch.Tensor
    semantic: torch.Tensor
    projected_structural: torch.Tensor
    projected_semantic: torch.Tensor


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBCL(nn.Module):
    def __init__(
        self,
        input_dims: list[int | None],
        num_nodes: list[int],
        n_relations: int,
        n_metapaths: int,
        hidden_dim: int = 64,
        projection_dim: int = 64,
        structural_layers: int = 2,
        gat_heads: int = 4,
        transformer_heads: int = 4,
        dropout: float = 0.2,
        semantic_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_transform = TypeFeatureTransform(input_dims, num_nodes, hidden_dim, dropout)
        self.structural_branch = StructuralBranch(hidden_dim, n_relations, n_metapaths, structural_layers, dropout)
        self.semantic_branch = SemanticBranch(hidden_dim, n_metapaths, gat_heads, transformer_heads, dropout, semantic_beta)
        self.structural_projection = ProjectionHead(hidden_dim, projection_dim, dropout)
        self.semantic_projection = ProjectionHead(hidden_dim * n_metapaths, projection_dim, dropout)

    def forward(self, features: list[torch.Tensor | None], graph: PreparedGraph, target_type: int, target_n: int) -> CBCLOutput:
        all_h, per_type = self.feature_transform(features)
        structural_all = self.structural_branch(all_h, graph.relation_edges, graph.metapath_norm_edges)
        start = graph.offsets[target_type]
        structural = structural_all[start:start + target_n]
        semantic = self.semantic_branch(per_type[target_type], graph.metapath_edges)
        return CBCLOutput(structural, semantic, self.structural_projection(structural), self.semantic_projection(semantic))

    def loss(
        self,
        output: CBCLOutput,
        positives: PositiveIndex,
        temperature: float,
        direction_weight: float,
        chunk_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not 0 <= direction_weight <= 1:
            raise ValueError("direction_weight must be in [0, 1]")
        c_to_s = multi_positive_infonce(output.projected_structural, output.projected_semantic, positives, temperature, chunk_size)
        s_to_c = multi_positive_infonce(output.projected_semantic, output.projected_structural, positives, temperature, chunk_size)
        loss = direction_weight * c_to_s + (1.0 - direction_weight) * s_to_c
        return loss, {
            "loss": float(loss.detach().cpu()),
            "c_to_s": float(c_to_s.detach().cpu()),
            "s_to_c": float(s_to_c.detach().cpu()),
        }

    @torch.no_grad()
    def embed(self, features: list[torch.Tensor | None], graph: PreparedGraph, target_type: int, target_n: int) -> torch.Tensor:
        self.eval()
        return F.normalize(self.forward(features, graph, target_type, target_n).semantic, dim=-1)
