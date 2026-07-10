from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TypeFeatureTransform(nn.Module):
    """Type-specific feature projection corresponding to Eq. (2).

    A featureless node type uses a trainable embedding. This is equivalent to
    projecting one-hot identity features, while avoiding a huge dense identity matrix.
    """

    def __init__(
        self,
        input_dims: list[int | None],
        num_nodes: list[int],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleList()
        self.featureless = []
        for dim, n_nodes in zip(input_dims, num_nodes):
            if dim is None:
                self.transforms.append(nn.Embedding(n_nodes, hidden_dim))
                self.featureless.append(True)
            else:
                self.transforms.append(nn.Linear(dim, hidden_dim))
                self.featureless.append(False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: list[torch.Tensor | None]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        projected: list[torch.Tensor] = []
        for transform, is_featureless, feature in zip(self.transforms, self.featureless, features):
            if is_featureless:
                assert isinstance(transform, nn.Embedding)
                h = transform.weight
            else:
                if feature is None:
                    raise ValueError("A feature tensor is missing for a feature-based node type")
                h = transform(self.dropout(feature))
            projected.append(F.relu(h))
        return torch.cat(projected, dim=0), projected
