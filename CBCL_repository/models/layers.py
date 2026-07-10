from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SparseGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        dst, src = edge_index
        h = self.proj(x).reshape(n, self.heads, self.head_dim)
        score = (h[dst] * self.att_dst).sum(-1) + (h[src] * self.att_src).sum(-1)
        score = F.leaky_relu(score, negative_slope=0.2)
        index = dst[:, None].expand(-1, self.heads)

        maximum = torch.full((n, self.heads), -torch.inf, dtype=score.dtype, device=score.device)
        maximum.scatter_reduce_(0, index, score, reduce="amax", include_self=True)
        exp_score = torch.exp(score - maximum[dst])
        denominator = torch.zeros((n, self.heads), dtype=score.dtype, device=score.device)
        denominator.scatter_add_(0, index, exp_score)
        alpha = self.dropout(exp_score / denominator[dst].clamp_min(1e-12))

        messages = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros((n, self.heads, self.head_dim), dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, messages)
        return self.out_proj(out.reshape(n, self.hidden_dim))


class PathMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaPathTransformer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, beta: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        x = self.norm1(tokens + self.beta * self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))
