from __future__ import annotations

import torch
import torch.nn.functional as F

from preprocessing.metapath import PositiveIndex


def multi_positive_infonce(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    positives: PositiveIndex,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """Exact full-negative multi-positive InfoNCE in anchor chunks."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    anchors = F.normalize(anchors, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    if anchors.shape[0] != candidates.shape[0]:
        raise ValueError("Both branches must contain the same target nodes")
    n = anchors.shape[0]
    chunk_size = n if chunk_size <= 0 else chunk_size
    total = anchors.new_zeros(())
    count = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ids = torch.arange(start, end, device=anchors.device)
        logits = anchors[start:end] @ candidates.T / temperature
        positive_mask = positives.mask(ids, anchors.device)
        numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        total = total + (denominator - numerator).sum()
        count += end - start
    return total / count
