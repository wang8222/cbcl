from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch

from preprocessing.pathsim import pathsim_topk
from utils.data import HINData


@dataclass
class PositiveIndex:
    indptr: np.ndarray
    indices: np.ndarray
    n_nodes: int

    def mask(self, anchor_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        anchors = anchor_ids.detach().cpu().numpy()
        mask = torch.zeros((len(anchors), self.n_nodes), dtype=torch.bool, device=device)
        for row, node in enumerate(anchors):
            start, end = self.indptr[node], self.indptr[node + 1]
            columns = torch.as_tensor(self.indices[start:end], dtype=torch.long, device=device)
            mask[row, columns] = True
        return mask


@dataclass
class PreparedGraph:
    relation_edges: list[tuple[str, torch.Tensor, torch.Tensor]]
    metapath_edges: list[torch.Tensor]
    metapath_norm_edges: list[tuple[torch.Tensor, torch.Tensor]]
    positive_index: PositiveIndex
    filtered_metapaths: list[sp.csr_matrix]
    offsets: list[int]
    total_nodes: int

    def to(self, device: torch.device) -> "PreparedGraph":
        return PreparedGraph(
            [(name, edge.to(device), weight.to(device)) for name, edge, weight in self.relation_edges],
            [edge.to(device) for edge in self.metapath_edges],
            [(edge.to(device), weight.to(device)) for edge, weight in self.metapath_norm_edges],
            self.positive_index,
            self.filtered_metapaths,
            self.offsets,
            self.total_nodes,
        )


def _offsets(sizes: Sequence[int]) -> list[int]:
    return np.cumsum([0] + list(sizes[:-1])).astype(int).tolist()


def _relation_edges(matrix: sp.csr_matrix, src_offset: int, dst_offset: int, total_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    coo = matrix.tocoo()
    src = torch.from_numpy(coo.row.astype(np.int64)) + src_offset
    dst = torch.from_numpy(coo.col.astype(np.int64)) + dst_offset
    edge = torch.stack([src, dst], dim=0)
    degree = torch.bincount(dst, minlength=total_nodes).float().clamp_min(1.0)
    weight = 1.0 / degree[dst]
    return edge, weight


def _semantic_edges(matrix: sp.csr_matrix) -> torch.Tensor:
    with_self = matrix.maximum(sp.eye(matrix.shape[0], dtype=np.float32, format="csr"))
    coo = with_self.tocoo()
    # destination first, source second, matching SparseGAT.
    return torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))


def _symmetric_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    inv_sqrt[degree > 0] = degree[degree > 0] ** -0.5
    d = sp.diags(inv_sqrt)
    return (d @ matrix @ d).tocsr()


def _target_global_edges(matrix: sp.csr_matrix, target_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    norm = _symmetric_normalize(matrix)
    coo = norm.tocoo()
    # Source=column, destination=row for message passing.
    edge = torch.from_numpy(np.vstack([coo.col + target_offset, coo.row + target_offset]).astype(np.int64))
    weight = torch.from_numpy(coo.data.astype(np.float32))
    return edge, weight


def prepare_graph(
    data: HINData,
    topk: Sequence[int],
    positive_threshold: int = 0,
    symmetrize_topk: bool = True,
) -> PreparedGraph:
    data.validate()
    if len(topk) != len(data.metapath_counts):
        raise ValueError(f"Expected {len(data.metapath_counts)} top-K values, got {len(topk)}")
    offsets = _offsets(data.num_nodes)
    total_nodes = int(sum(data.num_nodes))

    relation_edges: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for relation in data.relations:
        edge, weight = _relation_edges(relation.matrix, offsets[relation.src_type], offsets[relation.dst_type], total_nodes)
        relation_edges.append((relation.name, edge, weight))

    filtered = [pathsim_topk(matrix, k, symmetrize_topk) for matrix, k in zip(data.metapath_counts, topk)]
    semantic_edges = [_semantic_edges(matrix) for matrix in filtered]
    target_offset = offsets[data.target_type]
    norm_edges = [_target_global_edges(matrix, target_offset) for matrix in filtered]

    relevance = filtered[0].copy()
    for matrix in filtered[1:]:
        relevance = relevance + matrix
    relevance = relevance.tocsr()
    relevance.data = (relevance.data > positive_threshold).astype(np.float32)
    relevance.eliminate_zeros()
    relevance = relevance.maximum(sp.eye(relevance.shape[0], dtype=np.float32, format="csr"))
    relevance.data[:] = 1.0
    relevance.sort_indices()
    positives = PositiveIndex(relevance.indptr.copy(), relevance.indices.copy(), relevance.shape[0])

    return PreparedGraph(relation_edges, semantic_edges, norm_edges, positives, filtered, offsets, total_nodes)
