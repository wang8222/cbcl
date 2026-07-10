from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def pathsim_topk(counts: sp.spmatrix, k: int, symmetrize: bool = True) -> sp.csr_matrix:
    """Compute PathSim and retain top-K neighbours per row without densifying.

    PathSim(i,j) = 2 * M[i,j] / (M[i,i] + M[j,j]) for a symmetric meta-path.
    The returned matrix is binary and has no self loops.
    """

    counts = counts.tocsr().astype(np.float32)
    if counts.shape[0] != counts.shape[1]:
        raise ValueError("PathSim requires a square path-count matrix")
    if k <= 0:
        return sp.csr_matrix(counts.shape, dtype=np.float32)

    diagonal = counts.diagonal().astype(np.float32)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(counts.shape[0]):
        start, end = counts.indptr[i], counts.indptr[i + 1]
        js = counts.indices[start:end]
        values = counts.data[start:end]
        keep = js != i
        js, values = js[keep], values[keep]
        if js.size == 0:
            continue
        denominator = diagonal[i] + diagonal[js]
        scores = np.divide(2.0 * values, denominator, out=np.zeros_like(values), where=denominator > 0)
        positive = scores > 0
        js, scores = js[positive], scores[positive]
        if js.size == 0:
            continue
        take = min(k, js.size)
        if take < js.size:
            selected = np.argpartition(scores, -take)[-take:]
            selected = selected[np.argsort(scores[selected])[::-1]]
        else:
            selected = np.argsort(scores)[::-1]
        rows.extend([i] * take)
        cols.extend(js[selected].tolist())

    result = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=counts.shape)
    if symmetrize:
        result = result.maximum(result.T).tocsr()
    result.setdiag(0)
    result.eliminate_zeros()
    result.sort_indices()
    return result
