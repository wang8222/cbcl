from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/ACM_toy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--papers", type=int, default=72)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_p, n_a, n_s, n_classes, dim = args.papers, 40, 9, 3, 16
    labels = np.arange(n_p) % n_classes
    rng.shuffle(labels)
    centers = rng.normal(size=(n_classes, dim)).astype(np.float32)
    p_feat = centers[labels] + 0.2 * rng.normal(size=(n_p, dim)).astype(np.float32)
    author_class = rng.integers(0, n_classes, n_a)
    a_feat = centers[author_class] + 0.3 * rng.normal(size=(n_a, dim)).astype(np.float32)

    pa_rows: list[int] = []
    pa_cols: list[int] = []
    ps_rows: list[int] = []
    ps_cols: list[int] = []
    nei_a = np.empty(n_p, dtype=object)
    nei_s = np.empty(n_p, dtype=object)
    for p in range(n_p):
        preferred_a = np.where(author_class == labels[p])[0]
        authors = rng.choice(preferred_a, size=min(3, len(preferred_a)), replace=False)
        subject = int(labels[p] + 3 * rng.integers(0, 3))
        nei_a[p] = authors.astype(np.int64)
        nei_s[p] = np.array([subject], dtype=np.int64)
        pa_rows.extend([p] * len(authors)); pa_cols.extend(authors.tolist())
        ps_rows.append(p); ps_cols.append(subject)
    pa = sp.csr_matrix((np.ones(len(pa_rows)), (pa_rows, pa_cols)), shape=(n_p, n_a))
    ps = sp.csr_matrix((np.ones(len(ps_rows)), (ps_rows, ps_cols)), shape=(n_p, n_s))
    pap = (pa @ pa.T).tocsr().astype(np.float32)
    psp = (ps @ ps.T).tocsr().astype(np.float32)

    sp.save_npz(out / "p_feat.npz", sp.csr_matrix(p_feat))
    sp.save_npz(out / "a_feat.npz", sp.csr_matrix(a_feat))
    np.save(out / "labels.npy", labels.astype(np.int64))
    np.save(out / "nei_a.npy", nei_a, allow_pickle=True)
    np.save(out / "nei_s.npy", nei_s, allow_pickle=True)
    np.savetxt(out / "pa.txt", np.column_stack([pa_rows, pa_cols]), fmt="%d")
    np.savetxt(out / "ps.txt", np.column_stack([ps_rows, ps_cols]), fmt="%d")
    sp.save_npz(out / "pap.npz", pap)
    sp.save_npz(out / "psp.npz", psp)
    sp.save_npz(out / "pos.npz", ((pap + psp) > 0).astype(np.float32))

    for ratio in (20, 40, 60):
        train: list[int] = []
        val: list[int] = []
        test: list[int] = []
        for c in range(n_classes):
            ids = np.where(labels == c)[0]
            rng.shuffle(ids)
            n_train = min(max(2, ratio // 20), max(2, len(ids) // 3))
            n_val = min(2, max(1, len(ids) - n_train - 1))
            train.extend(ids[:n_train]); val.extend(ids[n_train:n_train+n_val]); test.extend(ids[n_train+n_val:])
        np.save(out / f"train_{ratio}.npy", np.array(train, dtype=np.int64))
        np.save(out / f"val_{ratio}.npy", np.array(val, dtype=np.int64))
        np.save(out / f"test_{ratio}.npy", np.array(test, dtype=np.int64))
    print(f"Toy ACM data written to {out.resolve()}")


if __name__ == "__main__":
    main()
