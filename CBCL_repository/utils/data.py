from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class Relation:
    name: str
    src_type: int
    dst_type: int
    matrix: sp.csr_matrix


@dataclass
class HINData:
    type_names: list[str]
    num_nodes: list[int]
    features: list[torch.Tensor | None]
    relations: list[Relation]
    metapath_counts: list[sp.csr_matrix]
    metapath_names: list[str]
    target_type: int
    labels: torch.Tensor | None = None
    splits: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def validate(self) -> None:
        n_types = len(self.type_names)
        if not (len(self.num_nodes) == len(self.features) == n_types):
            raise ValueError("type_names, num_nodes and features must have equal length")
        for i, feature in enumerate(self.features):
            if feature is not None and feature.shape[0] != self.num_nodes[i]:
                raise ValueError(f"Feature row count mismatch for type {self.type_names[i]}")
        target_n = self.num_nodes[self.target_type]
        if len(self.metapath_counts) != len(self.metapath_names):
            raise ValueError("metapath_names and metapath_counts must have equal length")
        for name, matrix in zip(self.metapath_names, self.metapath_counts):
            if matrix.shape != (target_n, target_n):
                raise ValueError(f"{name} shape={matrix.shape}, expected {(target_n, target_n)}")
        for relation in self.relations:
            expected = (self.num_nodes[relation.src_type], self.num_nodes[relation.dst_type])
            if relation.matrix.shape != expected:
                raise ValueError(f"Relation {relation.name} shape={relation.matrix.shape}, expected={expected}")
        if self.labels is not None and self.labels.numel() != target_n:
            raise ValueError("labels must have one value per target node")


_SUFFIXES = ("", ".npy", ".npz", ".txt", ".csv")


def resolve_file(directory: Path, stem: str | Sequence[str], required: bool = True) -> Path | None:
    stems = [stem] if isinstance(stem, str) else list(stem)
    lower_map = {p.name.lower(): p for p in directory.iterdir()} if directory.exists() else {}
    for item in stems:
        item_path = directory / item
        if item_path.exists():
            return item_path
        base = Path(item).stem if Path(item).suffix else item
        for suffix in _SUFFIXES:
            candidate_name = (base + suffix).lower()
            if candidate_name in lower_map:
                return lower_map[candidate_name]
    if required:
        tried = ", ".join(str(directory / s) for s in stems)
        raise FileNotFoundError(f"Required data file not found. Tried: {tried} (with common suffixes)")
    return None


def load_array(path: Path, allow_pickle: bool = False) -> np.ndarray:
    loaded = np.load(path, allow_pickle=allow_pickle)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            keys = loaded.files
            if len(keys) == 1:
                return np.asarray(loaded[keys[0]])
            for key in ("arr_0", "labels", "indices", "data"):
                if key in keys:
                    return np.asarray(loaded[key])
            raise ValueError(f"Cannot choose an array from {path}; keys={keys}")
        finally:
            loaded.close()
    return np.asarray(loaded)


def load_sparse(path: Path) -> sp.csr_matrix:
    try:
        return sp.load_npz(path).tocsr().astype(np.float32)
    except Exception:
        array = load_array(path, allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D sparse/dense matrix in {path}, got {array.shape}")
        return sp.csr_matrix(array.astype(np.float32))


def load_feature(path: Path) -> torch.Tensor:
    try:
        matrix = sp.load_npz(path).tocsr().astype(np.float32)
        rowsum = np.asarray(matrix.sum(axis=1)).ravel()
        inv = np.zeros_like(rowsum, dtype=np.float32)
        inv[rowsum != 0] = 1.0 / rowsum[rowsum != 0]
        matrix = sp.diags(inv) @ matrix
        return torch.from_numpy(matrix.toarray().astype(np.float32))
    except Exception:
        array = load_array(path, allow_pickle=False).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"Feature file must be 2-D: {path}, shape={array.shape}")
        rowsum = array.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        return torch.from_numpy(array / rowsum)


def _normalise_labels(array: np.ndarray) -> torch.Tensor:
    labels = np.asarray(array).reshape(-1).astype(np.int64)
    _, encoded = np.unique(labels, return_inverse=True)
    return torch.from_numpy(encoded.astype(np.int64))


def _load_splits(directory: Path, ratios: Iterable[int]) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for ratio in ratios:
        paths = {part: resolve_file(directory, f"{part}_{ratio}", required=False) for part in ("train", "val", "test")}
        if all(paths.values()):
            result[str(ratio)] = {
                part: torch.from_numpy(load_array(path, allow_pickle=False).reshape(-1).astype(np.int64))
                for part, path in paths.items() if path is not None
            }
    return result


def _binary(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def _bidirectional(name: str, src_type: int, dst_type: int, matrix: sp.csr_matrix) -> list[Relation]:
    matrix = _binary(matrix)
    return [
        Relation(name, src_type, dst_type, matrix),
        Relation(f"{name}^-1", dst_type, src_type, matrix.T.tocsr()),
    ]


def _adjust_ids(values: np.ndarray, n: int, possible_offsets: Sequence[int]) -> np.ndarray:
    values = values.astype(np.int64)
    if values.size == 0:
        return values
    for offset in possible_offsets:
        shifted = values - offset
        if shifted.min(initial=0) >= 0 and shifted.max(initial=-1) < n:
            return shifted
    if values.min(initial=0) >= 1 and values.max(initial=-1) <= n:
        return values - 1
    if values.min(initial=0) >= 0 and values.max(initial=-1) < n:
        return values
    raise ValueError(f"IDs out of range for n={n}: min={values.min()}, max={values.max()}")


def _load_edge_text(path: Path, n_src: int, n_dst: int, dst_offsets: Sequence[int]) -> sp.csr_matrix:
    try:
        raw = np.loadtxt(path, dtype=np.int64, delimiter=None)
    except ValueError:
        raw = np.loadtxt(path, dtype=np.int64, delimiter=",")
    raw = np.atleast_2d(raw)
    if raw.shape[1] < 2:
        raise ValueError(f"Edge file needs at least two columns: {path}")
    a, b = raw[:, 0], raw[:, 1]

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for src, dst in ((a, b), (b, a)):
        try:
            src2 = _adjust_ids(src, n_src, [0])
            dst2 = _adjust_ids(dst, n_dst, list(dst_offsets) + [0])
            candidates.append((src2, dst2))
        except ValueError:
            pass
    if not candidates:
        raise ValueError(f"Cannot infer edge orientation or ID offsets in {path}")
    src, dst = candidates[0]
    return _binary(sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n_src, n_dst)))


def _load_neighbor_lists(path: Path, n_src: int, n_dst: int, dst_offsets: Sequence[int]) -> sp.csr_matrix:
    raw = load_array(path, allow_pickle=True)
    if raw.ndim == 2 and raw.dtype != object and raw.shape == (n_src, n_dst):
        return _binary(sp.csr_matrix(raw))
    if raw.ndim == 2 and raw.dtype != object and raw.shape[1] == 2:
        src = _adjust_ids(raw[:, 0], n_src, [0])
        dst = _adjust_ids(raw[:, 1], n_dst, list(dst_offsets) + [0])
        return _binary(sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n_src, n_dst)))
    if len(raw) != n_src:
        raise ValueError(f"Neighbor list length in {path} is {len(raw)}, expected {n_src}")
    rows: list[int] = []
    cols: list[int] = []
    for i, neighbours in enumerate(raw):
        arr = np.asarray(neighbours).reshape(-1)
        if arr.size == 0:
            continue
        arr = _adjust_ids(arr, n_dst, list(dst_offsets) + [0])
        rows.extend([i] * len(arr))
        cols.extend(arr.tolist())
    return _binary(sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_src, n_dst)))


def _load_relation(
    directory: Path,
    text_stem: str,
    neighbour_stem: str,
    n_src: int,
    n_dst: int,
    dst_offsets: Sequence[int],
) -> sp.csr_matrix:
    text = resolve_file(directory, text_stem, required=False)
    if text is not None and text.suffix.lower() in {"", ".txt", ".csv"}:
        try:
            return _load_edge_text(text, n_src, n_dst, dst_offsets)
        except Exception as exc:
            print(f"[CBCL] Warning: failed to parse {text}: {exc}; trying {neighbour_stem}")
    neighbours = resolve_file(directory, neighbour_stem, required=True)
    assert neighbours is not None
    return _load_neighbor_lists(neighbours, n_src, n_dst, dst_offsets)


def load_acm(directory: str | Path, ratios: Iterable[int] = (20, 40, 60)) -> HINData:
    directory = Path(directory)
    p_path = resolve_file(directory, "p_feat")
    a_path = resolve_file(directory, "a_feat")
    assert p_path is not None and a_path is not None
    p_feat = load_feature(p_path)
    a_feat = load_feature(a_path)
    n_p, n_a = p_feat.shape[0], a_feat.shape[0]

    pa = _load_relation(directory, "pa", "nei_a", n_p, n_a, [n_p])

    # Subject count can be inferred from PS edges. The benchmark has 60 subjects.
    ps_text = resolve_file(directory, "ps", required=False)
    nei_s = resolve_file(directory, "nei_s", required=False)
    n_s = 60
    ps: sp.csr_matrix | None = None
    if ps_text is not None and ps_text.suffix.lower() in {"", ".txt", ".csv"}:
        try:
            raw = np.loadtxt(ps_text, dtype=np.int64, delimiter=None)
            raw = np.atleast_2d(raw)
            for candidate in (raw[:, 1], raw[:, 0]):
                for offset in (n_p + n_a, n_p, 0, 1):
                    shifted = candidate - offset
                    if shifted.size and shifted.min() >= 0 and shifted.max() < 10000:
                        n_s = max(n_s, int(shifted.max()) + 1)
                        break
            ps = _load_edge_text(ps_text, n_p, n_s, [n_p + n_a, n_p])
        except Exception as exc:
            print(f"[CBCL] Warning: failed to parse {ps_text}: {exc}; trying nei_s")
    if ps is None:
        if nei_s is None:
            raise FileNotFoundError("Neither ps.txt nor nei_s.npy could be loaded")
        # First inspect object lists to infer subject count under local/global IDs.
        raw = load_array(nei_s, allow_pickle=True)
        flat = np.concatenate([np.asarray(x).reshape(-1) for x in raw if np.asarray(x).size]) if len(raw) else np.array([], dtype=np.int64)
        for offset in (n_p + n_a, n_p, 0, 1):
            shifted = flat - offset
            if shifted.size and shifted.min() >= 0 and shifted.max() < 10000:
                n_s = max(1, int(shifted.max()) + 1)
                break
        ps = _load_neighbor_lists(nei_s, n_p, n_s, [n_p + n_a, n_p])

    pap_path = resolve_file(directory, "pap")
    psp_path = resolve_file(directory, "psp")
    labels_path = resolve_file(directory, "labels")
    assert pap_path is not None and psp_path is not None and labels_path is not None
    pap = load_sparse(pap_path)
    psp = load_sparse(psp_path)
    labels = _normalise_labels(load_array(labels_path, allow_pickle=False))

    data = HINData(
        type_names=["paper", "author", "subject"],
        num_nodes=[n_p, n_a, n_s],
        features=[p_feat, a_feat, None],
        relations=_bidirectional("P-A", 0, 1, pa) + _bidirectional("P-S", 0, 2, ps),
        metapath_counts=[pap, psp],
        metapath_names=["PAP", "PSP"],
        target_type=0,
        labels=labels,
        splits=_load_splits(directory, ratios),
    )
    data.validate()
    return data


def _feature_or_none(directory: Path, names: Sequence[str], expected_rows: int | None = None) -> torch.Tensor | None:
    path = resolve_file(directory, names, required=False)
    if path is None:
        return None
    feature = load_feature(path)
    if expected_rows is not None and feature.shape[0] != expected_rows:
        raise ValueError(f"Feature {path} has {feature.shape[0]} rows; expected {expected_rows}")
    return feature


def load_dblp(directory: str | Path, ratios: Iterable[int] = (20, 40, 60)) -> HINData:
    directory = Path(directory)
    labels_path = resolve_file(directory, "labels")
    assert labels_path is not None
    labels = _normalise_labels(load_array(labels_path))
    n_a = len(labels)
    apa = load_sparse(resolve_file(directory, "apa"))  # type: ignore[arg-type]
    apcpa = load_sparse(resolve_file(directory, ["apcpa", "apvpa"]))  # type: ignore[arg-type]
    aptpa = load_sparse(resolve_file(directory, "aptpa"))  # type: ignore[arg-type]

    nei_p_path = resolve_file(directory, ["nei_p", "nei_ap"], required=True)
    raw_p = load_array(nei_p_path, allow_pickle=True)  # type: ignore[arg-type]
    flat_p = np.concatenate([np.asarray(x).reshape(-1) for x in raw_p if np.asarray(x).size])
    n_p = int(flat_p.max()) + 1 if flat_p.size else 14328
    ap = _load_neighbor_lists(nei_p_path, n_a, n_p, [n_a])  # type: ignore[arg-type]

    # Conference and term channels are loaded when neighbour lists exist.
    relations = _bidirectional("A-P", 0, 1, ap)
    num_nodes = [n_a, n_p]
    type_names = ["author", "paper"]
    features: list[torch.Tensor | None] = [
        _feature_or_none(directory, ["a_feat", "features_a"], n_a),
        _feature_or_none(directory, ["p_feat", "features_p"], n_p),
    ]
    for stem, name, default_n in (("nei_c", "conference", 20), ("nei_t", "term", 7723)):
        path = resolve_file(directory, stem, required=False)
        if path is not None:
            raw = load_array(path, allow_pickle=True)
            flat = np.concatenate([np.asarray(x).reshape(-1) for x in raw if np.asarray(x).size]) if len(raw) else np.array([])
            n_ctx = int(flat.max()) + 1 if flat.size else default_n
            matrix = _load_neighbor_lists(path, n_a, n_ctx, [n_a, n_a + n_p])
            idx = len(num_nodes)
            relations += _bidirectional(f"A-{name[0].upper()}-context", 0, idx, matrix)
            num_nodes.append(n_ctx)
            type_names.append(name)
            features.append(_feature_or_none(directory, [f"{name[0]}_feat", f"features_{name[0]}"], n_ctx))

    data = HINData(type_names, num_nodes, features, relations, [apa, apcpa, aptpa], ["APA", "APCPA", "APTPA"], 0, labels, _load_splits(directory, ratios))
    data.validate()
    return data


def load_aminer(directory: str | Path, ratios: Iterable[int] = (20, 40, 60)) -> HINData:
    directory = Path(directory)
    labels_path = resolve_file(directory, "labels")
    assert labels_path is not None
    labels = _normalise_labels(load_array(labels_path))
    n_p = len(labels)
    pap = load_sparse(resolve_file(directory, ["pap", "adj_pap"]))  # type: ignore[arg-type]
    prp = load_sparse(resolve_file(directory, ["prp", "adj_prp"]))  # type: ignore[arg-type]

    nei_a_path = resolve_file(directory, "nei_a")
    nei_r_path = resolve_file(directory, "nei_r")
    assert nei_a_path is not None and nei_r_path is not None
    raw_a = load_array(nei_a_path, allow_pickle=True)
    raw_r = load_array(nei_r_path, allow_pickle=True)
    flat_a = np.concatenate([np.asarray(x).reshape(-1) for x in raw_a if np.asarray(x).size])
    flat_r = np.concatenate([np.asarray(x).reshape(-1) for x in raw_r if np.asarray(x).size])
    n_a = int(flat_a.max()) + 1 if flat_a.size else 13329
    n_r = int(flat_r.max()) + 1 if flat_r.size else 35890
    pa = _load_neighbor_lists(nei_a_path, n_p, n_a, [n_p])
    pr = _load_neighbor_lists(nei_r_path, n_p, n_r, [n_p, n_p + n_a])

    p_feature = _feature_or_none(directory, ["p_feat", "feat_p_pap.w1000.l100", "features_p"], n_p)
    data = HINData(
        ["paper", "author", "reference"], [n_p, n_a, n_r],
        [p_feature, _feature_or_none(directory, ["a_feat", "feat_a.w1000.l100", "features_a"], n_a), _feature_or_none(directory, ["r_feat", "feat_r.w1000.l100", "features_r"], n_r)],
        _bidirectional("P-A", 0, 1, pa) + _bidirectional("P-R", 0, 2, pr),
        [pap, prp], ["PAP", "PRP"], 0, labels, _load_splits(directory, ratios),
    )
    data.validate()
    return data


def load_dataset(name: str, directory: str | Path, ratios: Iterable[int] = (20, 40, 60)) -> HINData:
    name = name.lower()
    if name == "acm":
        return load_acm(directory, ratios)
    if name == "dblp":
        return load_dblp(directory, ratios)
    if name == "aminer":
        return load_aminer(directory, ratios)
    raise ValueError(f"Unknown dataset: {name}")
