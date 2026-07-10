from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import resolve_file  # noqa: E402


def describe(path: Path) -> str:
    size = path.stat().st_size
    try:
        matrix = sp.load_npz(path)
        return f"scipy.sparse shape={matrix.shape} dtype={matrix.dtype} nnz={matrix.nnz} size={size}"
    except Exception:
        pass
    try:
        value = np.load(path, allow_pickle=True)
        if isinstance(value, np.lib.npyio.NpzFile):
            info = ", ".join(f"{key}:{value[key].shape}/{value[key].dtype}" for key in value.files)
            value.close()
            return f"numpy.npz keys=[{info}] size={size}"
        return f"numpy.npy shape={value.shape} dtype={value.dtype} size={size}"
    except Exception:
        pass
    try:
        array = np.loadtxt(path, max_rows=5)
        return f"text first_rows_shape={np.atleast_2d(array).shape} size={size}"
    except Exception as exc:
        return f"unknown/binary size={size} ({type(exc).__name__})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ACM")
    args = parser.parse_args()
    directory = Path(args.data_dir)
    if not directory.exists():
        raise FileNotFoundError(directory)
    stems = ["p_feat", "a_feat", "labels", "nei_a", "nei_s", "pa", "ps", "pap", "psp", "pos", "train_20", "val_20", "test_20", "train_40", "val_40", "test_40", "train_60", "val_60", "test_60", "graph.bin"]
    print(f"Inspecting: {directory.resolve()}")
    for stem in stems:
        path = resolve_file(directory, stem, required=False)
        if path is None:
            print(f"[MISSING] {stem}")
        else:
            print(f"[OK] {stem:10s} -> {path.name:20s} | {describe(path)}")


if __name__ == "__main__":
    main()
