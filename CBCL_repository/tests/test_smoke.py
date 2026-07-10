from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from models.cbcl import CBCL
from preprocessing.metapath import prepare_graph
from preprocessing.pathsim import pathsim_topk
from utils.data import load_acm


def _make_toy(tmp_path: Path) -> Path:
    out = tmp_path / "ACM"
    subprocess.run([sys.executable, "scripts/make_toy_acm.py", "--output", str(out), "--papers", "36"], check=True)
    return out


def test_acm_loader_and_pathsim(tmp_path: Path) -> None:
    directory = _make_toy(tmp_path)
    data = load_acm(directory)
    assert data.num_nodes[0] == 36
    assert data.metapath_names == ["PAP", "PSP"]
    filtered = pathsim_topk(data.metapath_counts[0], 3, symmetrize=False)
    assert filtered.getnnz(axis=1).max() <= 3
    assert filtered.diagonal().sum() == 0


def test_forward_backward(tmp_path: Path) -> None:
    directory = _make_toy(tmp_path)
    data = load_acm(directory)
    graph = prepare_graph(data, [3, 3], positive_threshold=0).to(torch.device("cpu"))
    model = CBCL(
        [None if x is None else x.shape[1] for x in data.features],
        data.num_nodes,
        len(graph.relation_edges),
        len(graph.metapath_edges),
        hidden_dim=16,
        projection_dim=16,
        structural_layers=1,
        gat_heads=2,
        transformer_heads=2,
        dropout=0.1,
    )
    output = model(data.features, graph, data.target_type, data.num_nodes[0])
    assert output.semantic.shape == (36, 32)
    loss, _ = model.loss(output, graph.positive_index, 0.4, 0.5, 16)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
