from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.classification import evaluate_classification
from evaluation.clustering import evaluate_clustering
from models.cbcl import CBCL
from preprocessing.metapath import prepare_graph
from utils.data import load_dataset
from utils.logger import create_logger, save_json
from utils.seed import resolve_device, set_seed


def train_from_config(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    cpu_threads = int(config.get("cpu_threads", 4))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
        except RuntimeError:
            pass
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config.get("output_dir", "runs/cbcl"))
    logger = create_logger(output_dir)

    evaluation_cfg = config.get("evaluation", {})
    ratios = evaluation_cfg.get("split_ratios", [20, 40, 60])
    data = load_dataset(config["dataset"], config["data_dir"], ratios)
    preprocess_cfg = config["preprocess"]
    graph = prepare_graph(
        data,
        topk=[int(x) for x in preprocess_cfg["topk"]],
        positive_threshold=int(preprocess_cfg.get("positive_threshold", 0)),
        symmetrize_topk=bool(preprocess_cfg.get("symmetrize_topk", True)),
    ).to(device)
    features = [feature.to(device) if feature is not None else None for feature in data.features]

    model_cfg = config["model"]
    input_dims = [None if feature is None else int(feature.shape[1]) for feature in data.features]
    model = CBCL(
        input_dims=input_dims,
        num_nodes=data.num_nodes,
        n_relations=len(graph.relation_edges),
        n_metapaths=len(graph.metapath_edges),
        hidden_dim=int(model_cfg["hidden_dim"]),
        projection_dim=int(model_cfg["projection_dim"]),
        structural_layers=int(model_cfg["structural_layers"]),
        gat_heads=int(model_cfg["gat_heads"]),
        transformer_heads=int(model_cfg["transformer_heads"]),
        dropout=float(model_cfg["dropout"]),
        semantic_beta=float(model_cfg.get("semantic_beta", 0.5)),
    ).to(device)

    train_cfg = config["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("patience", epochs))
    log_every = int(train_cfg.get("log_every", 10))
    target_n = data.num_nodes[data.target_type]

    logger.info("dataset=%s data_dir=%s device=%s", config["dataset"], config["data_dir"], device)
    logger.info("types=%s num_nodes=%s relations=%d metapaths=%s", data.type_names, data.num_nodes, len(graph.relation_edges), data.metapath_names)
    logger.info("topk=%s target_nodes=%d", preprocess_cfg["topk"], target_n)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features, graph, data.target_type, target_n)
        loss, stats = model.loss(
            output,
            graph.positive_index,
            temperature=float(train_cfg["temperature"]),
            direction_weight=float(train_cfg["direction_weight"]),
            chunk_size=int(train_cfg.get("contrastive_chunk_size", 512)),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
        optimizer.step()
        history.append({"epoch": float(epoch), **stats})

        current = float(loss.detach().cpu())
        if current < best_loss - 1e-7:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % log_every == 0:
            logger.info("epoch=%04d loss=%.6f c->s=%.6f s->c=%.6f", epoch, stats["loss"], stats["c_to_s"], stats["s_to_c"])
        if stale >= patience:
            logger.info("early stopping at epoch=%d best_loss=%.6f", epoch, best_loss)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    embeddings = model.embed(features, graph, data.target_type, target_n).cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "input_dims": input_dims,
        "num_nodes": data.num_nodes,
        "n_relations": len(graph.relation_edges),
        "n_metapaths": len(graph.metapath_edges),
    }, output_dir / "best_model.pt")
    np.save(output_dir / "embeddings.npy", embeddings.numpy())
    save_json(output_dir / "history.json", history)

    metrics: dict[str, Any] = {}
    if bool(evaluation_cfg.get("enabled", True)) and data.labels is not None:
        metrics["clustering"] = evaluate_clustering(
            embeddings, data.labels, seed, int(evaluation_cfg.get("clustering_repeats", 10))
        )
        if data.splits:
            metrics["classification"] = evaluate_classification(embeddings, data.labels, data.splits, seed)

    summary = {
        "dataset": config["dataset"],
        "device": str(device),
        "best_loss": best_loss,
        "epochs_ran": len(history),
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "config_path": None if config_path is None else str(config_path),
    }
    save_json(output_dir / "metrics.json", summary)
    logger.info("finished; outputs saved to %s", output_dir)
    return summary
