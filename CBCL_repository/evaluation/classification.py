from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


def evaluate_classification(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    splits: dict[str, dict[str, torch.Tensor]],
    seed: int,
) -> dict[str, dict[str, float]]:
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    results: dict[str, dict[str, float]] = {}
    for split_name, split in splits.items():
        train_idx = split["train"].cpu().numpy()
        test_idx = split["test"].cpu().numpy()
        classifier = LogisticRegression(max_iter=3000, random_state=seed, solver="lbfgs")
        classifier.fit(x[train_idx], y[train_idx])
        prediction = classifier.predict(x[test_idx])
        probability = classifier.predict_proba(x[test_idx])
        metrics = {
            "macro_f1": float(f1_score(y[test_idx], prediction, average="macro")),
            "micro_f1": float(f1_score(y[test_idx], prediction, average="micro")),
        }
        try:
            if probability.shape[1] == 2:
                metrics["auc"] = float(roc_auc_score(y[test_idx], probability[:, 1]))
            else:
                metrics["auc"] = float(roc_auc_score(y[test_idx], probability, multi_class="ovr", labels=classifier.classes_))
        except ValueError:
            metrics["auc"] = float("nan")
        results[split_name] = metrics
    return results
