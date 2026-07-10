from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_clustering(embeddings: torch.Tensor, labels: torch.Tensor, seed: int, repeats: int = 10) -> dict[str, float]:
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    n_clusters = int(np.unique(y).size)
    nmi_values: list[float] = []
    ari_values: list[float] = []
    for run in range(repeats):
        prediction = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed + run).fit_predict(x)
        nmi_values.append(float(normalized_mutual_info_score(y, prediction, average_method="arithmetic")))
        ari_values.append(float(adjusted_rand_score(y, prediction)))
    return {
        "nmi_mean": float(np.mean(nmi_values)),
        "nmi_std": float(np.std(nmi_values)),
        "ari_mean": float(np.mean(ari_values)),
        "ari_std": float(np.std(ari_values)),
    }
