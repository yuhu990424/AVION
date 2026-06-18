from __future__ import annotations

import numpy as np


def top1_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_samples, num_classes].")
    pred = logits.argmax(axis=1)
    return float((pred == labels).mean() * 100.0)


def harmonic_mean(base: float, novel: float, eps: float = 1e-12) -> float:
    if base <= 0 or novel <= 0:
        return 0.0
    return float(2.0 * base * novel / max(base + novel, eps))


def recall_at_k(similarity: np.ndarray, relevance: np.ndarray, k: int) -> float:
    similarity = np.asarray(similarity)
    relevance = np.asarray(relevance).astype(bool)
    if similarity.shape != relevance.shape:
        raise ValueError("similarity and relevance must have the same shape.")
    if similarity.ndim != 2:
        raise ValueError("similarity must be a matrix.")
    k = min(k, similarity.shape[1])
    order = np.argsort(-similarity, axis=1)[:, :k]
    hits = []
    for row_idx, cols in enumerate(order):
        hits.append(bool(relevance[row_idx, cols].any()))
    return float(np.mean(hits) * 100.0)


def retrieval_metrics(similarity_i2t: np.ndarray, relevance_i2t: np.ndarray) -> dict[str, float]:
    i2t = {
        f"I2T_R{k}": recall_at_k(similarity_i2t, relevance_i2t, k)
        for k in (1, 5, 10)
    }
    t2i = {
        f"T2I_R{k}": recall_at_k(similarity_i2t.T, relevance_i2t.T, k)
        for k in (1, 5, 10)
    }
    all_values = list(i2t.values()) + list(t2i.values())
    return {**i2t, **t2i, "mR": float(np.mean(all_values))}

