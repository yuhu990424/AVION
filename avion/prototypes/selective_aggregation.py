from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avion.prototypes.mad_pruning import mad_keep_mask
from avion.prototypes.visual_prototype import l2_normalize


@dataclass(frozen=True)
class AggregationResult:
    prototype: np.ndarray
    scores: np.ndarray
    z_scores: np.ndarray
    kept: np.ndarray
    weights: np.ndarray
    median: float
    mad: float


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def aggregate_text_prototype(
    visual_prototype: np.ndarray,
    text_features: np.ndarray,
    rs_flags: np.ndarray,
    beta: float = 10.0,
    gamma: float = 2.0,
    mad_threshold: float = 3.0,
    eps: float = 1e-8,
) -> AggregationResult:
    visual = l2_normalize(np.asarray(visual_prototype, dtype=np.float64), axis=-1)
    texts = l2_normalize(np.asarray(text_features, dtype=np.float64), axis=-1)
    flags = np.asarray(rs_flags, dtype=np.float64)
    if texts.ndim != 2:
        raise ValueError("text_features must have shape [num_candidates, dim].")
    if flags.shape[0] != texts.shape[0]:
        raise ValueError("rs_flags length must match text_features.")

    scores = texts @ visual
    kept, z_scores, median, mad = mad_keep_mask(scores, threshold=mad_threshold, eps=eps)
    if not np.any(kept):
        kept = np.ones_like(kept, dtype=bool)
    logits = beta * scores[kept] + gamma * flags[kept]
    kept_weights = softmax(logits)
    weights = np.zeros_like(scores, dtype=np.float64)
    weights[kept] = kept_weights
    prototype = l2_normalize((texts * weights[:, None]).sum(axis=0), axis=-1)
    return AggregationResult(
        prototype=prototype,
        scores=scores,
        z_scores=z_scores,
        kept=kept,
        weights=weights,
        median=median,
        mad=mad,
    )

