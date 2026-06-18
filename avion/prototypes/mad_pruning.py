from __future__ import annotations

import numpy as np


def robust_z_scores(scores: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    median = float(np.median(scores))
    deviations = np.abs(scores - median)
    mad = float(np.median(deviations))
    z_scores = deviations / (mad + eps)
    return z_scores, median, mad


def mad_keep_mask(scores: np.ndarray, threshold: float = 3.0, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, float, float]:
    z_scores, median, mad = robust_z_scores(scores, eps=eps)
    return z_scores <= threshold, z_scores, median, mad

