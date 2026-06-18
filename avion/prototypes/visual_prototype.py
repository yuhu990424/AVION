from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def mean_visual_prototype(features: np.ndarray) -> np.ndarray:
    features = l2_normalize(np.asarray(features, dtype=np.float64), axis=-1)
    return l2_normalize(features.mean(axis=0), axis=-1)

