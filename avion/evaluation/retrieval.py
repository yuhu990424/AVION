from __future__ import annotations

import numpy as np

from avion.evaluation.metrics import retrieval_metrics


def build_relevance_matrix(query_image_ids: list[str], gallery_caption_image_ids: list[str]) -> np.ndarray:
    relevance = np.zeros((len(query_image_ids), len(gallery_caption_image_ids)), dtype=bool)
    for i, image_id in enumerate(query_image_ids):
        for j, caption_image_id in enumerate(gallery_caption_image_ids):
            relevance[i, j] = image_id == caption_image_id
    return relevance


def evaluate_retrieval(similarity_i2t: np.ndarray, relevance_i2t: np.ndarray) -> dict[str, float]:
    return retrieval_metrics(similarity_i2t, relevance_i2t)

