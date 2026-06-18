from __future__ import annotations

import numpy as np

from avion.evaluation.classification import evaluate_base_to_novel


def evaluate_base_to_novel_by_names(
    logits: np.ndarray,
    labels: np.ndarray,
    class_id_to_name: dict[int, str],
    base_classes: set[str],
    novel_classes: set[str],
) -> dict[str, float]:
    base_ids = {class_id for class_id, name in class_id_to_name.items() if name in base_classes}
    novel_ids = {class_id for class_id, name in class_id_to_name.items() if name in novel_classes}
    return evaluate_base_to_novel(logits, labels, base_ids, novel_ids)

