from __future__ import annotations

from typing import Any

from avion.trainers.avion_classification import ClassificationLossConfig, compute_avion_classification_loss


def compute_avion_base_to_novel_loss(
    *,
    active_base_class_indices: list[int],
    config: ClassificationLossConfig = ClassificationLossConfig(),
    **kwargs: Any,
) -> dict[str, Any]:
    return compute_avion_classification_loss(
        config=config,
        active_class_indices=active_base_class_indices,
        **kwargs,
    )

