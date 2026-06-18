from __future__ import annotations

import numpy as np

from avion.evaluation.metrics import harmonic_mean, top1_accuracy


def evaluate_classification(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {"accuracy": top1_accuracy(logits, labels)}


def evaluate_base_to_novel(
    logits: np.ndarray,
    labels: np.ndarray,
    base_class_indices: set[int],
    novel_class_indices: set[int],
) -> dict[str, float]:
    labels = np.asarray(labels)
    base_mask = np.array([int(label) in base_class_indices for label in labels])
    novel_mask = np.array([int(label) in novel_class_indices for label in labels])
    base_acc = top1_accuracy(logits[base_mask], labels[base_mask]) if base_mask.any() else 0.0
    novel_acc = top1_accuracy(logits[novel_mask], labels[novel_mask]) if novel_mask.any() else 0.0
    return {"Base": base_acc, "Novel": novel_acc, "HM": harmonic_mean(base_acc, novel_acc)}

