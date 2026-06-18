from __future__ import annotations

from typing import Any

import numpy as np

from avion.prototypes.selective_aggregation import AggregationResult


def candidate_score_rows(
    dataset: str,
    protocol: str,
    seed: int,
    shots: int | None,
    class_name: str,
    captions: list[str],
    rs_flags: list[int],
    result: AggregationResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, caption in enumerate(captions):
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "seed": seed,
                "shots": shots,
                "class_name": class_name,
                "candidate_index": idx,
                "caption": caption,
                "rs_flag": int(rs_flags[idx]),
                "teacher_similarity": float(result.scores[idx]),
                "median": float(result.median),
                "mad": float(result.mad),
                "mad_z": float(result.z_scores[idx]),
                "kept": bool(result.kept[idx]),
                "aggregation_weight": float(result.weights[idx]),
            }
        )
    return rows


def kept_ratio(kept: np.ndarray) -> float:
    kept = np.asarray(kept, dtype=bool)
    return float(kept.mean()) if kept.size else 0.0

