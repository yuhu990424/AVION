from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from avion.prototypes.diagnostics import candidate_score_rows
from avion.prototypes.selective_aggregation import aggregate_text_prototype
from avion.prototypes.visual_prototype import mean_visual_prototype
from avion.utils.io import ensure_dir, read_json, read_jsonl, write_jsonl
from avion.utils.tensor import require_torch


def _split_train_rows(split: dict[str, Any]) -> list[dict[str, Any]]:
    if "train" in split:
        return list(split["train"])
    if "train_base" in split:
        return list(split["train_base"])
    if "samples" in split:
        return list(split["samples"])
    raise ValueError("Split/metadata must contain train, train_base, or samples.")


def _group_by_class(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["class_name"], []).append(row)
    return grouped


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)


def build_class_prototypes(
    split_or_metadata: dict[str, Any],
    annotations: list[dict[str, Any]],
    image_cache: dict[str, Any],
    text_cache: dict[str, Any],
    dataset: str,
    protocol: str,
    seed: int,
    shots: int | None,
    beta: float = 10.0,
    gamma: float = 2.0,
    mad_threshold: float = 3.0,
    eps: float = 1e-8,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    torch = require_torch()
    image_features = image_cache["features"]
    text_features = text_cache["features"]
    train_rows = _split_train_rows(split_or_metadata)
    by_class = _group_by_class(train_rows)
    ann_by_class: dict[str, list[dict[str, Any]]] = {}
    for row in annotations:
        ann_by_class.setdefault(row["class_name"], []).append(row)

    visual_prototypes: dict[str, Any] = {}
    text_prototypes: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []

    for class_name, rows in sorted(by_class.items()):
        class_image_features = []
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id not in image_features:
                raise KeyError(f"Missing teacher image feature for sample_id={sample_id}")
            class_image_features.append(_tensor_to_numpy(image_features[sample_id]))
        visual = mean_visual_prototype(np.stack(class_image_features, axis=0))
        visual_prototypes[class_name] = torch.as_tensor(visual, dtype=torch.float32)

        candidates = sorted(ann_by_class.get(class_name, []), key=lambda item: int(item["candidate_index"]))
        if not candidates:
            raise ValueError(f"No annotation candidates found for class={class_name}")
        candidate_text_features = []
        rs_flags = []
        captions = []
        for candidate in candidates:
            key = f"{candidate['dataset']}::{candidate['class_name']}::{candidate['candidate_index']}"
            if key not in text_features:
                raise KeyError(f"Missing teacher text feature for key={key}")
            candidate_text_features.append(_tensor_to_numpy(text_features[key]))
            rs_flags.append(int(candidate.get("rs_flag", 0)))
            captions.append(str(candidate["caption"]))

        result = aggregate_text_prototype(
            visual_prototype=visual,
            text_features=np.stack(candidate_text_features, axis=0),
            rs_flags=np.asarray(rs_flags),
            beta=beta,
            gamma=gamma,
            mad_threshold=mad_threshold,
            eps=eps,
        )
        text_prototypes[class_name] = torch.as_tensor(result.prototype, dtype=torch.float32)
        diagnostics.extend(
            candidate_score_rows(
                dataset=dataset,
                protocol=protocol,
                seed=seed,
                shots=shots,
                class_name=class_name,
                captions=captions,
                rs_flags=rs_flags,
                result=result,
            )
        )

    return visual_prototypes, text_prototypes, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-or-metadata", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-cache", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--mad-threshold", type=float, default=3.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    torch = require_torch()
    split = read_json(args.split_or_metadata)
    annotations = list(read_jsonl(args.annotations))
    image_cache = torch.load(args.image_cache, map_location="cpu")
    text_cache = torch.load(args.text_cache, map_location="cpu")
    visual, text, diagnostics = build_class_prototypes(
        split_or_metadata=split,
        annotations=annotations,
        image_cache=image_cache,
        text_cache=text_cache,
        dataset=args.dataset,
        protocol=args.protocol,
        seed=args.seed,
        shots=args.shots,
        beta=args.beta,
        gamma=args.gamma,
        mad_threshold=args.mad_threshold,
        eps=args.eps,
    )
    out_dir = ensure_dir(args.out_dir)
    torch.save({"prototypes": visual}, Path(out_dir) / "visual_prototypes.pt")
    torch.save({"prototypes": text}, Path(out_dir) / "text_prototypes_selective.pt")
    write_jsonl(diagnostics, Path(out_dir) / "candidate_scores.jsonl")
    try:
        import pandas as pd

        pd.DataFrame(diagnostics).to_parquet(Path(out_dir) / "candidate_scores.parquet")
    except Exception:
        pass


if __name__ == "__main__":
    main()

