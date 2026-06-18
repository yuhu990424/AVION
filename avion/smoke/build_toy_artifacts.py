from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from avion.datasets.base import ClassificationMetadata
from avion.datasets.build_metadata import build_imagefolder_metadata
from avion.datasets.split_tools.build_fewshot_splits import build_fewshot_split
from avion.prototypes.build_class_prototypes import build_class_prototypes
from avion.utils.io import ensure_dir, write_json, write_jsonl
from avion.utils.tensor import require_torch


def _caption_rows(dataset: str, classes: list[str]) -> list[dict[str, object]]:
    templates = {
        "airport": [
            "An aerial view of an airport with runways and taxiways.",
            "Satellite imagery of airport terminals, aprons, and paved runways.",
        ],
        "forest": [
            "Satellite imagery showing dense forest canopy and natural texture.",
            "An overhead view of contiguous woodland with green vegetation.",
        ],
    }
    rows = []
    for class_name in classes:
        for idx, caption in enumerate(templates[class_name]):
            rows.append(
                {
                    "dataset": dataset,
                    "class_name": class_name,
                    "candidate_index": idx,
                    "caption": caption,
                    "viewpoint": "aerial" if idx == 0 else "satellite",
                    "visual_cues": [],
                    "spatial_cues": [],
                    "llm_model": "toy",
                    "prompt_version": "toy",
                    "kp": 2,
                    "raw_response_sha256": "toy",
                    "rs_flag": 1,
                }
            )
    return rows


def build_toy_artifacts(root: str | Path) -> dict[str, str]:
    torch = require_torch()
    root = ensure_dir(root)
    image_root = ensure_dir(root / "processed" / "toy" / "images")
    classes = ["airport", "forest"]
    colors = {"airport": (180, 180, 180), "forest": (20, 120, 40)}
    for class_name in classes:
        class_dir = ensure_dir(image_root / class_name)
        for idx in range(3):
            Image.new("RGB", (16, 16), colors[class_name]).save(class_dir / f"{idx}.jpg")

    metadata = build_imagefolder_metadata("toy", image_root)
    metadata_path = root / "processed" / "toy" / "metadata.json"
    write_json(metadata, metadata_path)
    split = build_fewshot_split(ClassificationMetadata.from_json(metadata_path), shots=1, seed=1)
    split_path = root / "splits" / "toy" / "seed1" / "fewshot_1.json"
    write_json(split, split_path)

    annotations = _caption_rows("toy", classes)
    annotation_path = root / "annotations" / "gemini25_flash" / "toy_candidates_kp2_v1.jsonl"
    write_jsonl(annotations, annotation_path)

    dim = 4
    class_vectors = {
        "airport": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "forest": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    image_features = {}
    for row in split["train"]:
        image_features[row["sample_id"]] = class_vectors[row["class_name"]]
    text_features = {}
    for row in annotations:
        key = f"{row['dataset']}::{row['class_name']}::{row['candidate_index']}"
        noise = torch.tensor([0.0, 0.0, 0.01 * int(row["candidate_index"]), 0.0])
        text_features[key] = class_vectors[row["class_name"]] + noise
    image_cache_path = root / "cache" / "teacher_images" / "toy.pt"
    text_cache_path = root / "cache" / "teacher_texts" / "toy.pt"
    ensure_dir(image_cache_path.parent)
    ensure_dir(text_cache_path.parent)
    torch.save({"features": image_features}, image_cache_path)
    torch.save({"features": text_features, "rows": annotations}, text_cache_path)

    proto_dir = ensure_dir(root / "cache" / "prototypes" / "toy" / "fewshot" / "seed1" / "shots_1")
    visual, text, diagnostics = build_class_prototypes(
        split_or_metadata=split,
        annotations=annotations,
        image_cache={"features": image_features},
        text_cache={"features": text_features},
        dataset="toy",
        protocol="few_shot",
        seed=1,
        shots=1,
        beta=10,
        gamma=2,
    )
    torch.save({"prototypes": visual}, proto_dir / "visual_prototypes.pt")
    torch.save({"prototypes": text}, proto_dir / "text_prototypes_selective.pt")
    write_jsonl(diagnostics, proto_dir / "candidate_scores.jsonl")
    return {
        "metadata": str(metadata_path),
        "split": str(split_path),
        "annotations": str(annotation_path),
        "image_cache": str(image_cache_path),
        "text_cache": str(text_cache_path),
        "prototype_dir": str(proto_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/tmp/avion_toy")
    args = parser.parse_args()
    write_json(build_toy_artifacts(args.root), Path(args.root) / "artifact_paths.json")
    print(f"Wrote toy artifacts under {args.root}")


if __name__ == "__main__":
    main()
