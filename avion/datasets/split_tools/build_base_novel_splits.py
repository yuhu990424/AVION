from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from avion.datasets.base import ClassificationMetadata, ClassificationSample
from avion.utils.io import write_json


def _sample_to_dict(sample: ClassificationSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": sample.image_path,
        "class_id": sample.class_id,
        "class_name": sample.class_name,
    }


def build_base_novel_split(
    metadata: ClassificationMetadata,
    shots: int = 16,
    seed: int = 1,
    policy: str = "seeded_random_50_50",
) -> dict[str, Any]:
    if policy != "seeded_random_50_50":
        raise ValueError(f"Unsupported base-to-novel policy: {policy}")
    rng = random.Random(seed)
    class_names = sorted(metadata.class_names)
    rng.shuffle(class_names)
    n_base = math.ceil(len(class_names) / 2)
    base_classes = sorted(class_names[:n_base])
    novel_classes = sorted(class_names[n_base:])

    by_class = metadata.samples_by_class()
    train_base: list[ClassificationSample] = []
    test_base: list[ClassificationSample] = []
    test_novel: list[ClassificationSample] = []
    for class_name in base_classes:
        samples = list(by_class[class_name])
        rng.shuffle(samples)
        if len(samples) < shots:
            raise ValueError(f"Base class {class_name} has {len(samples)} samples, needs {shots}.")
        train_base.extend(samples[:shots])
        test_base.extend(samples[shots:])
    for class_name in novel_classes:
        test_novel.extend(by_class[class_name])

    return {
        "dataset": metadata.dataset,
        "version": metadata.version,
        "protocol": "base_to_novel",
        "seed": seed,
        "policy": policy,
        "num_shots": shots,
        "base_classes": base_classes,
        "novel_classes": novel_classes,
        "train_base": [_sample_to_dict(sample) for sample in sorted(train_base, key=lambda item: item.sample_id)],
        "test_base": [_sample_to_dict(sample) for sample in sorted(test_base, key=lambda item: item.sample_id)],
        "test_novel": [_sample_to_dict(sample) for sample in sorted(test_novel, key=lambda item: item.sample_id)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata")
    parser.add_argument("--out-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--split-root")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--policy", default="seeded_random_50_50")
    args = parser.parse_args()

    shots = args.num_shots if args.num_shots is not None else args.shots
    jobs: list[tuple[Path, Path]] = []
    if args.metadata and args.out_dir:
        jobs.append((Path(args.metadata), Path(args.out_dir)))
    elif args.data_root and args.split_root and args.datasets:
        for dataset in args.datasets:
            jobs.append(
                (
                    Path(args.data_root) / dataset / "metadata.json",
                    Path(args.split_root) / dataset,
                )
            )
    else:
        raise SystemExit("Provide either --metadata/--out-dir or --data-root/--split-root/--datasets.")

    for metadata_path, out_dir in jobs:
        metadata = ClassificationMetadata.from_json(metadata_path)
        for seed in args.seeds:
            split = build_base_novel_split(metadata, shots=shots, seed=seed, policy=args.policy)
            write_json(split, out_dir / f"seed{seed}" / "base_to_novel.json")


if __name__ == "__main__":
    main()
