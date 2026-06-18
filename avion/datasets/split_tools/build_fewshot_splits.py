from __future__ import annotations

import argparse
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


def build_fewshot_split(metadata: ClassificationMetadata, shots: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    train: list[ClassificationSample] = []
    test: list[ClassificationSample] = []
    for class_name, samples in sorted(metadata.samples_by_class().items()):
        shuffled = list(samples)
        rng.shuffle(shuffled)
        if len(shuffled) < shots:
            raise ValueError(f"Class {class_name} has {len(shuffled)} samples, needs {shots}.")
        train.extend(shuffled[:shots])
        test.extend(shuffled[shots:])
    return {
        "dataset": metadata.dataset,
        "version": metadata.version,
        "protocol": "few_shot",
        "seed": seed,
        "num_shots": shots,
        "train": [_sample_to_dict(sample) for sample in sorted(train, key=lambda item: item.sample_id)],
        "test": [_sample_to_dict(sample) for sample in sorted(test, key=lambda item: item.sample_id)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata")
    parser.add_argument("--out-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--split-root")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

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
            for shots in args.shots:
                split = build_fewshot_split(metadata, shots=shots, seed=seed)
                write_json(split, out_dir / f"seed{seed}" / f"fewshot_{shots}.json")


if __name__ == "__main__":
    main()
