from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from avion.config.defaults import CLASSIFICATION_DATASETS, DEFAULT_SEEDS, DEFAULT_SHOTS, RETRIEVAL_DATASETS


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: str
    required_for: str
    exists: bool


def _check(name: str, path: Path, required_for: str) -> ArtifactCheck:
    return ArtifactCheck(name=name, path=str(path), required_for=required_for, exists=path.exists())


def _as_list(values: Iterable[str] | None, default: list[str]) -> list[str]:
    return list(values) if values is not None else list(default)


def _as_int_list(values: Iterable[int] | None, default: list[int]) -> list[int]:
    return [int(value) for value in values] if values is not None else list(default)


def build_readiness_checks(
    data_root: str | Path = "/data/avion-repro/data",
    ckpt_root: str | Path = "/data/avion-repro/checkpoints",
    cache_root: str | Path = "/data/avion-repro/cache",
    protocols: Iterable[str] | None = None,
    classification_datasets: Iterable[str] | None = None,
    retrieval_datasets: Iterable[str] | None = None,
    shots: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[ArtifactCheck]:
    data_root = Path(data_root)
    ckpt_root = Path(ckpt_root)
    cache_root = Path(cache_root)
    protocols = _as_list(protocols, ["few_shot", "base_to_novel", "retrieval"])
    classification_datasets = _as_list(classification_datasets, CLASSIFICATION_DATASETS)
    retrieval_datasets = _as_list(retrieval_datasets, RETRIEVAL_DATASETS)
    shots = _as_int_list(shots, DEFAULT_SHOTS)
    seeds = _as_int_list(seeds, DEFAULT_SEEDS)

    checks: list[ArtifactCheck] = [
        _check("student_georsclip_vitb32", ckpt_root / "georsclip" / "RS5M_ViT-B-32.pt", "all_training"),
        _check("teacher_georsclip_vith14", ckpt_root / "georsclip" / "RS5M_ViT-H-14.pt", "teacher_cache_and_prototypes"),
    ]

    if "few_shot" in protocols:
        for dataset in classification_datasets:
            checks.append(_check(f"{dataset}_metadata", data_root / "processed" / dataset / "metadata.json", "few_shot"))
            for seed in seeds:
                for shot in shots:
                    stem = f"{dataset}_fewshot_seed{seed}_shots{shot}"
                    checks.extend(
                        [
                            _check(stem + "_split", data_root / "splits" / dataset / f"seed{seed}" / f"fewshot_{shot}.json", "few_shot"),
                            _check(
                                stem + "_teacher_images",
                                cache_root / "teacher_images" / dataset / "fewshot" / f"seed{seed}" / f"shots_{shot}" / "vith14_image_features.pt",
                                "few_shot",
                            ),
                            _check(
                                stem + "_text_prototypes",
                                cache_root / "prototypes" / dataset / "fewshot" / f"seed{seed}" / f"shots_{shot}" / "text_prototypes_selective.pt",
                                "few_shot",
                            ),
                        ]
                    )

    if "base_to_novel" in protocols:
        for dataset in classification_datasets:
            checks.append(_check(f"{dataset}_metadata", data_root / "processed" / dataset / "metadata.json", "base_to_novel"))
            for seed in seeds:
                stem = f"{dataset}_base2new_seed{seed}"
                checks.extend(
                    [
                        _check(stem + "_split", data_root / "splits" / dataset / f"seed{seed}" / "base_to_novel.json", "base_to_novel"),
                        _check(
                            stem + "_teacher_images",
                            cache_root / "teacher_images" / dataset / "base2new" / f"seed{seed}" / "vith14_base_image_features.pt",
                            "base_to_novel",
                        ),
                        _check(
                            stem + "_text_prototypes",
                            cache_root / "prototypes" / dataset / "base2new" / f"seed{seed}" / "text_prototypes_selective.pt",
                            "base_to_novel",
                        ),
                    ]
                )

    if "retrieval" in protocols:
        for dataset in retrieval_datasets:
            checks.extend(
                [
                    _check(f"{dataset}_metadata", data_root / "processed" / dataset / "metadata.json", "retrieval"),
                    _check(
                        f"{dataset}_teacher_retrieval_cache",
                        cache_root / "teacher_retrieval" / dataset / "vith14_retrieval_features.pt",
                        "retrieval",
                    ),
                ]
            )
    return checks


def build_readiness_report(checks: list[ArtifactCheck], max_missing: int | None = None) -> dict[str, object]:
    rows = [asdict(check) for check in checks]
    missing = [row for row in rows if not row["exists"]]
    displayed_missing = missing[:max_missing] if max_missing is not None else missing
    return {
        "total": len(rows),
        "present": len(rows) - len(missing),
        "missing": len(missing),
        "ready": not missing,
        "missing_required": displayed_missing,
        "missing_truncated": max_missing is not None and len(missing) > max_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/avion-repro/data")
    parser.add_argument("--ckpt-root", default="/data/avion-repro/checkpoints")
    parser.add_argument("--cache-root", default="/data/avion-repro/cache")
    parser.add_argument("--protocol", choices=["few_shot", "base_to_novel", "retrieval", "all"], default="all")
    parser.add_argument("--classification-datasets", nargs="+")
    parser.add_argument("--retrieval-datasets", nargs="+")
    parser.add_argument("--shots", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-missing", type=int, default=100)
    parser.add_argument("--show-present", action="store_true")
    args = parser.parse_args()
    protocols = ["few_shot", "base_to_novel", "retrieval"] if args.protocol == "all" else [args.protocol]
    checks = build_readiness_checks(
        data_root=args.data_root,
        ckpt_root=args.ckpt_root,
        cache_root=args.cache_root,
        protocols=protocols,
        classification_datasets=args.classification_datasets,
        retrieval_datasets=args.retrieval_datasets,
        shots=args.shots,
        seeds=args.seeds,
    )
    report = build_readiness_report(checks, max_missing=args.max_missing)
    if args.show_present:
        report["checks"] = [asdict(check) for check in checks]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
