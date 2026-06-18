from __future__ import annotations

from pathlib import Path
from typing import Any

from avion.datasets.class_names import canonicalize_class_name, prompt_name


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_imagefolder_metadata(dataset: str, image_root: str | Path, version: str = "public_rebuild_v1") -> dict[str, Any]:
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    class_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {image_root}")

    classes: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for class_id, class_dir in enumerate(class_dirs):
        raw_name = class_dir.name
        canonical_name = canonicalize_class_name(raw_name)
        classes.append(
            {
                "class_id": class_id,
                "raw_name": raw_name,
                "canonical_name": canonical_name,
                "prompt_name": prompt_name(raw_name),
            }
        )
        for image_path in iter_image_files(class_dir):
            rel = image_path.relative_to(image_root).as_posix()
            samples.append(
                {
                    "sample_id": f"{dataset}/{rel}",
                    "image_path": str(image_path.resolve()),
                    "class_id": class_id,
                    "class_name": canonical_name,
                }
            )

    return {
        "dataset": dataset,
        "version": version,
        "num_classes": len(classes),
        "num_samples": len(samples),
        "classes": classes,
        "samples": sorted(samples, key=lambda row: row["sample_id"]),
    }

