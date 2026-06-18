from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avion.datasets.registry import DATASET_REGISTRY
from avion.utils.hashing import sha256_file
from avion.utils.io import read_json, write_json


def _verify_image(path: str, check_pixels: bool) -> tuple[bool, str | None]:
    target = Path(path)
    if not target.exists():
        return False, "missing"
    if check_pixels:
        try:
            from PIL import Image

            with Image.open(target) as image:
                image.verify()
        except Exception as exc:
            return False, f"bad_image:{exc}"
    return True, None


def verify_classification_metadata(metadata: dict[str, Any], check_pixels: bool = False, hash_images: bool = False) -> dict[str, Any]:
    dataset_key = str(metadata["dataset"]).lower().replace("-", "_")
    spec = DATASET_REGISTRY.get(dataset_key)
    classes = metadata.get("classes", [])
    samples = metadata.get("samples", [])
    errors: list[str] = []
    if spec and spec.expected_classes is not None and len(classes) != spec.expected_classes:
        errors.append(f"expected {spec.expected_classes} classes, found {len(classes)}")
    if spec and spec.expected_images is not None and len(samples) != spec.expected_images:
        errors.append(f"expected {spec.expected_images} images, found {len(samples)}")

    seen_ids: set[str] = set()
    missing_or_bad: list[dict[str, str]] = []
    image_hashes: dict[str, str] = {}
    duplicate_hashes: dict[str, list[str]] = {}
    for row in samples:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        ok, reason = _verify_image(row["image_path"], check_pixels=check_pixels)
        if not ok:
            missing_or_bad.append({"sample_id": sample_id, "reason": reason or "invalid"})
        if hash_images and ok:
            digest = sha256_file(row["image_path"])
            image_hashes[sample_id] = digest
            duplicate_hashes.setdefault(digest, []).append(sample_id)

    duplicates = {digest: ids for digest, ids in duplicate_hashes.items() if len(ids) > 1}
    return {
        "dataset": metadata["dataset"],
        "task": "classification",
        "num_classes": len(classes),
        "num_samples": len(samples),
        "errors": errors,
        "missing_or_bad": missing_or_bad,
        "num_missing_or_bad": len(missing_or_bad),
        "num_duplicate_hashes": len(duplicates),
        "duplicate_hashes": duplicates,
        "ok": not errors and not missing_or_bad,
    }


def verify_retrieval_metadata(metadata: dict[str, Any], check_pixels: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    images = metadata.get("images", [])
    captions = metadata.get("captions", [])
    pairs = metadata.get("pairs", [])
    image_ids = {row["image_id"] for row in images}
    caption_ids = {row["caption_id"] for row in captions}
    for pair in pairs:
        if pair["image_id"] not in image_ids:
            errors.append(f"pair references missing image_id: {pair['image_id']}")
        if pair["caption_id"] not in caption_ids:
            errors.append(f"pair references missing caption_id: {pair['caption_id']}")
    missing_or_bad = []
    for row in images:
        ok, reason = _verify_image(row["image_path"], check_pixels=check_pixels)
        if not ok:
            missing_or_bad.append({"image_id": row["image_id"], "reason": reason or "invalid"})
    return {
        "dataset": metadata["dataset"],
        "task": "retrieval",
        "num_images": len(images),
        "num_captions": len(captions),
        "num_pairs": len(pairs),
        "errors": errors,
        "missing_or_bad": missing_or_bad,
        "num_missing_or_bad": len(missing_or_bad),
        "ok": not errors and not missing_or_bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-pixels", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()

    metadata = read_json(args.metadata)
    if "samples" in metadata:
        report = verify_classification_metadata(metadata, check_pixels=args.check_pixels, hash_images=args.hash_images)
    else:
        report = verify_retrieval_metadata(metadata, check_pixels=args.check_pixels)
    write_json(report, args.out)


if __name__ == "__main__":
    main()

