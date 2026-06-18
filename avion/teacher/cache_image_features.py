from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from avion.models.georsclip_loader import GeoRSCLIPConfig, load_georsclip
from avion.utils.io import read_json, ensure_dir
from avion.utils.tensor import require_torch


def _load_samples(metadata_or_split: dict[str, Any]) -> list[dict[str, Any]]:
    if "samples" in metadata_or_split:
        return list(metadata_or_split["samples"])
    rows: list[dict[str, Any]] = []
    for key in ("train", "test", "train_base", "test_base", "test_novel"):
        rows.extend(metadata_or_split.get(key, []))
    if not rows:
        raise ValueError("No image samples found in metadata/split file.")
    # Preserve unique sample ids in first-seen order.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["sample_id"], row)
    return list(unique.values())


def cache_image_features(
    samples: list[dict[str, Any]],
    model_config: GeoRSCLIPConfig,
    out_path: str | Path,
    batch_size: int = 64,
    device: str = "cpu",
) -> None:
    torch = require_torch()
    bundle = load_georsclip(model_config, device=device)
    features: dict[str, Any] = {}
    paths = [row["image_path"] for row in samples]
    sample_ids = [row["sample_id"] for row in samples]
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_ids = sample_ids[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(bundle.preprocess_val(image.convert("RGB")))
            batch = torch.stack(images).to(device)
            encoded = bundle.encode_image(batch).detach().cpu()
            for sample_id, vector in zip(batch_ids, encoded):
                features[sample_id] = vector
    payload = {"features": features, "model": model_config.__dict__}
    target = Path(out_path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-or-split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="ViT-H/14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = _load_samples(read_json(args.metadata_or_split))
    config = GeoRSCLIPConfig(args.backbone, args.pretrained, args.checkpoint)
    cache_image_features(rows, config, args.out, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()

