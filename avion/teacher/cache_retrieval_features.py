from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from avion.datasets.retrieval_base import RetrievalMetadata
from avion.models.georsclip_loader import GeoRSCLIPConfig, load_georsclip
from avion.utils.io import ensure_dir
from avion.utils.tensor import require_torch


def cache_retrieval_features(
    metadata: RetrievalMetadata,
    model_config: GeoRSCLIPConfig,
    out_path: str | Path,
    split: str | None = None,
    batch_size: int = 64,
    device: str = "cpu",
) -> None:
    torch = require_torch()
    bundle = load_georsclip(model_config, device=device)
    images = [row for row in metadata.images if split is None or row.split == split]
    captions = [row for row in metadata.captions if any(pair.caption_id == row.caption_id and (split is None or pair.split == split) for pair in metadata.pairs)]
    image_features: dict[str, Any] = {}
    caption_features: dict[str, Any] = {}
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_rows = images[start : start + batch_size]
            tensors = []
            for row in batch_rows:
                with Image.open(row.image_path) as image:
                    tensors.append(bundle.preprocess_val(image.convert("RGB")))
            encoded = bundle.encode_image(torch.stack(tensors).to(device)).detach().cpu()
            for row, vector in zip(batch_rows, encoded):
                image_features[row.image_id] = vector
        for start in range(0, len(captions), batch_size):
            batch_rows = captions[start : start + batch_size]
            encoded = bundle.encode_text([row.caption for row in batch_rows]).detach().cpu()
            for row, vector in zip(batch_rows, encoded):
                caption_features[row.caption_id] = vector
    payload = {"image_features": image_features, "caption_features": caption_features, "model": model_config.__dict__}
    target = Path(out_path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split")
    parser.add_argument("--backbone", default="ViT-H/14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metadata = RetrievalMetadata.from_json(args.metadata)
    config = GeoRSCLIPConfig(args.backbone, args.pretrained, args.checkpoint)
    cache_retrieval_features(metadata, config, args.out, split=args.split, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()

