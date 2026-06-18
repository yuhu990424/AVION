from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avion.models.georsclip_loader import GeoRSCLIPConfig, load_georsclip
from avion.utils.io import read_jsonl, ensure_dir
from avion.utils.tensor import require_torch


def cache_candidate_text_features(
    rows: list[dict[str, Any]],
    model_config: GeoRSCLIPConfig,
    out_path: str | Path,
    batch_size: int = 256,
    device: str = "cpu",
) -> None:
    torch = require_torch()
    bundle = load_georsclip(model_config, device=device)
    features: dict[str, Any] = {}
    texts = [str(row["caption"]) for row in rows]
    keys = [f"{row['dataset']}::{row['class_name']}::{row['candidate_index']}" for row in rows]
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_keys = keys[start : start + batch_size]
            encoded = bundle.encode_text(batch_texts).detach().cpu()
            for key, vector in zip(batch_keys, encoded):
                features[key] = vector
    payload = {"features": features, "rows": rows, "model": model_config.__dict__}
    target = Path(out_path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="ViT-H/14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = list(read_jsonl(args.annotations))
    config = GeoRSCLIPConfig(args.backbone, args.pretrained, args.checkpoint)
    cache_candidate_text_features(rows, config, args.out, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()

