from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avion.utils.hashing import sha256_file
from avion.utils.io import write_json


DEFAULT_GEORSCLIP_CHECKPOINTS = [
    {
        "filename": "RS5M_ViT-B-32.pt",
        "backbone": "ViT-B/32",
        "role": "student",
    },
    {
        "filename": "RS5M_ViT-H-14.pt",
        "backbone": "ViT-H/14",
        "role": "teacher",
    },
]


def verify_georsclip_checkpoints(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    report: dict[str, Any] = {}
    for spec in DEFAULT_GEORSCLIP_CHECKPOINTS:
        path = root / spec["filename"]
        report[spec["filename"]] = {
            "exists": path.exists(),
            "path": str(path),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            "backbone": spec["backbone"],
            "role": spec["role"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/avion-repro/checkpoints/georsclip")
    parser.add_argument("--out", default="/data/avion-repro/checkpoints/georsclip/model_manifest.json")
    args = parser.parse_args()
    write_json(verify_georsclip_checkpoints(args.root), args.out)


if __name__ == "__main__":
    main()

