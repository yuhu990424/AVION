from __future__ import annotations

import argparse

from avion.datasets.classification_base import build_imagefolder_metadata
from avion.datasets.retrieval_base import build_retrieval_metadata_from_rs_json
from avion.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "retrieval"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--annotation-json", help="Required for retrieval datasets.")
    parser.add_argument("--version", default="public_rebuild_v1")
    args = parser.parse_args()

    if args.task == "classification":
        metadata = build_imagefolder_metadata(args.dataset, args.image_root, version=args.version)
    else:
        if not args.annotation_json:
            raise ValueError("--annotation-json is required for retrieval metadata.")
        metadata = build_retrieval_metadata_from_rs_json(args.dataset, args.annotation_json, args.image_root)
    write_json(metadata, args.out)


if __name__ == "__main__":
    main()

