#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
Dataset preparation is dataset-specific.

Use the generic builders:
  python -m avion.datasets.build_metadata --task classification --dataset AID --image-root <image-folder> --out <metadata.json>
  python -m avion.datasets.build_metadata --task retrieval --dataset RSITMD --image-root <images> --annotation-json <json> --out <metadata.json>

After metadata is created:
  bash scripts/data/verify_datasets.sh <metadata.json> <verify_report.json>
MSG

