#!/usr/bin/env bash
set -euo pipefail

CKPT_ROOT="${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip"
OUT="${CKPT_ROOT}/model_manifest.json"

python -m avion.models.verify_checkpoints \
  --root "${CKPT_ROOT}" \
  --out "${OUT}"

echo "Wrote ${OUT}"
