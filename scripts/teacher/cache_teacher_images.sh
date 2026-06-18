#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON=${1:?metadata or split json required}
OUT=${2:?output .pt path required}
DEVICE=${3:-cpu}

python -m avion.teacher.cache_image_features \
  --metadata-or-split "${INPUT_JSON}" \
  --out "${OUT}" \
  --checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-H-14.pt" \
  --device "${DEVICE}"

