#!/usr/bin/env bash
set -euo pipefail

ANNOTATIONS=${1:?annotation jsonl required}
OUT=${2:?output .pt path required}
DEVICE=${3:-cpu}

python -m avion.teacher.cache_text_features \
  --annotations "${ANNOTATIONS}" \
  --out "${OUT}" \
  --checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-H-14.pt" \
  --device "${DEVICE}"

