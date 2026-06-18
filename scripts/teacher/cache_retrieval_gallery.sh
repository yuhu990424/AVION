#!/usr/bin/env bash
set -euo pipefail

METADATA=${1:?retrieval metadata json required}
OUT=${2:?output .pt path required}
SPLIT=${3:-test}
DEVICE=${4:-cpu}

python -m avion.teacher.cache_retrieval_features \
  --metadata "${METADATA}" \
  --out "${OUT}" \
  --split "${SPLIT}" \
  --checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-H-14.pt" \
  --device "${DEVICE}"

