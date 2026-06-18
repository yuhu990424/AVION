#!/usr/bin/env bash
set -euo pipefail

METADATA=${1:?metadata json required}
OUT=${2:-}
DEVICE=${3:-cpu}

CMD=(
  python -m avion.evaluation.zero_shot
  --metadata "${METADATA}"
  --checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-B-32.pt"
  --device "${DEVICE}"
)

if [[ -n "${OUT}" ]]; then
  CMD+=(--out "${OUT}")
fi

"${CMD[@]}"

