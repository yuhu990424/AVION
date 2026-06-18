#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:?split or metadata json required}
ANNOTATIONS=${2:?annotation jsonl required}
IMAGE_CACHE=${3:?teacher image cache .pt required}
TEXT_CACHE=${4:?teacher text cache .pt required}
OUT_DIR=${5:?output directory required}
DATASET=${6:?dataset key required}
PROTOCOL=${7:?protocol required}
SEED=${8:-1}
SHOTS=${9:-}

CMD=(
  python -m avion.prototypes.build_class_prototypes
  --split-or-metadata "${SPLIT}"
  --annotations "${ANNOTATIONS}"
  --image-cache "${IMAGE_CACHE}"
  --text-cache "${TEXT_CACHE}"
  --out-dir "${OUT_DIR}"
  --dataset "${DATASET}"
  --protocol "${PROTOCOL}"
  --seed "${SEED}"
)

if [[ -n "${SHOTS}" ]]; then
  CMD+=(--shots "${SHOTS}")
fi

"${CMD[@]}"

