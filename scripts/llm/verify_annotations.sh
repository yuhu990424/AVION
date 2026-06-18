#!/usr/bin/env bash
set -euo pipefail

METADATA=${1:?metadata json required}
ANNOTATIONS=${2:?annotation jsonl required}
OUT=${3:?report json required}
KP=${4:-30}

python -m avion.llm.verify_annotations \
  --metadata "${METADATA}" \
  --annotations "${ANNOTATIONS}" \
  --out "${OUT}" \
  --expected-kp "${KP}"

echo "Wrote ${OUT}"
