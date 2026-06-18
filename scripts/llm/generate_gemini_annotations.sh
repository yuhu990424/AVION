#!/usr/bin/env bash
set -euo pipefail

METADATA=${1:?metadata json required}
OUT=${2:?output jsonl required}
KP=${3:-30}

python -m avion.llm.generate_class_candidates \
  --metadata "${METADATA}" \
  --out "${OUT}" \
  --kp "${KP}"

