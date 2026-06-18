#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <metadata.json> <verify_report.json> [--check-pixels] [--hash-images]" >&2
  exit 2
fi

METADATA=$1
OUT=$2
shift 2

python -m avion.datasets.verify_metadata \
  --metadata "${METADATA}" \
  --out "${OUT}" \
  "$@"

echo "Wrote ${OUT}"
