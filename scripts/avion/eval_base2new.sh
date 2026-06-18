#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-${AVION_OUTPUT:-/data/avion-repro/output}}
OUT_DIR=${2:-${ROOT}/tables/base_to_novel}

python -m avion.evaluation.collect_tables \
  --root "${ROOT}" \
  --protocol base_to_novel \
  --csv "${OUT_DIR}/metrics.csv" \
  --md "${OUT_DIR}/metrics.md" \
  --summary-csv "${OUT_DIR}/summary.csv" \
  --summary-md "${OUT_DIR}/summary.md"
