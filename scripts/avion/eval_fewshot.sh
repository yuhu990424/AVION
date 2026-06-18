#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-${AVION_OUTPUT:-/data/avion-repro/output}}
OUT_DIR=${2:-${ROOT}/tables/few_shot}

python -m avion.evaluation.collect_tables \
  --root "${ROOT}" \
  --protocol few_shot \
  --csv "${OUT_DIR}/metrics.csv" \
  --md "${OUT_DIR}/metrics.md" \
  --summary-csv "${OUT_DIR}/summary.csv" \
  --summary-md "${OUT_DIR}/summary.md"
