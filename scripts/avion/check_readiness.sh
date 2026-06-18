#!/usr/bin/env bash
set -euo pipefail

PROTOCOL=${1:-all}

python -m avion.utils.readiness \
  --data-root "${AVION_DATA:-/data/avion-repro/data}" \
  --ckpt-root "${AVION_CKPT:-/data/avion-repro/checkpoints}" \
  --cache-root "${AVION_CACHE:-/data/avion-repro/cache}" \
  --protocol "${PROTOCOL}"
