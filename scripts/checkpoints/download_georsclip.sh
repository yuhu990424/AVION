#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
GeoRSCLIP checkpoints must be placed manually or downloaded from the approved
source for your environment.

Expected files:
  ${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-B-32.pt
  ${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-H-14.pt

After placing them, run:
  bash scripts/checkpoints/verify_georsclip.sh
MSG
