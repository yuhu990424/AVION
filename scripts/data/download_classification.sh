#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
Classification dataset downloader is intentionally conservative.

Preferred sources from the reproduction plan:
  AID        -> HuggingFace blanchon/AID or official Baidu fallback
  RESISC-45  -> HuggingFace timm/resisc45 or OneDrive fallback
  EuroSAT    -> Zenodo EuroSAT_RGB.zip
  WHU-RS19   -> HuggingFace jonathan-roberts1/WHU-RS19
  PatternNet -> HuggingFace blanchon/PatternNet
  UCMerced   -> HuggingFace blanchon/UC_Merced

Place or download raw datasets under:
  ${AVION_DATA:-/data/avion-repro/data}/raw/

Automated downloads will be added after source access is confirmed.
MSG

