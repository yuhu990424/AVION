#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
Retrieval datasets may require Google Drive/Baidu/MEGA access.

Manual placement expected if automated download fails:
  RSITMD -> ${AVION_DATA:-/data/avion-repro/data}/raw/RSITMD/
  RSICD  -> ${AVION_DATA:-/data/avion-repro/data}/raw/RSICD/

Expected RSITMD files include images/ and dataset_RSITMD.json.
Expected RSICD files include images and dataset_rsicd.json.
MSG

