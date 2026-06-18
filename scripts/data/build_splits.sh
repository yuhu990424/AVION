#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AVION_DATA:-/data/avion-repro/data}/processed"
SPLIT_ROOT="${AVION_DATA:-/data/avion-repro/data}/splits"
DATASETS=(aid resisc45 eurosat whu_rs19 patternnet ucmerced)

python -m avion.datasets.split_tools.build_fewshot_splits \
  --data-root "${DATA_ROOT}" \
  --split-root "${SPLIT_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --shots 1 2 4 8 16 \
  --seeds 1 2 3

python -m avion.datasets.split_tools.build_base_novel_splits \
  --data-root "${DATA_ROOT}" \
  --split-root "${SPLIT_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --num-shots 16 \
  --seeds 1 2 3 \
  --policy seeded_random_50_50

