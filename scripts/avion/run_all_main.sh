#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-${AVION_DATA:-/data/avion-repro/data}}
DEVICE=${2:-cpu}

CLASSIFICATION_DATASETS=(aid resisc45 eurosat whu_rs19 patternnet ucmerced)
RETRIEVAL_DATASETS=(rsitmd rsicd)
SHOTS=(1 2 4 8 16)
SEEDS=(1 2 3)

if [[ "${AVION_FULL_GRID:-0}" == "1" ]]; then
  for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
    for shots in "${SHOTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        bash scripts/avion/few_shot.sh "${DATA_ROOT}" "${dataset}" "${shots}" GeoRSCLIP "${seed}" "${DEVICE}"
      done
    done
  done

  for dataset in "${CLASSIFICATION_DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      bash scripts/avion/base2new.sh "${DATA_ROOT}" "${dataset}" GeoRSCLIP "${seed}" "${DEVICE}"
    done
  done

  for dataset in "${RETRIEVAL_DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      bash scripts/avion/retrieval.sh "${DATA_ROOT}" "${dataset}" GeoRSCLIP "${seed}" "${DEVICE}"
    done
  done
else
  bash scripts/avion/few_shot.sh "${DATA_ROOT}" aid 16 GeoRSCLIP 1 "${DEVICE}"
  bash scripts/avion/base2new.sh "${DATA_ROOT}" aid GeoRSCLIP 1 "${DEVICE}"
  bash scripts/avion/retrieval.sh "${DATA_ROOT}" rsitmd GeoRSCLIP 1 "${DEVICE}"
fi

echo "main orchestration complete: data=${DATA_ROOT} device=${DEVICE}"
