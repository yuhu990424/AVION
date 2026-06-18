#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-${AVION_DATA:-/data/avion-repro/data}}
DATASET=${2:-rsitmd}
MODEL=${3:-GeoRSCLIP}
SEED=${4:-1}
DEVICE=${5:-cpu}

if [[ "${AVION_RUN_TRAINING:-0}" == "1" ]]; then
  python train.py \
    --config configs/paths/default.yaml \
    --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
    --config "configs/trainers/AVION/retrieval/${DATASET}.yaml" \
    --metadata "${DATA_ROOT}/processed/${DATASET}/metadata.json" \
    --teacher-retrieval-cache "${AVION_CACHE:-/data/avion-repro/cache}/teacher_retrieval/${DATASET}/vith14_retrieval_features.pt" \
    --student-checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-B-32.pt" \
    --output-dir "${AVION_OUTPUT:-/data/avion-repro/output}/retrieval/${DATASET}/AVION_GeoRSCLIP/vp8_tp4_kp30_beta10_gamma2_zeta3/seed${SEED}" \
    --epochs "${AVION_EPOCHS:-50}" \
    --batch-size "${AVION_BATCH_SIZE:-4}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
else
  python train.py \
    --config configs/paths/default.yaml \
    --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
    --config "configs/trainers/AVION/retrieval/${DATASET}.yaml" \
    --seed "${SEED}" \
    --dry-run
fi

echo "retrieval command complete: data=${DATA_ROOT} dataset=${DATASET} model=${MODEL}"
