#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-${AVION_DATA:-/data/avion-repro/data}}
DATASET=${2:-aid}
MODEL=${3:-GeoRSCLIP}
SEED=${4:-1}
DEVICE=${5:-cpu}

if [[ "${AVION_RUN_TRAINING:-0}" == "1" ]]; then
  python train.py \
    --config configs/paths/default.yaml \
    --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
    --config "configs/trainers/AVION/base_to_novel/${DATASET}.yaml" \
    --split "${DATA_ROOT}/splits/${DATASET}/seed${SEED}/base_to_novel.json" \
    --teacher-image-cache "${AVION_CACHE:-/data/avion-repro/cache}/teacher_images/${DATASET}/base2new/seed${SEED}/vith14_base_image_features.pt" \
    --teacher-text-prototypes "${AVION_CACHE:-/data/avion-repro/cache}/prototypes/${DATASET}/base2new/seed${SEED}/text_prototypes_selective.pt" \
    --student-checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-B-32.pt" \
    --output-dir "${AVION_OUTPUT:-/data/avion-repro/output}/base2new/train_base/${DATASET}/shots_16/AVION_GeoRSCLIP/vp8_tp4_kp30_beta10_gamma2_zeta3/seed${SEED}" \
    --epochs "${AVION_EPOCHS:-50}" \
    --batch-size "${AVION_BATCH_SIZE:-4}" \
    --shots "${AVION_BASE_SHOTS:-16}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
else
  python train.py \
    --config configs/paths/default.yaml \
    --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
    --config "configs/trainers/AVION/base_to_novel/${DATASET}.yaml" \
    --shots "${AVION_BASE_SHOTS:-16}" \
    --seed "${SEED}" \
    --dry-run
fi

echo "base-to-novel command complete: data=${DATA_ROOT} dataset=${DATASET} model=${MODEL}"
