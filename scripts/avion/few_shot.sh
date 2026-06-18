#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-${AVION_DATA:-/data/avion-repro/data}}
DATASET=${2:-aid}
SHOTS=${3:-16}
MODEL=${4:-GeoRSCLIP}
SEED=${5:-1}
DEVICE=${6:-cpu}

if [[ "${AVION_RUN_TRAINING:-0}" == "1" ]]; then
  python train.py \
    --config configs/paths/default.yaml \
    --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
    --config "configs/trainers/AVION/few_shot/${DATASET}.yaml" \
    --split "${DATA_ROOT}/splits/${DATASET}/seed${SEED}/fewshot_${SHOTS}.json" \
    --teacher-image-cache "${AVION_CACHE:-/data/avion-repro/cache}/teacher_images/${DATASET}/fewshot/seed${SEED}/shots_${SHOTS}/vith14_image_features.pt" \
    --teacher-text-prototypes "${AVION_CACHE:-/data/avion-repro/cache}/prototypes/${DATASET}/fewshot/seed${SEED}/shots_${SHOTS}/text_prototypes_selective.pt" \
    --student-checkpoint "${AVION_CKPT:-/data/avion-repro/checkpoints}/georsclip/RS5M_ViT-B-32.pt" \
    --output-dir "${AVION_OUTPUT:-/data/avion-repro/output}/${DATASET}/shots_${SHOTS}/AVION_GeoRSCLIP/vp8_tp4_kp30_beta10_gamma2_zeta3/seed${SEED}" \
    --epochs "${AVION_EPOCHS:-100}" \
    --batch-size "${AVION_BATCH_SIZE:-4}" \
    --shots "${SHOTS}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
else
  python train.py \
    --config configs/paths/default.yaml \
    --config configs/models/georsclip_vitb32_student_vith14_teacher.yaml \
    --config "configs/trainers/AVION/few_shot/${DATASET}.yaml" \
    --shots "${SHOTS}" \
    --seed "${SEED}" \
    --dry-run
fi

echo "few-shot command complete: data=${DATA_ROOT} dataset=${DATASET} shots=${SHOTS} model=${MODEL}"
