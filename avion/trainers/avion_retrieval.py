from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avion.trainers.loss import cosine_alignment_loss, kd_kl_loss, linear_warmup_factor, symmetric_clip_loss
from avion.utils.tensor import require_torch


@dataclass(frozen=True)
class RetrievalLossConfig:
    lambda_img: float = 0.5
    lambda_text: float = 0.5
    lambda_logit: float = 1.0
    logit_warmup_ratio: float = 0.30
    temperature: float = 2.0


def compute_avion_retrieval_loss(
    student_similarity: Any,
    teacher_similarity: Any,
    student_image_features: Any,
    teacher_image_features: Any,
    student_text_features: Any,
    teacher_text_features: Any,
    global_step: int,
    total_steps: int,
    config: RetrievalLossConfig = RetrievalLossConfig(),
) -> dict[str, Any]:
    torch = require_torch()
    task = symmetric_clip_loss(student_similarity)
    img = cosine_alignment_loss(student_image_features, teacher_image_features)
    text = cosine_alignment_loss(student_text_features, teacher_text_features)
    logit_i2t = kd_kl_loss(student_similarity, teacher_similarity, temperature=config.temperature)
    logit_t2i = kd_kl_loss(student_similarity.t(), teacher_similarity.t(), temperature=config.temperature)
    logit = 0.5 * (logit_i2t + logit_t2i)
    warmup = linear_warmup_factor(global_step, total_steps, config.logit_warmup_ratio)
    total = task + config.lambda_img * img + config.lambda_text * text + config.lambda_logit * warmup * logit
    return {
        "loss": total,
        "Ltask": task.detach(),
        "Limg": img.detach(),
        "Ltext": text.detach(),
        "Llogit": logit.detach(),
        "lambda_logit_eff": torch.as_tensor(config.lambda_logit * warmup, device=student_similarity.device),
    }

