from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avion.trainers.loss import cosine_alignment_loss, kd_kl_loss, linear_warmup_factor
from avion.utils.tensor import require_torch


@dataclass(frozen=True)
class ClassificationLossConfig:
    lambda_img: float = 0.5
    lambda_text: float = 0.5
    lambda_logit: float = 1.0
    logit_warmup_ratio: float = 0.30
    temperature: float = 2.0


def compute_avion_classification_loss(
    student_logits: Any,
    labels: Any,
    student_image_features: Any,
    teacher_image_features: Any,
    student_text_features: Any,
    teacher_text_prototypes: Any,
    teacher_logits: Any,
    global_step: int,
    total_steps: int,
    config: ClassificationLossConfig = ClassificationLossConfig(),
    active_class_indices: list[int] | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    task = torch.nn.functional.cross_entropy(student_logits, labels)
    img = cosine_alignment_loss(student_image_features, teacher_image_features)
    text = cosine_alignment_loss(student_text_features, teacher_text_prototypes)
    logit = kd_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=config.temperature,
        active_indices=active_class_indices,
    )
    warmup = linear_warmup_factor(global_step, total_steps, config.logit_warmup_ratio)
    total = task + config.lambda_img * img + config.lambda_text * text + config.lambda_logit * warmup * logit
    return {
        "loss": total,
        "Ltask": task.detach(),
        "Limg": img.detach(),
        "Ltext": text.detach(),
        "Llogit": logit.detach(),
        "lambda_logit_eff": torch.as_tensor(config.lambda_logit * warmup, device=student_logits.device),
    }

