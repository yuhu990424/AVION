from __future__ import annotations

from typing import Any

from avion.utils.tensor import require_torch


def linear_warmup_factor(step: int, total_steps: int, ratio: float) -> float:
    warmup_steps = max(int(total_steps * ratio), 1)
    return min(max(step / warmup_steps, 0.0), 1.0)


def cosine_alignment_loss(student: Any, teacher: Any) -> Any:
    torch = require_torch()
    student = torch.nn.functional.normalize(student, dim=-1)
    teacher = torch.nn.functional.normalize(teacher, dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


def masked_log_softmax(logits: Any, active_indices: list[int], dim: int = -1) -> Any:
    torch = require_torch()
    mask = torch.full_like(logits, float("-inf"))
    mask.index_fill_(dim, torch.as_tensor(active_indices, device=logits.device), 0.0)
    return torch.nn.functional.log_softmax(logits + mask, dim=dim)


def masked_softmax(logits: Any, active_indices: list[int], dim: int = -1) -> Any:
    torch = require_torch()
    return masked_log_softmax(logits, active_indices=active_indices, dim=dim).exp()


def kd_kl_loss(student_logits: Any, teacher_logits: Any, temperature: float = 2.0, active_indices: list[int] | None = None) -> Any:
    torch = require_torch()
    if active_indices is None:
        teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    else:
        teacher_probs = masked_softmax(teacher_logits / temperature, active_indices=active_indices, dim=-1)
        student_log_probs = masked_log_softmax(student_logits / temperature, active_indices=active_indices, dim=-1)
    return (temperature**2) * torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def symmetric_clip_loss(similarity: Any) -> Any:
    torch = require_torch()
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    loss_i2t = torch.nn.functional.cross_entropy(similarity, labels)
    loss_t2i = torch.nn.functional.cross_entropy(similarity.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

