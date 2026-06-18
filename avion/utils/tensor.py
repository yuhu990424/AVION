from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for this operation.") from exc
    return torch


def l2_normalize_torch(x: Any, dim: int = -1, eps: float = 1e-12) -> Any:
    torch = require_torch()
    return torch.nn.functional.normalize(x, p=2, dim=dim, eps=eps)

