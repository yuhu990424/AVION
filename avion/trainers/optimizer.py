from __future__ import annotations

from typing import Iterable

from avion.utils.tensor import require_torch


def build_adamw(parameters: Iterable[object], lr: float = 5e-4, weight_decay: float = 0.0):
    torch = require_torch()
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

