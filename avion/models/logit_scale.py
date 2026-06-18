from __future__ import annotations

import math

from avion.utils.tensor import require_torch


class LearnableLogitScale:
    def __init__(self, initial_value: float = 1 / 0.07) -> None:
        torch = require_torch()
        self.module = torch.nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))

    def value(self):
        torch = require_torch()
        return torch.exp(self.module)

    def parameters(self):
        yield self.module

