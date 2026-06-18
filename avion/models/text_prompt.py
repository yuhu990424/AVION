from __future__ import annotations

from avion.utils.tensor import require_torch


class TextDeepPrompts:
    def __init__(self, num_layers: int, tokens_per_layer: int, width: int, init_std: float = 0.02) -> None:
        torch = require_torch()
        self.module = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(tokens_per_layer, width).normal_(std=init_std)) for _ in range(num_layers)]
        )

    def parameters(self):
        return self.module.parameters()

    def __len__(self) -> int:
        return len(self.module)

    def __getitem__(self, index: int):
        return self.module[index]

