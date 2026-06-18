from __future__ import annotations

from typing import Any

from avion.utils.tensor import require_torch


_torch = require_torch()


class DeepPromptSet(_torch.nn.Module):
    """Parameter container for AVION deep prompts.

    OpenCLIP layer injection is handled by trainer/model adapters. This class
    keeps prompt parameters explicit and easy to count/test.
    """

    def __init__(
        self,
        vision_layers: int,
        vision_tokens: int,
        vision_width: int,
        text_layers: int,
        text_tokens: int,
        text_width: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.visual = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(vision_tokens, vision_width).normal_(std=init_std)) for _ in range(vision_layers)]
        )
        self.text = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(text_tokens, text_width).normal_(std=init_std)) for _ in range(text_layers)]
        )

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def visual_prompt(self, layer_idx: int, batch_size: int, dtype: Any, device: Any) -> Any:
        prompt = self.visual[layer_idx].to(device=device, dtype=dtype)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    def text_prompt(self, layer_idx: int, batch_size: int, dtype: Any, device: Any) -> Any:
        prompt = self.text[layer_idx].to(device=device, dtype=dtype)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)
