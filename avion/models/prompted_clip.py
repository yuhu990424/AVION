from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avion.models.deep_prompt import DeepPromptSet
from avion.utils.tensor import require_torch


@dataclass(frozen=True)
class PromptConfig:
    vision_tokens_per_layer: int = 8
    text_tokens_per_layer: int = 4
    init_std: float = 0.02


_torch = require_torch()


class PromptedCLIP(_torch.nn.Module):
    """Deep-prompt adapter for standard OpenCLIP ViT-style CLIP models."""

    def __init__(self, model: Any, tokenizer: Any, config: PromptConfig = PromptConfig()) -> None:
        torch = require_torch()
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        visual_width = int(model.visual.transformer.width)
        text_width = int(model.transformer.width)
        vision_layers = len(model.visual.transformer.resblocks)
        text_layers = len(model.transformer.resblocks)
        self.prompts = DeepPromptSet(
            vision_layers=vision_layers,
            vision_tokens=config.vision_tokens_per_layer,
            vision_width=visual_width,
            text_layers=text_layers,
            text_tokens=config.text_tokens_per_layer,
            text_width=text_width,
            init_std=config.init_std,
        )
        initial_logit_scale = model.logit_scale.detach().clone() if hasattr(model, "logit_scale") else torch.tensor(1.0)
        self.logit_scale = torch.nn.Parameter(initial_logit_scale.float())

    def trainable_parameters(self):
        yield from self.prompts.parameters()
        yield self.logit_scale

    def freeze_backbone_check(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.model.parameters())

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        visual = self.model.visual
        if hasattr(visual, "conv1"):
            return visual.conv1.weight.dtype
        return next(self.model.parameters()).dtype

    def _run_block_nld(self, transformer: Any, block: Any, x: Any, attn_mask: Any | None = None) -> Any:
        if getattr(transformer, "batch_first", False):
            return block(x, attn_mask=attn_mask)
        return block(x.transpose(0, 1), attn_mask=attn_mask).transpose(0, 1)

    def _expand_text_attn_mask(self, attn_mask: Any | None, prompt_len: int, device: Any, dtype: Any) -> Any | None:
        if attn_mask is None:
            return None
        torch = require_torch()
        old_len = attn_mask.shape[-1]
        new_len = old_len + prompt_len
        expanded = torch.zeros((new_len, new_len), device=device, dtype=attn_mask.dtype)
        old_indices = torch.cat(
            [
                torch.arange(0, 1, device=device),
                torch.arange(1 + prompt_len, new_len, device=device),
            ]
        )
        expanded[old_indices[:, None], old_indices[None, :]] = attn_mask.to(device=device)
        return expanded.to(dtype=dtype) if expanded.dtype.is_floating_point else expanded

    def encode_image(self, images: Any) -> Any:
        torch = require_torch()
        visual = self.model.visual
        required = ["conv1", "class_embedding", "positional_embedding", "transformer", "ln_pre", "ln_post"]
        missing = [name for name in required if not hasattr(visual, name)]
        if missing:
            raise RuntimeError(f"PromptedCLIP only supports standard ViT visual towers; missing {missing}")

        x = images.to(dtype=self.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)

        for layer_idx, block in enumerate(visual.transformer.resblocks):
            prompt = self.prompts.visual_prompt(layer_idx, x.shape[0], x.dtype, x.device)
            x = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
            x = self._run_block_nld(visual.transformer, block, x, attn_mask=None)
            x = torch.cat([x[:, :1], x[:, 1 + prompt.shape[1] :]], dim=1)

        x = visual.ln_post(x[:, 0])
        if getattr(visual, "proj", None) is not None:
            x = x @ visual.proj
        return torch.nn.functional.normalize(x, dim=-1)

    def encode_text_tokens(self, tokens: Any) -> Any:
        torch = require_torch()
        model = self.model
        required = ["token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection"]
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise RuntimeError(f"PromptedCLIP only supports standard OpenCLIP text towers; missing {missing}")

        tokens = tokens.to(self.device)
        x = model.token_embedding(tokens).to(dtype=self.dtype)
        x = x + model.positional_embedding[: x.shape[1]].to(dtype=x.dtype, device=x.device)
        eot_indices = tokens.argmax(dim=-1)
        base_mask = getattr(model, "attn_mask", None)

        for layer_idx, block in enumerate(model.transformer.resblocks):
            prompt = self.prompts.text_prompt(layer_idx, x.shape[0], x.dtype, x.device)
            x = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)
            attn_mask = self._expand_text_attn_mask(base_mask, prompt.shape[1], x.device, x.dtype)
            x = self._run_block_nld(model.transformer, block, x, attn_mask=attn_mask)
            x = torch.cat([x[:, :1], x[:, 1 + prompt.shape[1] :]], dim=1)

        x = model.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ model.text_projection
        return torch.nn.functional.normalize(x, dim=-1)

    def encode_text(self, texts: list[str]) -> Any:
        tokens = self.tokenizer(texts)
        return self.encode_text_tokens(tokens)

    def logits(self, image_features: Any, text_features: Any) -> Any:
        return self.logit_scale.exp() * image_features @ text_features.t()
