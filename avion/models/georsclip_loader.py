from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avion.utils.tensor import require_torch


@dataclass(frozen=True)
class GeoRSCLIPConfig:
    backbone: str
    pretrained: str
    checkpoint: str | None = None
    image_size: int = 224


@dataclass
class GeoRSCLIPBundle:
    model: Any
    preprocess_train: Any
    preprocess_val: Any
    tokenizer: Any

    def encode_image(self, images: Any) -> Any:
        features = self.model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_text(self, texts: list[str]) -> Any:
        tokens = self.tokenizer(texts)
        try:
            device = next(self.model.parameters()).device
            tokens = tokens.to(device)
        except Exception:
            pass
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload
    raise ValueError("Checkpoint payload must be a state dict or a dict containing one.")


def _strip_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_georsclip(config: GeoRSCLIPConfig, device: str = "cpu") -> GeoRSCLIPBundle:
    torch = require_torch()
    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError("open_clip_torch is required to load GeoRSCLIP.") from exc

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        config.backbone,
        pretrained=config.pretrained,
    )
    if config.checkpoint:
        checkpoint_path = Path(config.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GeoRSCLIP checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _strip_prefixes(_extract_state_dict(payload))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[georsclip] missing keys: {len(missing)}")
        if unexpected:
            print(f"[georsclip] unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(config.backbone)
    return GeoRSCLIPBundle(model=model, preprocess_train=preprocess_train, preprocess_val=preprocess_val, tokenizer=tokenizer)

