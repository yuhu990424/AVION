from __future__ import annotations

from pathlib import Path
from typing import Any

from avion.utils.io import ensure_dir
from avion.utils.tensor import require_torch


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    torch = require_torch()
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    torch = require_torch()
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dict: {path}")
    return payload

