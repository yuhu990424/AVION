from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re


_OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+),([^}]+)\}$")


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read YAML config files.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_env_interpolations(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_env_interpolations(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_interpolations(item) for item in value]
    if isinstance(value, str):
        match = _OC_ENV_PATTERN.match(value)
        if match:
            env_name, fallback = match.groups()
            return os.environ.get(env_name, fallback)
        # Handle path strings that embed one or more ${oc.env:...} fragments.
        def repl(fragment: re.Match[str]) -> str:
            env_name, fallback = fragment.groups()
            return os.environ.get(env_name, fallback)

        return re.sub(r"\$\{oc\.env:([^,}]+),([^}]+)\}", repl, value)
    return value


def load_and_merge(paths: list[str | Path]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in paths:
        config = deep_merge(config, load_yaml(path))
    return resolve_env_interpolations(config)


def get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
