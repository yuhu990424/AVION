from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from avion.utils.hashing import sha256_file, stable_json_sha256
from avion.utils.io import write_json


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def build_run_manifest(
    config: dict[str, Any],
    artifacts: dict[str, str | Path],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_hashes: dict[str, str | None] = {}
    for name, path in artifacts.items():
        target = Path(path)
        artifact_hashes[name] = sha256_file(target) if target.exists() and target.is_file() else None
    return {
        "git_commit": git_commit(),
        "seed": seed,
        "config_hash": stable_json_sha256(config),
        "artifact_hashes": artifact_hashes,
        "extra": extra or {},
    }


def write_run_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    write_json(manifest, path)

