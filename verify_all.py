from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from avion.models.verify_checkpoints import verify_georsclip_checkpoints
from avion.utils.readiness import build_readiness_checks, build_readiness_report


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_environment() -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {
            "AVION_DATA": os.environ.get("AVION_DATA"),
            "AVION_CKPT": os.environ.get("AVION_CKPT"),
            "AVION_CACHE": os.environ.get("AVION_CACHE"),
            "AVION_OUTPUT": os.environ.get("AVION_OUTPUT"),
            "GEMINI_API_KEY_present": bool(os.environ.get("GEMINI_API_KEY")),
        },
        "modules": {
            "numpy": _module_available("numpy"),
            "torch": _module_available("torch"),
            "open_clip": _module_available("open_clip"),
            "google.genai": _module_available("google.genai"),
        },
    }
    if _module_available("torch"):
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    return report


def check_imports() -> dict[str, object]:
    modules = [
        "avion.datasets.base",
        "avion.datasets.split_tools.build_fewshot_splits",
        "avion.llm.rs_flag",
        "avion.prototypes.selective_aggregation",
        "avion.evaluation.metrics",
        "avion.trainers.loss",
    ]
    result: dict[str, object] = {}
    for module in modules:
        try:
            __import__(module)
            result[module] = True
        except Exception as exc:
            result[module] = repr(exc)
    return result


def check_git() -> dict[str, object]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, text=True).strip()

    try:
        return {
            "branch": run(["git", "branch", "--show-current"]),
            "status": run(["git", "status", "--short"]),
            "remotes": run(["git", "remote", "-v"]),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def check_georsclip() -> dict[str, object]:
    ckpt_root = Path(os.environ.get("AVION_CKPT", "/data/avion-repro/checkpoints")) / "georsclip"
    report = verify_georsclip_checkpoints(ckpt_root)
    report["_module_open_clip_available"] = _module_available("open_clip")
    return report


def check_readiness() -> dict[str, object]:
    checks = build_readiness_checks(
        data_root=os.environ.get("AVION_DATA", "/data/avion-repro/data"),
        ckpt_root=os.environ.get("AVION_CKPT", "/data/avion-repro/checkpoints"),
        cache_root=os.environ.get("AVION_CACHE", "/data/avion-repro/cache"),
    )
    return build_readiness_report(checks, max_missing=50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["environment", "imports", "git", "georsclip", "readiness", "all"], default="all")
    args = parser.parse_args()
    report: dict[str, object] = {}
    if args.stage in ("environment", "all"):
        report["environment"] = check_environment()
    if args.stage in ("imports", "all"):
        report["imports"] = check_imports()
    if args.stage in ("git", "all"):
        report["git"] = check_git()
    if args.stage in ("georsclip", "all"):
        report["georsclip"] = check_georsclip()
    if args.stage in ("readiness", "all"):
        report["readiness"] = check_readiness()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
