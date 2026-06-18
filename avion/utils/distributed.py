from __future__ import annotations


def is_distributed_available() -> bool:
    try:
        import torch

        return torch.distributed.is_available()
    except Exception:
        return False


def get_rank() -> int:
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def is_main_process() -> bool:
    return get_rank() == 0

