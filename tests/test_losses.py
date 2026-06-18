import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_kd_loss_and_warmup():
    import torch

    from avion.trainers.loss import kd_kl_loss, linear_warmup_factor

    student = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    teacher = torch.tensor([[1.5, 1.0], [1.0, 1.5]])
    loss = kd_kl_loss(student, teacher, temperature=2.0)
    assert loss.ndim == 0
    assert linear_warmup_factor(1, 10, 0.3) > 0.0

