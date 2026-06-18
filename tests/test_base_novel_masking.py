import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_masked_softmax_excludes_novel_classes():
    import torch

    from avion.trainers.loss import masked_softmax

    logits = torch.tensor([[1.0, 2.0, 100.0]])
    probs = masked_softmax(logits, active_indices=[0, 1])
    assert torch.isclose(probs[0, 2], torch.tensor(0.0))
    assert torch.isclose(probs[0, :2].sum(), torch.tensor(1.0))

