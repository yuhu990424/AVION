import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_embedding_dim_mismatch_is_explicit():
    import torch

    from avion.trainers.classification_runner import _require_same_dim

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        _require_same_dim(torch.zeros(2, 512), torch.zeros(2, 1024), "student", "teacher")

