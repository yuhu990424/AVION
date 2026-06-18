import numpy as np

from avion.prototypes.selective_aggregation import aggregate_text_prototype


def test_selective_aggregation_normalizes_and_weights_flags():
    visual = np.array([1.0, 0.0])
    texts = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    flags = np.array([0, 1, 0])
    result = aggregate_text_prototype(visual, texts, flags, beta=1.0, gamma=2.0, mad_threshold=100.0)
    assert np.isclose(np.linalg.norm(result.prototype), 1.0)
    assert result.weights[1] > result.weights[0]
    assert np.isclose(result.weights.sum(), 1.0)

