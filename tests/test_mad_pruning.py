import numpy as np

from avion.prototypes.mad_pruning import mad_keep_mask


def test_mad_pruning_removes_extreme_outlier():
    scores = np.array([0.51, 0.52, 0.50, 0.53, 10.0])
    kept, z_scores, median, mad = mad_keep_mask(scores, threshold=3.0)
    assert kept.tolist() == [True, True, True, True, False]
    assert median == 0.52
    assert mad > 0
    assert z_scores[-1] > 3.0

