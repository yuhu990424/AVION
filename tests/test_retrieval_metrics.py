import numpy as np

from avion.evaluation.metrics import retrieval_metrics


def test_retrieval_metrics_perfect_diagonal():
    sim = np.eye(3)
    relevance = np.eye(3, dtype=bool)
    metrics = retrieval_metrics(sim, relevance)
    assert metrics["I2T_R1"] == 100.0
    assert metrics["T2I_R1"] == 100.0
    assert metrics["mR"] == 100.0

