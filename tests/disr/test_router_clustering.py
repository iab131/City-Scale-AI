"""Sensor-cluster construction sanity test."""
import numpy as np

from models.disr.residual_router import build_sensor_clusters


def test_build_sensor_clusters_assigns_all_nodes():
    rng = np.random.default_rng(0)
    N = 50
    A = (rng.random((N, N)) < 0.10).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)
    labels = build_sensor_clusters(A, n_clusters=6, X_train=None)
    assert labels.shape == (N,)
    assert labels.dtype == np.int64
    assert set(np.unique(labels)).issubset(set(range(6)))
    # At least 2 distinct clusters appear (random graph should give that easily)
    assert len(set(labels.tolist())) >= 2
