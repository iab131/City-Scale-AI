"""Unit tests for SemanticGraph kNN + eigenbasis."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.ssm_magma.semantic_graph import SemanticGraph


def test_knn_laplacian_is_symmetric_and_psd():
    g = SemanticGraph(n_nodes=30, d_sem=8, k_neighbors=4, k_modes=10)
    L = g._knn_laplacian()
    assert L.shape == (30, 30)
    # symmetric
    np.testing.assert_allclose(L.detach().numpy(),
                                L.detach().numpy().T,
                                atol=1e-5)
    # eigenvalues non-negative (PSD)
    evals = torch.linalg.eigvalsh(L)
    assert float(evals.min()) > -1e-4


def test_basis_shape_and_orthogonality():
    g = SemanticGraph(n_nodes=40, d_sem=8, k_neighbors=5, k_modes=12,
                       refresh_steps=1)
    g.eval()
    U, evals = g.get_basis()
    assert U.shape == (40, 12)
    assert evals.shape == (12,)
    UTU = U.T @ U
    I = torch.eye(12)
    np.testing.assert_allclose(UTU.detach().numpy(), I.numpy(), atol=1e-4)


def test_refresh_counter_progresses_in_train():
    g = SemanticGraph(n_nodes=30, d_sem=8, k_neighbors=4, k_modes=8,
                       refresh_steps=5)
    g.train()
    for _ in range(10):
        g.maybe_refresh()
    assert int(g.step_counter.item()) >= 10


def test_ortho_penalty_low_after_basis_build():
    g = SemanticGraph(n_nodes=30, d_sem=8, k_neighbors=4, k_modes=10)
    g._compute_basis()
    pen = float(g.ortho_penalty())
    assert pen < 1e-5
