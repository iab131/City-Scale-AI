"""Unit tests for spectral basis and magnetic Laplacian utilities."""
import math
import numpy as np
import torch

from models.disr.spectral_basis import (
    build_symmetric_basis,
    project as sym_project,
    unproject as sym_unproject,
)
from models.disr.magnetic_laplacian import (
    estimate_lagged_direction,
    build_magnetic_laplacian,
    eigendecompose_hermitian,
    magnetic_basis_from_adjacency,
    project_complex,
    unproject_complex,
)


def _random_sym_adj(N: int, edge_prob: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = (rng.random((N, N)) < edge_prob).astype(np.float64)
    M = np.triu(M, 1)
    M = M + M.T
    return M


def _random_dir_adj(N: int, edge_prob: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = (rng.random((N, N)) < edge_prob).astype(np.float32)
    np.fill_diagonal(M, 0.0)
    return M


# ------------------ symmetric basis ------------------
def test_symmetric_basis_shape_and_orthogonality():
    A = _random_sym_adj(60, 0.05, seed=0)
    evals, U = build_symmetric_basis(A, k=20, side="low")
    assert evals.shape == (20,)
    assert U.shape == (60, 20)
    # Columns should be orthonormal to numerical precision.
    UTU = U.T @ U
    np.testing.assert_allclose(UTU, np.eye(20), atol=1e-4)


def test_symmetric_project_inverse_shape():
    A = _random_sym_adj(50, 0.05, seed=1)
    _, U = build_symmetric_basis(A, k=16, side="low")
    Ut = torch.from_numpy(U).float()
    X = torch.randn(2, 12, 50, 1)
    Z = sym_project(X, Ut)
    assert Z.shape == (2, 12, 16, 1), Z.shape
    X_back = sym_unproject(Z, Ut)
    assert X_back.shape == X.shape


# ------------------ magnetic Laplacian ------------------
def test_magnetic_laplacian_is_hermitian():
    A_dir = torch.from_numpy(_random_dir_adj(40, 0.06, seed=2))
    L = build_magnetic_laplacian(A_dir, q=0.10, normalized=True)
    assert L.dtype == torch.complex64
    diff = (L - L.conj().T).abs().max().item()
    assert diff < 1e-5, f"L_q not Hermitian: max|L - L^H|={diff}"


def test_magnetic_laplacian_unnormalized_hermitian():
    A_dir = torch.from_numpy(_random_dir_adj(35, 0.07, seed=3))
    L = build_magnetic_laplacian(A_dir, q=0.15, normalized=False)
    diff = (L - L.conj().T).abs().max().item()
    assert diff < 1e-5


def test_eigendecompose_shapes():
    A_dir = torch.from_numpy(_random_dir_adj(50, 0.08, seed=4))
    L = build_magnetic_laplacian(A_dir, q=0.10)
    evals, U = eigendecompose_hermitian(L, k=24, side="low")
    assert evals.shape == (24,)
    assert U.shape == (50, 24)
    assert torch.is_complex(U)


def test_magnetic_projection_shape_and_real_unproject():
    A_dir = torch.from_numpy(_random_dir_adj(40, 0.05, seed=5))
    L = build_magnetic_laplacian(A_dir, q=0.10)
    _, U = eigendecompose_hermitian(L, k=16, side="low")
    X = torch.randn(2, 12, 40, 1)
    Z = project_complex(X, U)
    assert Z.shape == (2, 12, 16, 1) and torch.is_complex(Z)
    Xb = unproject_complex(Z, U)
    assert Xb.shape == X.shape and Xb.dtype == X.dtype


def test_real_imag_split_roundtrip_shape():
    """
    Simulate the network path: project, split to real/imag channels, run a
    real-valued identity, recombine, unproject. Output must be real and
    shape-preserving.
    """
    A_dir = torch.from_numpy(_random_dir_adj(45, 0.05, seed=6))
    L = build_magnetic_laplacian(A_dir, q=0.10)
    _, U = eigendecompose_hermitian(L, k=20, side="low")
    X = torch.randn(2, 12, 45, 1)
    Z = project_complex(X, U)  # complex [B, T, K, 1]
    re = Z.real.squeeze(-1)
    im = Z.imag.squeeze(-1)
    # real-valued net would output (re', im'). Identity here.
    re_out, im_out = re, im
    Z_out = torch.complex(re_out, im_out).unsqueeze(-1)
    Xb = unproject_complex(Z_out, U)
    assert Xb.shape == X.shape and Xb.dtype == X.dtype


def test_lagged_direction_basic():
    # Construct a simple synthetic series with a real lead/lag relationship
    rng = np.random.default_rng(42)
    T, N = 500, 6
    base = rng.standard_normal((T,))
    X = np.zeros((T, N), dtype=np.float32)
    X[:, 0] = base
    X[:, 1] = np.roll(base, 3)        # 1 lags 0 by 3 steps -> 0 leads 1
    X[:, 2] = np.roll(base, -2)       # 2 leads 0
    X[3:, 3] = base[:-3] + 0.1 * rng.standard_normal(T - 3)
    X[:, 4] = rng.standard_normal((T,))  # uncorrelated
    X[:, 5] = rng.standard_normal((T,))
    A_sym = (np.ones((N, N)) - np.eye(N)).astype(np.float32)
    A_dir = estimate_lagged_direction(X, A_sym, max_lag=5)
    # Pair (0, 1): 0 leads 1 -> A_dir[0,1] should be > A_dir[1,0]
    assert A_dir[0, 1] >= A_dir[1, 0], (A_dir[0, 1], A_dir[1, 0])
    # Pair (0, 2): 2 leads 0 -> A_dir[2,0] >= A_dir[0,2]
    assert A_dir[2, 0] >= A_dir[0, 2]


def test_charge_q_zero_recovers_symmetric_laplacian():
    # When q=0 the magnetic Laplacian (normalized) reduces to the symmetric one.
    A_dir = torch.from_numpy(_random_dir_adj(30, 0.1, seed=7))
    A_dir = 0.5 * (A_dir + A_dir.T)
    L_q0 = build_magnetic_laplacian(A_dir, q=0.0, normalized=True)
    # Build symmetric reference manually
    A_s = 0.5 * (A_dir + A_dir.T)
    N = A_s.shape[0]
    A_s = A_s + torch.eye(N)  # match self-loop default
    deg = A_s.sum(dim=1).clamp(min=1e-8)
    D_iv = torch.diag(1.0 / torch.sqrt(deg))
    L_sym = torch.eye(N) - D_iv @ A_s @ D_iv
    diff = (L_q0.real - L_sym).abs().max().item()
    assert diff < 1e-5, diff
    # And imaginary part should be zero
    assert L_q0.imag.abs().max().item() < 1e-5
