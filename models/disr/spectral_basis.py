"""
Symmetric Laplacian spectral basis utilities.

Builds the standard graph Fourier basis from a symmetric (undirected)
adjacency. Provides projection/unprojection helpers that batch the FFT-style
operation over the (B, T, C) leading axes while keeping the eigenvectors
on the spatial axis.

Tensors:
  X_node ~ [..., N, C]    node-space signal
  Z      ~ [..., K, C]    spectral-space coefficients (real-valued)
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch


def normalized_laplacian_from_adj(A: np.ndarray, add_self_loop: bool = True,
                                  eps: float = 1e-8) -> np.ndarray:
    """
    Build the symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.

    A: dense [N, N] non-negative adjacency, symmetrized internally.
    """
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + A.T)
    if add_self_loop:
        np.fill_diagonal(A, A.diagonal() + 1.0)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    # Force exact symmetry to avoid eig instability
    L_sym = 0.5 * (L_sym + L_sym.T)
    return L_sym


def build_symmetric_basis(
    A: np.ndarray,
    k: int,
    side: str = "low",
    add_self_loop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the top/bottom K eigenpairs of the normalized symmetric Laplacian.

    Args
    ----
    A: [N, N] dense symmetric adjacency.
    k: number of eigenpairs to keep.
    side: "low" for smallest-eigenvalue modes (smooth signals);
          "high" for largest-eigenvalue modes (noise/high-frequency);
          "both" returns top-k/2 from each end concatenated.

    Returns
    -------
    eigvals: [k] sorted ascending within the kept side.
    U:       [N, k] orthonormal eigenvectors as columns.
    """
    L = normalized_laplacian_from_adj(A, add_self_loop=add_self_loop)
    evals_full, U_full = np.linalg.eigh(L)
    order = np.argsort(evals_full)
    evals_full = evals_full[order]
    U_full = U_full[:, order]

    N = L.shape[0]
    if side == "low":
        idx = np.arange(k)
    elif side == "high":
        idx = np.arange(N - k, N)
    elif side == "both":
        k_lo = k // 2
        k_hi = k - k_lo
        idx = np.concatenate([np.arange(k_lo), np.arange(N - k_hi, N)])
    else:
        raise ValueError(f"unknown side {side}")
    return evals_full[idx].astype(np.float32), U_full[:, idx].astype(np.float32)


def project(x_node: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Project a node-space signal into the spectral basis.

    x_node: [..., N, C]
    U:      [N, K] real eigenvectors
    returns Z: [..., K, C]
    """
    # einsum is robust to arbitrary leading dims.
    return torch.einsum("nk,...nc->...kc", U, x_node)


def unproject(z_spec: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Inverse projection: take spectral coefficients back to node space.

    z_spec: [..., K, C]
    U:      [N, K] real eigenvectors
    returns X: [..., N, C]
    """
    return torch.einsum("nk,...kc->...nc", U, z_spec)


def load_or_build_symmetric_basis(
    A: np.ndarray,
    k: int,
    side: str,
    cache_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience that caches eigenpairs to a .npz on disk.

    Returns torch float32 tensors (eigvals[k], U[N,k]).
    """
    import os
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        evals, U = z["evals"], z["U"]
    else:
        evals, U = build_symmetric_basis(A, k=k, side=side)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, evals=evals, U=U)
    return torch.from_numpy(evals).float(), torch.from_numpy(U).float()
