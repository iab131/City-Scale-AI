"""
Magnetic Laplacian spectral basis on a directed graph.

Given a directed adjacency A_dir we build the Hermitian magnetic Laplacian

    A_s     = 0.5 * (A_dir + A_dir.T)
    Theta_q = 2 * pi * q * (A_dir - A_dir.T)
    H_q     = A_s * exp(1j * Theta_q)        # element-wise
    D_s     = diag(sum(A_s, dim=-1))
    L_q     = D_s - H_q

L_q is Hermitian by construction. Its eigenvectors are complex and we keep
the K lowest-eigenvalue modes.

When the only adjacency available is the undirected distance matrix used by
METR-LA, we infer a pseudo-direction A_dir from lagged cross-correlation
between sensors that share an edge in the original graph: sensor i is
treated as upstream of j if lagged corr(x_i, x_j shifted by +tau) exceeds
the reverse for the small tau > 0 we test. The magnitude is kept from the
symmetric (distance-based) support.

All functions accept torch tensors (CPU is fine for this small problem).
"""

from __future__ import annotations
from typing import Optional, Tuple

import math
import numpy as np
import torch


def _normalize_signal(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Center and scale each sensor column to unit variance."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + eps
    return (x - mu) / sd


def estimate_lagged_direction(
    X_train: np.ndarray,
    A_sym: np.ndarray,
    max_lag: int = 6,
    missing_val: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Build a directed adjacency from a symmetric one by checking whether i
    *leads* j over short lags.

    Args
    ----
    X_train: [T, N] training-only speeds (raw, may contain missing zeros).
    A_sym:   [N, N] symmetric adjacency. Pairs with weight 0 are skipped.
    max_lag: largest positive shift tested.

    Returns
    -------
    A_dir: [N, N] non-negative directed adjacency. Magnitude = |A_sym[i, j]|.
           Direction is i -> j if i leads j; both directions kept only if the
           leading-correlation evidence is roughly symmetric (defined here as
           |c_fwd - c_rev| < 0.05 * max(|c_fwd|, |c_rev|, eps)).
    """
    assert X_train.ndim == 2, X_train.shape
    T, N = X_train.shape
    assert A_sym.shape == (N, N), A_sym.shape

    # Pre-compute a missing mask so missing entries don't dominate correlations.
    valid = (X_train != missing_val).astype(np.float32)
    # zero-fill so we can shift cheaply
    Xz = np.where(valid > 0, X_train, X_train.mean()).astype(np.float64)
    Xz = _normalize_signal(Xz)

    A_dir = np.zeros_like(A_sym, dtype=np.float64)
    nz_i, nz_j = np.where(A_sym > 0)
    pairs = set()
    for i, j in zip(nz_i, nz_j):
        if i == j:
            continue
        pairs.add((min(i, j), max(i, j)))

    # For each unordered pair, compute c(i -> j) = max_{tau>=1} corr(X[:-tau, i], X[tau:, j])
    # and the reverse, then assign weight accordingly.
    for (a, b) in pairs:
        c_ab_best = 0.0
        c_ba_best = 0.0
        for tau in range(1, max_lag + 1):
            xa = Xz[:-tau, a]
            xb_lag = Xz[tau:, b]
            xb = Xz[:-tau, b]
            xa_lag = Xz[tau:, a]
            # zero-mean already, divide by length for normalized correlation
            denom = max(len(xa) - 1, 1)
            c_ab = float((xa * xb_lag).sum() / denom)
            c_ba = float((xb * xa_lag).sum() / denom)
            c_ab_best = max(c_ab_best, c_ab)
            c_ba_best = max(c_ba_best, c_ba)
        w_sym = abs(float(A_sym[a, b])) + abs(float(A_sym[b, a]))
        # Decision rule: significant asymmetry => single direction.
        rel = abs(c_ab_best - c_ba_best) / max(abs(c_ab_best), abs(c_ba_best), eps)
        if rel < 0.05:
            # mutual / undirected — keep both halves of the symmetric weight
            A_dir[a, b] = 0.5 * w_sym
            A_dir[b, a] = 0.5 * w_sym
        elif c_ab_best > c_ba_best:
            A_dir[a, b] = w_sym
        else:
            A_dir[b, a] = w_sym
    return A_dir.astype(np.float32)


def build_magnetic_laplacian(
    A_dir: torch.Tensor,
    q: float = 0.1,
    add_self_loop: bool = True,
    eps: float = 1e-8,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Build a Hermitian magnetic Laplacian (complex64) from a directed
    adjacency.

    Args
    ----
    A_dir: [N, N] real, non-negative directed adjacency tensor.
    q:     charge parameter in (0, 0.5).
    normalized: if True, returns the normalized variant
                L_q = I - D_s^{-1/2} H_q D_s^{-1/2} (still Hermitian).
                If False, returns the un-normalized L_q = D_s - H_q.

    Returns
    -------
    L_q: complex64 [N, N] Hermitian.
    """
    if isinstance(A_dir, np.ndarray):
        A_dir = torch.from_numpy(A_dir).float()
    A_dir = A_dir.to(torch.float32)
    N = A_dir.shape[0]
    if add_self_loop:
        A_dir = A_dir + torch.eye(N, device=A_dir.device, dtype=A_dir.dtype)
    A_s = 0.5 * (A_dir + A_dir.T)
    Theta = 2.0 * math.pi * q * (A_dir - A_dir.T)
    # H_q = A_s ⊙ exp(i Theta)
    H_q = torch.complex(A_s * torch.cos(Theta), A_s * torch.sin(Theta))
    if not normalized:
        D_s = torch.diag(A_s.sum(dim=1))
        L_q = D_s.to(torch.complex64) - H_q
    else:
        deg = A_s.sum(dim=1).clamp(min=eps)
        d_inv_sqrt = 1.0 / torch.sqrt(deg)
        D_iv = torch.diag(d_inv_sqrt).to(torch.complex64)
        L_q = torch.eye(N, dtype=torch.complex64, device=A_dir.device) - D_iv @ H_q @ D_iv
    # Force exact Hermitian to avoid numerical issues in eigh.
    L_q = 0.5 * (L_q + L_q.conj().T)
    return L_q


def eigendecompose_hermitian(
    L: torch.Tensor,
    k: int,
    side: str = "low",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sorted eigendecomposition of a Hermitian matrix.

    Returns
    -------
    eigvals: real [k], ascending within the selected side.
    U:       complex [N, k] columns are eigenvectors.
    """
    L = L.to(torch.complex64)
    evals, V = torch.linalg.eigh(L)
    # eigh returns ascending. Pick K modes from the requested end.
    N = L.shape[0]
    if side == "low":
        idx = torch.arange(k, device=L.device)
    elif side == "high":
        idx = torch.arange(N - k, N, device=L.device)
    elif side == "both":
        k_lo = k // 2
        k_hi = k - k_lo
        idx = torch.cat([torch.arange(k_lo, device=L.device),
                         torch.arange(N - k_hi, N, device=L.device)])
    else:
        raise ValueError(f"unknown side {side}")
    return evals.real[idx].to(torch.float32), V[:, idx]


def magnetic_basis_from_adjacency(
    A_sym: np.ndarray,
    X_train: Optional[np.ndarray],
    k: int,
    q: float,
    side: str = "low",
    max_lag: int = 6,
    cache_path: Optional[str] = None,
    a_dir_override: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience builder. If `X_train` is provided and `a_dir_override` is None,
    estimate the directed adjacency via lagged cross-correlation; otherwise
    use the override. Returns eigvals (real [k]) and U (complex64 [N, k]).
    Caches both to disk if cache_path is given.
    """
    import os
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        evals = torch.from_numpy(z["evals"]).float()
        U_real = torch.from_numpy(z["U_real"]).float()
        U_imag = torch.from_numpy(z["U_imag"]).float()
        U = torch.complex(U_real, U_imag)
        return evals, U

    if a_dir_override is not None:
        A_dir = a_dir_override.astype(np.float32)
    else:
        assert X_train is not None, "either X_train or a_dir_override must be given"
        A_dir = estimate_lagged_direction(X_train, A_sym, max_lag=max_lag)

    A_dir_t = torch.from_numpy(A_dir).float()
    L_q = build_magnetic_laplacian(A_dir_t, q=q, normalized=True)
    evals, U = eigendecompose_hermitian(L_q, k=k, side=side)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            evals=evals.cpu().numpy(),
            U_real=U.real.cpu().numpy(),
            U_imag=U.imag.cpu().numpy(),
            A_dir=A_dir,
            q=np.float32(q),
        )
    return evals, U


def project_complex(x_node: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Project real node-space signal into the complex eigenbasis.

    x_node: [..., N, C]   real
    U:      [N, K]        complex
    Returns Z: [..., K, C] complex.

    Uses Z = U^H @ X_node. (U^H is the conjugate transpose.)
    """
    Uh = U.conj().transpose(-2, -1)  # [K, N] complex
    x_complex = x_node.to(Uh.dtype)
    return torch.einsum("kn,...nc->...kc", Uh, x_complex)


def unproject_complex(z_spec: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Inverse projection. Returns real node-space signal (we take the real part
    after multiplying back by U).
    """
    out = torch.einsum("nk,...kc->...nc", U, z_spec)
    return out.real
