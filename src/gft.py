import numpy as np
from scipy.sparse.linalg import eigsh


def compute_gft_basis(L, k: int):
    """
    Compute the first k Laplacian eigenvectors/eigenvalues.

    Args:
        L: sparse Laplacian matrix [N, N]
        k: number of eigenvectors

    Returns:
        evals: [k]
        evecs: [N, k]
    """
    # Smallest algebraic eigenvalues -> smoothest graph frequencies
    evals, evecs = eigsh(L, k=k, which="SA")
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals.astype(np.float32), evecs.astype(np.float32)


def gft(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Graph Fourier Transform.

    Args:
        x: [T, N] or [N]
        U: [N, k]

    Returns:
        x_hat: [T, k] or [k]
    """
    if x.ndim == 1:
        return U.T @ x
    return x @ U


def igft(x_hat: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Inverse Graph Fourier Transform (truncated reconstruction).

    Args:
        x_hat: [T, k] or [k]
        U: [N, k]

    Returns:
        x_rec: [T, N] or [N]
    """
    if x_hat.ndim == 1:
        return U @ x_hat
    return x_hat @ U.T