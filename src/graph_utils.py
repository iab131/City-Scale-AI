import numpy as np
import scipy.sparse as sp


def symmetrize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Make adjacency symmetric for spectral decomposition.
    """
    return np.maximum(A, A.T)


def normalized_laplacian(A: np.ndarray) -> sp.csr_matrix:
    """
    Compute symmetric normalized Laplacian:
        L = I - D^{-1/2} A D^{-1/2}
    """
    A = symmetrize_adjacency(A)
    A = sp.csr_matrix(A)

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-12)

    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    I = sp.eye(A.shape[0], format="csr")
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L