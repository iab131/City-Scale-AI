"""
Learned semantic graph over METR-LA sensors. Each sensor has a learnable
embedding; a symmetric top-k cosine-similarity kNN graph is derived from
those embeddings on each forward pass (cheap — 207x207, ~ms).

A normalized Laplacian of this graph is exposed via a stable projection
operator (using its top-K eigenvectors) so the rest of the model can plug it
into the same spectral pipeline as the symmetric / magnetic experts. To
avoid eigendecomposing every step, the eigenbasis is computed at module
init and refreshed every `refresh_steps` training steps. At inference we
recompute once and cache.
"""
from __future__ import annotations
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGraph(nn.Module):
    """
    Learnable sensor embedding + symmetric top-k kNN Laplacian.

    Parameters
    ----------
    n_nodes: number of sensors (207 for METR-LA).
    d_sem:   embedding dimension (16-32 is plenty).
    k_neighbors: top-K nearest neighbours per node (8-16). Larger K -> denser
        graph, but pure kNN ignores edges below the cutoff, so K controls
        sparsity.
    k_modes: number of eigenmodes to expose for the spectral pipeline.
    refresh_steps: how often the eigenbasis is recomputed during training.
        Set to a fraction of an epoch so the basis is roughly stable within
        any single optimisation phase but tracks slow embedding drift.
    """

    def __init__(self, n_nodes: int, d_sem: int = 24, k_neighbors: int = 12,
                 k_modes: int = 48, refresh_steps: int = 200):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_sem = d_sem
        self.k_neighbors = k_neighbors
        self.k_modes = k_modes
        self.refresh_steps = refresh_steps

        self.embed = nn.Embedding(n_nodes, d_sem)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Cached eigenbasis (refreshed periodically)
        self.register_buffer("U", torch.empty(0))
        self.register_buffer("evals", torch.empty(0))
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

    # ----------------------------------------------------------------
    def _knn_laplacian(self) -> torch.Tensor:
        """Return the symmetric normalized Laplacian of the current kNN graph."""
        e = F.normalize(self.embed.weight, p=2, dim=-1)            # [N, d]
        sim = e @ e.T                                              # [N, N]
        # zero out self-similarity for kNN selection
        sim = sim - torch.eye(self.n_nodes, device=sim.device) * 2.0
        # top-k per row (directed nearest neighbours)
        K = min(self.k_neighbors, self.n_nodes - 1)
        top_vals, top_idx = sim.topk(K, dim=-1)
        # Build a sparse-style dense adjacency: only top-k entries set.
        adj = torch.zeros_like(sim)
        adj.scatter_(1, top_idx, top_vals.relu())                  # non-negative weights
        # Symmetrize (so an edge exists if either direction is a top-K nbr)
        adj = torch.maximum(adj, adj.T)
        # Add self-loop
        adj = adj + torch.eye(self.n_nodes, device=adj.device)
        # Normalised Laplacian L = I - D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1).clamp(min=1e-6)
        d_inv_sqrt = 1.0 / deg.sqrt()
        D = torch.diag(d_inv_sqrt)
        A_norm = D @ adj @ D
        L = torch.eye(self.n_nodes, device=adj.device) - A_norm
        # Force exact symmetry
        L = 0.5 * (L + L.T)
        return L

    # ----------------------------------------------------------------
    def _compute_basis(self):
        L = self._knn_laplacian()
        # eigh works fine on a 207x207 matrix on GPU.
        evals, V = torch.linalg.eigh(L)
        # Take K lowest eigenvalues (smooth modes).
        K = self.k_modes
        idx = torch.arange(K, device=L.device)
        self.U = V[:, idx].contiguous()
        self.evals = evals[idx].contiguous()

    # ----------------------------------------------------------------
    def maybe_refresh(self, force: bool = False) -> None:
        if force or self.U.numel() == 0 or (
            self.training
            and (int(self.step_counter.item()) % self.refresh_steps == 0)
        ):
            self._compute_basis()
        if self.training:
            self.step_counter += 1

    # ----------------------------------------------------------------
    def get_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (U, evals). U: [N, K] real. evals: [K] real."""
        self.maybe_refresh()
        return self.U, self.evals

    # ----------------------------------------------------------------
    def ortho_penalty(self) -> torch.Tensor:
        """||U^T U - I||_F^2 — encourages near-orthonormal columns even though
        eigh already returns orthonormal columns, to keep the basis from
        rotating between refreshes."""
        if self.U.numel() == 0:
            self._compute_basis()
        UTU = self.U.T @ self.U
        I = torch.eye(UTU.shape[0], device=UTU.device, dtype=UTU.dtype)
        return (UTU - I).pow(2).sum()
