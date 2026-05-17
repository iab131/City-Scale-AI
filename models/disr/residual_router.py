"""
Horizon-cluster residual router for DiSR-Mamba.

Low-capacity gating that produces per-(sample, horizon, cluster) weights
across residual experts plus a per-(horizon, cluster) residual scale alpha.

Inputs (all optional, gated by config flags):
  - horizon index embedding         h ∈ {0..T_out-1}
  - sensor cluster embedding        c ∈ {0..n_clusters-1}
  - time-of-day embedding           tod-bin
  - day-of-week embedding           dow
  - recent volatility summary       std over T_in for each (sample, cluster)
  - congestion bin summary          fraction of recent speeds < threshold

Restrictions enforced by construction:
  - parameter count grows linearly in {T_out + n_clusters + n_experts}
    rather than in B*N (no per-(sample,sensor) MoE).
  - output weights are softmax-normalised across experts.
  - the router does NOT modify the trunk; it only mixes Delta_Y from experts.
"""

from __future__ import annotations
from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sensor_clusters(
    adj: np.ndarray,
    n_clusters: int = 12,
    random_state: int = 0,
    X_train: Optional[np.ndarray] = None,
    use_history: bool = True,
) -> np.ndarray:
    """
    Build a [N] integer cluster assignment from a graph adjacency, optionally
    refined by a historical-correlation kernel computed on `X_train`.

    Uses spectral clustering on a normalized affinity matrix
        S = 0.5 * (A_norm + Corr) if X_train and use_history else A_norm
    where A_norm is the symmetric normalized adjacency.

    Returns a numpy int array of shape [N].
    """
    from sklearn.cluster import KMeans

    A = np.asarray(adj, dtype=np.float64)
    A = 0.5 * (A + A.T)
    N = A.shape[0]
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1) + 1e-9
    A_norm = A / np.sqrt(np.outer(deg, deg))

    if X_train is not None and use_history:
        # Pearson correlation of valid history per sensor (impute missing with mean)
        Xc = np.where(X_train == 0.0, np.nan, X_train)
        col_mean = np.nanmean(Xc, axis=0)
        Xc = np.where(np.isnan(Xc), col_mean[None, :], Xc)
        Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-6)
        Corr = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        np.fill_diagonal(Corr, 0.0)
        Corr = np.clip(Corr, 0.0, 1.0)
        S = 0.5 * A_norm + 0.5 * Corr
    else:
        S = A_norm

    # Build the symmetric normalized Laplacian and take its bottom-k eigenvectors
    L = np.eye(N) - S
    L = 0.5 * (L + L.T)
    evals, V = np.linalg.eigh(L)
    feats = V[:, :max(n_clusters, 4)]
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(feats)
    return labels.astype(np.int64)


class HorizonClusterRouter(nn.Module):
    """
    Produces (sample, horizon, cluster, expert) mixing weights and a global
    alpha scale per (horizon, cluster).

    The router is intentionally tiny. With defaults (d_router=32, T_out=12,
    n_clusters=12, n_experts=3) it has < 5k parameters.

    forward inputs
    --------------
    tod_in:     [B, T_in]        time-of-day in [0,1)
    dow_in:     [B, T_in]        day-of-week int
    x_norm:     [B, T_in, N]     input (normalised) speeds (for volatility)
    x_recent:   [B, T_in, N]     input raw speeds (for congestion summary)
    cluster_id: [N] long         per-sensor cluster id (registered as buffer)

    forward outputs
    ---------------
    expert_weights: [B, T_out, N, n_experts]  softmax over experts
    alpha:          [B, T_out, N]              residual scale in (0, alpha_max)
    aux:            dict with router entropy (scalar) and gate stats
    """

    def __init__(
        self,
        n_experts: int,
        n_nodes: int,
        n_clusters: int = 12,
        in_steps: int = 12,
        out_steps: int = 12,
        d_router: int = 32,
        steps_per_day: int = 288,
        n_dow: int = 7,
        alpha_init: float = 0.1,
        alpha_max: float = 1.5,
        congestion_thr: float = 25.0,
        cluster_id: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.alpha_max = alpha_max
        self.congestion_thr = congestion_thr

        # Buffers / embeddings -----------------------------------------------
        if cluster_id is None:
            cluster_id = torch.zeros(n_nodes, dtype=torch.long)
        else:
            cluster_id = cluster_id.long().clone()
        self.register_buffer("cluster_id", cluster_id, persistent=True)

        self.h_emb = nn.Embedding(out_steps, d_router)
        self.c_emb = nn.Embedding(n_clusters, d_router)
        self.tod_emb = nn.Embedding(steps_per_day, d_router // 2)
        self.dow_emb = nn.Embedding(n_dow, d_router // 4)

        # context summary features: [mean_recent_speed, std_recent_speed, congestion_frac]
        # built per-(sample, cluster) inside forward, then projected to d_router
        n_ctx = 3
        self.ctx_proj = nn.Linear(n_ctx, d_router)

        # Final gate MLP: combines all signals, outputs per-expert logits + alpha logit.
        gate_in = d_router  # all heads sum into one bag for low-capacity routing
        self.mlp = nn.Sequential(
            nn.Linear(gate_in, d_router),
            nn.SiLU(),
            nn.Linear(d_router, n_experts + 1),
        )
        # Initialize alpha bias such that sigmoid(alpha_bias)*alpha_max ≈ alpha_init
        with torch.no_grad():
            # sigmoid^-1(alpha_init / alpha_max)
            p = max(min(alpha_init / max(alpha_max, 1e-6), 1.0 - 1e-3), 1e-3)
            self.mlp[-1].bias[-1].fill_(math.log(p / (1.0 - p)))

    # ------------------------------------------------------------------
    @staticmethod
    def _scatter_mean_by_cluster(values: torch.Tensor,
                                 cluster_id: torch.Tensor,
                                 n_clusters: int) -> torch.Tensor:
        """
        values: [B, N]  cluster_id: [N]  -> out: [B, n_clusters]
        Mean over each cluster's sensors.
        """
        B, N = values.shape
        out = values.new_zeros(B, n_clusters)
        cnt = values.new_zeros(B, n_clusters)
        cid = cluster_id.unsqueeze(0).expand(B, N)
        out.scatter_add_(1, cid, values)
        ones = torch.ones_like(values)
        cnt.scatter_add_(1, cid, ones)
        return out / cnt.clamp(min=1.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        tod_in: torch.Tensor,
        dow_in: torch.Tensor,
        x_recent_raw: torch.Tensor,
        x_recent_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        B = tod_in.shape[0]
        N = self.n_nodes
        T_in = self.in_steps
        T_out = self.out_steps
        K = self.n_clusters
        device = tod_in.device

        # ---- cluster-level recent-context summary ---------------------
        # per-(sample, sensor) summaries
        mean_speed = x_recent_raw.mean(dim=1)                           # [B, N]
        std_speed = x_recent_raw.std(dim=1).clamp(min=0.0)              # [B, N]
        cong_frac = (x_recent_raw < self.congestion_thr).float().mean(dim=1)  # [B, N]

        # aggregate to clusters
        cid = self.cluster_id
        mean_c = self._scatter_mean_by_cluster(mean_speed, cid, K)      # [B, K]
        std_c  = self._scatter_mean_by_cluster(std_speed,  cid, K)
        cong_c = self._scatter_mean_by_cluster(cong_frac,  cid, K)
        ctx = torch.stack([mean_c, std_c, cong_c], dim=-1)              # [B, K, 3]
        ctx_emb = self.ctx_proj(ctx)                                    # [B, K, d]

        # ---- time-of-day / dow embeddings (window-aggregated) ---------
        # use last in-window step as the routing query
        tod_idx = (tod_in[:, -1] * 288.0).long().clamp(0, 287)
        dow_idx = dow_in[:, -1].long().clamp(0, 6)
        tod_e = self.tod_emb(tod_idx)                                   # [B, d/2]
        dow_e = self.dow_emb(dow_idx)                                   # [B, d/4]
        # pad to full d
        d = self.h_emb.embedding_dim
        cal = torch.cat([tod_e, dow_e], dim=-1)                         # [B, d/2 + d/4]
        cal = F.pad(cal, (0, d - cal.shape[-1]))                        # [B, d]

        # ---- horizon and cluster embeddings ---------------------------
        h_e = self.h_emb(torch.arange(T_out, device=device))            # [T_out, d]
        c_e = self.c_emb(torch.arange(K, device=device))                # [K, d]

        # ---- broadcast all into [B, T_out, K, d] and fuse -------------
        feat = (
            h_e.view(1, T_out, 1, d)
            + c_e.view(1, 1, K, d)
            + cal.view(B, 1, 1, d)
            + ctx_emb.view(B, 1, K, d)
        )

        logits = self.mlp(feat)                                         # [B, T_out, K, E+1]
        expert_logits = logits[..., :-1]                                # [B, T_out, K, E]
        alpha_logit  = logits[..., -1]                                  # [B, T_out, K]

        gate = F.softmax(expert_logits, dim=-1)                         # [B, T_out, K, E]
        alpha = torch.sigmoid(alpha_logit) * self.alpha_max             # [B, T_out, K]

        # Scatter from clusters back to sensors:
        # gather using cluster_id ∈ [0..K)  -> sensor-level gate
        gate_sensor = gate.index_select(2, cid)                         # [B, T_out, N, E]
        alpha_sensor = alpha.index_select(2, cid)                       # [B, T_out, N]

        # entropy regularization signal (mean entropy of mixing weights)
        eps = 1e-8
        entropy = -(gate * (gate + eps).log()).sum(dim=-1).mean()

        aux = {
            "entropy": entropy,
            "mean_alpha": alpha_sensor.mean().detach(),
            "max_gate": gate_sensor.max(dim=-1).values.mean().detach(),
        }
        return gate_sensor, alpha_sensor, aux
