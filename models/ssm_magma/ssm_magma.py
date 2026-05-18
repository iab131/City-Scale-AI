"""
SSM-Magma — three spectral experts (physical-symmetric, physical-magnetic,
semantic-kNN) blended by a low-capacity horizon-cluster router. End-to-end
trained on masked MAE; no frozen trunk.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Reuse the existing pieces from the DiSR-Mamba implementation.
from models.disr.biaxis_mamba import BiAxisMambaBlock
from models.disr.disr_mamba import (
    SymmetricSpectralExpert,
    MagneticSpectralExpert,
)
from models.disr.residual_router import HorizonClusterRouter
from .semantic_graph import SemanticGraph


# ---------------------------------------------------------------------------
class SemanticSpectralExpert(nn.Module):
    """
    Uses the kNN-derived spectral basis from `SemanticGraph`. Architecture
    mirrors `SymmetricSpectralExpert` from models.disr.disr_mamba: project
    node-space signal into the K-mode basis, bi-axis Mamba over (T, K),
    project back to node space. The basis is REFRESHED periodically (see
    SemanticGraph.refresh_steps); both `U` and the embedding are trained
    end-to-end.
    """

    def __init__(self, n_nodes: int, semantic_graph: SemanticGraph,
                 in_steps: int = 12, out_steps: int = 12,
                 d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1, mode_axis: bool = True,
                 head_init_std: Optional[float] = None,
                 d_adp_emb: int = 0):
        # forward-declared to keep the signature symmetric with the base
        # spectral experts in models/disr/disr_mamba.py
        super().__init__()
        self.n_nodes = n_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.d_model = d_model
        self.semantic_graph = semantic_graph
        self.d_adp_emb = d_adp_emb

        self.in_proj = nn.Linear(1, d_model)
        self.tod_embed = nn.Embedding(288, d_model // 4)
        self.dow_embed = nn.Embedding(7, d_model // 4)
        self.feat_proj = nn.Linear(d_model + d_model // 2, d_model)

        if d_adp_emb > 0:
            self.adp_proj = nn.Linear(d_adp_emb, d_model)
        else:
            self.adp_proj = None

        self.block = BiAxisMambaBlock(
            d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, mode_axis=mode_axis,
        )
        self.head = nn.Linear(in_steps * d_model, out_steps)
        if head_init_std is not None:
            nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
            nn.init.zeros_(self.head.bias)

    def forward(self, x_norm: torch.Tensor, tod: torch.Tensor,
                dow: torch.Tensor,
                adaptive_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T_in, N = x_norm.shape
        U, _ = self.semantic_graph.get_basis()                  # [N, K]
        # project: Z = U^T X
        z = torch.einsum("nk,btn->btk", U, x_norm)              # [B, T_in, K]
        K = z.shape[-1]

        h = self.in_proj(z.unsqueeze(-1))                       # [B, T_in, K, d]
        tod_idx = (tod * 288.0).long().clamp(0, 287)
        dow_idx = dow.long().clamp(0, 6)
        tod_e = self.tod_embed(tod_idx).unsqueeze(2).expand(B, T_in, K, -1)
        dow_e = self.dow_embed(dow_idx).unsqueeze(2).expand(B, T_in, K, -1)
        h = torch.cat([h, tod_e, dow_e], dim=-1)
        h = self.feat_proj(h)                                   # [B, T_in, K, d]

        # Adaptive-embedding side channel through the (refreshed) learned
        # basis. Same shape contract as the sym/mag experts:
        # adaptive_emb is [B, T_in, N, d_adp_emb], TOD-indexed by SSMMagma.
        if adaptive_emb is not None and self.adp_proj is not None:
            adp_spec = torch.einsum("nk,btne->btke", U, adaptive_emb)  # [B, T_in, K, d_adp]
            adp_feat = self.adp_proj(adp_spec)                          # [B, T_in, K, d_model]
            h = h + adp_feat

        h = self.block(h)                                       # [B, T_in, K, d]

        h = h.permute(0, 2, 1, 3).contiguous().view(B, K, T_in * self.d_model)
        out = self.head(h)                                      # [B, K, T_out]
        out = out.transpose(1, 2).contiguous()                  # [B, T_out, K]
        # unproject back to node space
        y = torch.einsum("nk,btk->btn", U, out)
        return y


# ---------------------------------------------------------------------------
class SSMMagma(nn.Module):
    """
    Standalone forecaster: three spectral experts + horizon-cluster router.

    forward(x_norm, tod, dow)
      x_norm: [B, T_in, N] float (normalized speed)
      tod:    [B, T_in] float in [0, 1)
      dow:    [B, T_in] long in {0..6}
    Returns y_pred_norm: [B, T_out, N].
    """

    def __init__(
        self,
        n_nodes: int = 207,
        in_steps: int = 12,
        out_steps: int = 12,
        # spectral bases (passed in pre-computed for physical experts)
        U_sym: Optional[torch.Tensor] = None,        # [N, K] real
        U_mag: Optional[torch.Tensor] = None,        # [N, K] complex
        # semantic graph
        d_sem: int = 24,
        k_neighbors: int = 12,
        # expert hyperparams
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.10,
        biaxis_mode_axis: bool = True,
        k_modes: int = 48,
        # routing
        use_router: bool = True,
        n_clusters: int = 12,
        cluster_id: Optional[torch.Tensor] = None,
        alpha_init: float = 1.0,
        alpha_max: float = 1.5,
        router_d: int = 32,
        # expert switches (for ablation)
        use_sym: bool = True,
        use_mag: bool = True,
        use_sem: bool = True,
        # mean prediction injection: predict y as base mean + residual scale
        use_node_bias: bool = True,
        # SSM-Magma is a standalone forecaster, NOT a frozen-trunk residual,
        # so expert heads default to PyTorch's standard init. The DiSR-Mamba
        # σ=1e-3 default would clip the experts to near-zero output and lock
        # the model into pure-persistence mode (verified empirically).
        head_init_std: Optional[float] = None,
        # Persistent shortcut and per-(sensor, horizon) bias act as the
        # baseline the experts deviate from. Both default ON but become
        # ablation knobs.
        use_persistent_shortcut: bool = True,
        # Adaptive embedding: a [T_in, N, d_adp_emb] learnable tensor that
        # carries per-(time, sensor) identity. Each expert projects it
        # through its own basis and adds it to the per-mode features.
        # This is the analogue of STAEformer's adaptive embedding, the
        # single biggest source of model expressiveness on METR-LA.
        d_adp_emb: int = 24,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps

        self.experts = nn.ModuleDict()
        self._expert_names: List[str] = []

        # STAEformer-style adaptive embedding indexed by *time-of-day bin*
        # (not by window-position). Shape: [steps_per_day, N, d_adp_emb].
        # At forward time we index by the input window's tod_idx so the same
        # absolute time-of-day always gets the same per-sensor embedding,
        # regardless of where it appears within the input window. This is
        # what gives STAEformer its representational power on METR-LA, and
        # what fixes the position-indexed-overfitting we observed earlier.
        self.d_adp_emb = d_adp_emb
        self.steps_per_day = 288
        if d_adp_emb > 0:
            self.adaptive_emb = nn.Parameter(
                0.02 * torch.randn(self.steps_per_day, n_nodes, d_adp_emb)
            )
        else:
            self.adaptive_emb = None

        if use_sym:
            assert U_sym is not None, "use_sym requires U_sym"
            self.experts["sym_spectral"] = SymmetricSpectralExpert(
                n_nodes=n_nodes, U_sym=U_sym,
                in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=biaxis_mode_axis,
                head_init_std=head_init_std,
                d_adp_emb=d_adp_emb,
            )
            self._expert_names.append("sym_spectral")

        if use_mag:
            assert U_mag is not None, "use_mag requires U_mag"
            self.experts["mag_spectral"] = MagneticSpectralExpert(
                n_nodes=n_nodes, U_mag=U_mag,
                in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=biaxis_mode_axis,
                head_init_std=head_init_std,
                d_adp_emb=d_adp_emb,
            )
            self._expert_names.append("mag_spectral")

        if use_sem:
            self.semantic_graph = SemanticGraph(
                n_nodes=n_nodes, d_sem=d_sem, k_neighbors=k_neighbors,
                k_modes=k_modes,
            )
            self.experts["semantic_spectral"] = SemanticSpectralExpert(
                n_nodes=n_nodes, semantic_graph=self.semantic_graph,
                in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=biaxis_mode_axis,
                head_init_std=head_init_std,
                d_adp_emb=d_adp_emb,
            )
            self._expert_names.append("semantic_spectral")
        else:
            self.semantic_graph = None

        # Per-sensor learnable bias (helps a lot when the spectral residuals
        # alone are small at init). Predicts a per-(sensor, horizon) offset
        # in normalized space.
        if use_node_bias:
            self.node_horizon_bias = nn.Parameter(
                torch.zeros(out_steps, n_nodes)
            )
        else:
            self.node_horizon_bias = None

        # Persistent shortcut: each sample's mean-x prediction (last in-step
        # broadcast to all out-steps) anchors the model to the trunk-like
        # "no change" baseline. The experts learn the *delta* from this in
        # a single-step regression (analogous to STAEformer's residual but
        # *trained end-to-end* not as a frozen-trunk add-on).
        self.use_persistent_shortcut = use_persistent_shortcut

        # router
        if use_router:
            assert cluster_id is not None, "use_router needs cluster_id"
            self.router = HorizonClusterRouter(
                n_experts=len(self._expert_names),
                n_nodes=n_nodes, n_clusters=n_clusters,
                in_steps=in_steps, out_steps=out_steps,
                d_router=router_d,
                alpha_init=alpha_init, alpha_max=alpha_max,
                cluster_id=cluster_id,
            )
            self.alpha_global = None
        else:
            self.router = None
            self.alpha_global = nn.Parameter(torch.tensor(float(alpha_init)))

    # ------------------------------------------------------------------
    @property
    def expert_names(self) -> List[str]:
        return list(self._expert_names)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_norm: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        x_recent_raw: Optional[torch.Tensor] = None,
        return_experts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns either a tensor [B, T_out, N] (default) or, if
        `return_experts=True`, a dict {y_pred, per_expert, gate, alpha, aux}.
        """
        # Index the TOD-conditioned adaptive embedding once per forward.
        # tod is in [0,1); convert to 5-min bins in [0, steps_per_day).
        if self.adaptive_emb is not None:
            tod_idx = (tod * self.steps_per_day).long().clamp(
                0, self.steps_per_day - 1)                              # [B, T_in]
            adp_batch = self.adaptive_emb[tod_idx]                      # [B, T_in, N, d_adp]
        else:
            adp_batch = None

        per_expert = []
        for name in self._expert_names:
            per_expert.append(
                self.experts[name](x_norm, tod, dow,
                                   adaptive_emb=adp_batch)
            )

        E = len(per_expert)
        stacked = torch.stack(per_expert, dim=-1)                # [B, T_out, N, E]

        if self.router is not None:
            assert x_recent_raw is not None, "router needs x_recent_raw"
            gate, alpha, aux = self.router(
                tod_in=tod, dow_in=dow,
                x_recent_raw=x_recent_raw, x_recent_norm=x_norm,
            )
            mix = (stacked * gate).sum(dim=-1)                   # [B, T_out, N]
            delta = alpha * mix
            entropy = aux.get("entropy", torch.tensor(0.0,
                                                       device=x_norm.device))
        else:
            mix = stacked.mean(dim=-1)
            alpha = self.alpha_global.expand_as(mix)
            delta = alpha * mix
            gate = None
            entropy = torch.tensor(0.0, device=x_norm.device)

        # persistent shortcut: last in-window step broadcast across out-steps
        # forms a "no change" baseline (in normalized space).
        if self.use_persistent_shortcut:
            persist = x_norm[:, -1:, :].expand(-1, self.out_steps, -1)
        else:
            persist = torch.zeros_like(delta)

        bias = (self.node_horizon_bias.unsqueeze(0)
                if self.node_horizon_bias is not None else 0.0)
        y = persist + delta + bias

        if return_experts:
            return {
                "y_pred_norm": y,
                "per_expert": per_expert,
                "gate": gate,
                "alpha": alpha,
                "entropy": entropy,
            }
        return y


def build_ssm_magma(cfg: dict,
                     U_sym: Optional[torch.Tensor] = None,
                     U_mag: Optional[torch.Tensor] = None,
                     cluster_id: Optional[torch.Tensor] = None) -> SSMMagma:
    """Convenience constructor from a YAML-derived config dict."""
    m = cfg.get("model", {})
    return SSMMagma(
        n_nodes=int(cfg["n_nodes"]),
        in_steps=int(cfg.get("in_steps", 12)),
        out_steps=int(cfg.get("out_steps", 12)),
        U_sym=U_sym, U_mag=U_mag,
        d_sem=int(m.get("d_sem", 24)),
        k_neighbors=int(m.get("k_neighbors", 12)),
        d_model=int(m.get("d_model", 64)),
        n_layers=int(m.get("n_layers", 2)),
        d_state=int(m.get("d_state", 16)),
        d_conv=int(m.get("d_conv", 4)),
        expand=int(m.get("expand", 2)),
        dropout=float(m.get("dropout", 0.10)),
        biaxis_mode_axis=bool(m.get("biaxis_mode_axis", True)),
        k_modes=int(m.get("k_modes", 48)),
        use_router=bool(m.get("use_router", True)),
        n_clusters=int(m.get("n_clusters", 12)),
        cluster_id=cluster_id,
        alpha_init=float(m.get("alpha_init", 1.0)),
        alpha_max=float(m.get("alpha_max", 1.5)),
        router_d=int(m.get("router_d", 32)),
        use_sym=bool(m.get("use_sym", True)),
        use_mag=bool(m.get("use_mag", True)),
        use_sem=bool(m.get("use_sem", True)),
        use_node_bias=bool(m.get("use_node_bias", True)),
        use_persistent_shortcut=bool(m.get("use_persistent_shortcut", True)),
        head_init_std=m.get("head_init_std", None),
        d_adp_emb=int(m.get("d_adp_emb", 24)),
    )
