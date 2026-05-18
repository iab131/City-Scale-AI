"""
DiSR-Mamba: the residual model that sits on top of a frozen STAEformer trunk.

Composes independently switchable experts:

    1. TemporalMambaResidual      (node-space, no graph operation)
    2. SymmetricSpectralExpert    (sym. Laplacian + bi-axis Mamba)
    3. MagneticSpectralExpert     (magnetic Laplacian + bi-axis Mamba)
    4. LearnedBasisExpert         (optional, learned projection)

Each expert outputs Delta_Y_norm ∈ [B, T_out, N]. The final model returns:

    Y_pred_norm = Y_base_norm + alpha * Delta_Y_norm

(`alpha` is either a global learnable scalar or per-(horizon,cluster) from
the router.) De-normalization is handled by the trainer.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .biaxis_mamba import BiAxisMambaBlock, TemporalMambaResidual
from .residual_router import HorizonClusterRouter


# ---------------------------------------------------------------------------
# Spectral experts
# ---------------------------------------------------------------------------
class _SpectralExpertBase(nn.Module):
    """
    Shared spectral-expert backbone:
      Z = U^* X    [B, T_in, K, C_in]
      Z' = BiAxisMamba(Z)
      X_back = U Z'  -> Linear -> [B, T_out, N]
    Real- and complex-basis variants subclass and override `_project` /
    `_unproject` and the channel layout.
    """

    def __init__(
        self,
        n_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        mode_axis: bool = True,
        input_channels: int = 1,
        head_init_std: Optional[float] = 1.0e-3,
        d_adp_emb: int = 0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.d_model = d_model
        self.input_channels = input_channels
        self.d_adp_emb = d_adp_emb

        self.in_proj = nn.Linear(input_channels, d_model)
        self.tod_embed = nn.Embedding(288, d_model // 4)
        self.dow_embed = nn.Embedding(7, d_model // 4)
        self.feat_proj = nn.Linear(d_model + d_model // 2, d_model)

        # Adaptive embedding projector: takes [..., d_adp_eff] per-mode
        # adaptive features (after spectral projection) and lifts them to
        # d_model so they can be added to `h`. d_adp_eff = d_adp_emb for
        # real bases (sym/sem), 2*d_adp_emb for the complex magnetic basis
        # (each adp channel projects to Re|Im).
        if d_adp_emb > 0:
            d_adp_eff = d_adp_emb * input_channels
            self.adp_proj = nn.Linear(d_adp_eff, d_model)
        else:
            self.adp_proj = None

        self.block = BiAxisMambaBlock(
            d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, mode_axis=mode_axis,
        )
        # Head: per spectral mode, map T_in*D -> T_out*output_channels.
        # With `head_init_std` set (DiSR default = 1e-3), the initial output
        # is near-zero so a frozen trunk dominates while gradients still
        # flow into the scan layers. With `head_init_std=None`, we keep
        # PyTorch's default Linear init — needed for SSM-Magma where the
        # expert IS the predictor, not a residual.
        self.head = nn.Linear(in_steps * d_model, out_steps * input_channels)
        if head_init_std is not None:
            nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
            nn.init.zeros_(self.head.bias)

    # --- subclasses override ---------------------------------------------
    def _project(self, x_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _unproject(self, z_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _project_adp(self, adp: torch.Tensor) -> torch.Tensor:
        """Project a batched adaptive embedding [B, T_in, N, d_adp] through
        this expert's spectral basis to [B, T_in, K, d_adp_eff]. Subclass
        override when the basis is complex (magnetic)."""
        raise NotImplementedError
    # ---------------------------------------------------------------------

    def forward(self, x_norm: torch.Tensor, tod: torch.Tensor,
                dow: torch.Tensor,
                adaptive_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_norm: [B, T_in, N]   tod: [B, T_in]   dow: [B, T_in]
        adaptive_emb: optional [T_in, N, d_adp_emb] learnable per-(time, sensor)
            features projected through the expert's basis and added to h.
        returns Delta_Y_norm: [B, T_out, N]
        """
        B, T_in, N = x_norm.shape
        # 1) project to spectral
        # x_norm with channel dim: [B, T_in, N, C_in]
        x = x_norm.unsqueeze(-1)
        z = self._project(x)                                            # [B, T_in, K, C_in_spec]

        # 2) per-mode features:  embed input + time-of-day + day-of-week
        K = z.shape[2]
        h = self.in_proj(z)                                             # [B, T_in, K, d]
        tod_idx = (tod * 288.0).long().clamp(0, 287)
        dow_idx = dow.long().clamp(0, 6)
        tod_e = self.tod_embed(tod_idx).unsqueeze(2).expand(B, T_in, K, -1)
        dow_e = self.dow_embed(dow_idx).unsqueeze(2).expand(B, T_in, K, -1)
        h = torch.cat([h, tod_e, dow_e], dim=-1)
        h = self.feat_proj(h)                                           # [B, T_in, K, d]

        # 2b) optional adaptive-embedding side channel projected through
        # this expert's basis (this is the "STAEformer-class memory" hook).
        if adaptive_emb is not None and self.adp_proj is not None:
            adp_spec = self._project_adp(adaptive_emb)                  # [B, T_in, K, d_adp_eff]
            adp_feat = self.adp_proj(adp_spec)                          # [B, T_in, K, d_model]
            h = h + adp_feat

        # 3) bi-axis Mamba over (T, K)
        h = self.block(h)                                               # [B, T_in, K, d]

        # head over time: collapse T_in*d into a flat axis once
        h = h.permute(0, 2, 1, 3).contiguous().view(B, K, T_in * self.d_model)
        out = self.head(h).view(B, K, self.out_steps, self.input_channels)
        out = out.permute(0, 2, 1, 3).contiguous()

        # 5) project back to node space
        delta = self._unproject(out)                                    # [B, T_out, N, C_in]
        return delta.squeeze(-1)                                        # [B, T_out, N]


class SymmetricSpectralExpert(_SpectralExpertBase):
    """Symmetric Laplacian basis (real-valued)."""

    def __init__(self, n_nodes: int, U_sym: torch.Tensor,
                 head_init_std: Optional[float] = 1.0e-3,
                 d_adp_emb: int = 0, **kwargs):
        """
        U_sym: [N, K] real eigenvectors, registered as a non-trainable buffer.
        """
        super().__init__(n_nodes=n_nodes, input_channels=1,
                         head_init_std=head_init_std,
                         d_adp_emb=d_adp_emb, **kwargs)
        assert U_sym.dim() == 2 and U_sym.shape[0] == n_nodes
        self.register_buffer("U_sym", U_sym.float(), persistent=False)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]   U_sym: [N, K]
        return torch.einsum("nk,btnc->btkc", self.U_sym, x)

    def _unproject(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nk,btkc->btnc", self.U_sym, z)

    def _project_adp(self, adp: torch.Tensor) -> torch.Tensor:
        # adp: [B, T_in, N, d_adp] -> [B, T_in, K, d_adp]
        return torch.einsum("nk,btne->btke", self.U_sym, adp)


class MagneticSpectralExpert(_SpectralExpertBase):
    """
    Magnetic Laplacian basis: complex eigenvectors. We split each complex
    spectral coefficient into (real, imag) channels and feed them as
    input_channels=2 to a real-valued bi-axis Mamba.
    """

    def __init__(self, n_nodes: int, U_mag: torch.Tensor,
                 head_init_std: Optional[float] = 1.0e-3,
                 d_adp_emb: int = 0, **kwargs):
        """
        U_mag: complex [N, K]
        """
        super().__init__(n_nodes=n_nodes, input_channels=2,
                         head_init_std=head_init_std,
                         d_adp_emb=d_adp_emb, **kwargs)
        assert torch.is_complex(U_mag) and U_mag.shape[0] == n_nodes
        # Store real / imag halves as float buffers
        self.register_buffer("U_real", U_mag.real.contiguous().float(),
                             persistent=False)
        self.register_buffer("U_imag", U_mag.imag.contiguous().float(),
                             persistent=False)

    # ---- helpers --------------------------------------------------------
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # x: real [B, T, N, 1]  ->  Z = U^H X  ->  [B, T, K, 1] complex
        # Split into real/imag channels => [B, T, K, 2]
        # U^H = (U.conj()).T  =>  (re = U_real.T @ x,  im = -U_imag.T @ x)
        Ur = self.U_real  # [N, K]
        Ui = self.U_imag  # [N, K]
        # x squeezed in channel for the projection
        x_ = x.squeeze(-1)                                              # [B, T, N]
        zr = torch.einsum("nk,btn->btk", Ur, x_)                        # [B, T, K]
        zi = -torch.einsum("nk,btn->btk", Ui, x_)                       # [B, T, K]
        return torch.stack([zr, zi], dim=-1)                            # [B, T, K, 2]

    def _unproject(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, K, 2] real/imag  -> X = U Z  -> real part to node space
        zr = z[..., 0]                                                  # [B, T, K]
        zi = z[..., 1]                                                  # [B, T, K]
        Ur = self.U_real
        Ui = self.U_imag
        # (Ur + i Ui)(zr + i zi) = (Ur*zr - Ui*zi) + i(Ur*zi + Ui*zr)
        out_r = torch.einsum("nk,btk->btn", Ur, zr) - torch.einsum("nk,btk->btn", Ui, zi)
        return out_r.unsqueeze(-1)                                      # [B, T, N, 1]

    def _project_adp(self, adp: torch.Tensor) -> torch.Tensor:
        # adp: real [B, T_in, N, d_adp]. Complex projection Z = U^H @ adp.
        # U^H = (U_real - i U_imag)^T  =>  (zr =  U_real^T @ adp,
        #                                   zi = -U_imag^T @ adp).
        # Split [Re | Im] into 2 channels along the d_adp axis.
        zr = torch.einsum("nk,btne->btke", self.U_real, adp)
        zi = -torch.einsum("nk,btne->btke", self.U_imag, adp)
        return torch.cat([zr, zi], dim=-1)                              # [B, T_in, K, 2*d_adp]


class LearnedBasisExpert(_SpectralExpertBase):
    """
    Learned orthogonal-ish basis: a trainable [N, K] matrix, optionally
    initialised from the symmetric basis. Regularised toward orthonormality
    by the trainer if `ortho_reg > 0`.
    """

    def __init__(self, n_nodes: int, k: int, init_from: Optional[torch.Tensor] = None,
                 head_init_std: Optional[float] = 1.0e-3, **kwargs):
        super().__init__(n_nodes=n_nodes, input_channels=1,
                         head_init_std=head_init_std, **kwargs)
        if init_from is None:
            U0 = torch.randn(n_nodes, k) / math.sqrt(n_nodes)
        else:
            U0 = init_from.float()
        self.U_learn = nn.Parameter(U0)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nk,btnc->btkc", self.U_learn, x)

    def _unproject(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nk,btkc->btnc", self.U_learn, z)

    def _project_adp(self, adp: torch.Tensor) -> torch.Tensor:
        # adp: [B, T_in, N, d_adp] -> [B, T_in, K, d_adp]
        return torch.einsum("nk,btne->btke", self.U_learn, adp)

    def ortho_penalty(self) -> torch.Tensor:
        # ||U^T U - I||_F^2
        UTU = self.U_learn.T @ self.U_learn
        I = torch.eye(UTU.shape[0], device=UTU.device, dtype=UTU.dtype)
        return (UTU - I).pow(2).sum()


# ---------------------------------------------------------------------------
# DiSR-Mamba composite model
# ---------------------------------------------------------------------------
class DiSRMamba(nn.Module):
    """
    Composite residual model. Each expert independently produces a normalized
    Delta_Y_norm; the router (or uniform averaging) mixes them.

    Inputs at forward time:
      x_norm:     [B, T_in, N]
      tod:        [B, T_in]
      dow:        [B, T_in]
      x_recent:   [B, T_in, N]    raw mph (for router and congestion features)

    Returns dict:
      delta_y_norm:  [B, T_out, N]  combined residual in normalised space
      per_expert:    list[[B, T_out, N]]
      alpha:         [B, T_out, N] effective alpha scale
      aux:           {entropy, mean_alpha, ortho_penalty?}
    """

    def __init__(
        self,
        n_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        # expert switches
        use_temporal_residual: bool = True,
        use_symmetric_spectral: bool = True,
        use_magnetic_spectral: bool = True,
        use_learned_basis: bool = False,
        # expert hyperparams
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        biaxis_mode_axis: bool = True,
        # spectral bases
        U_sym: Optional[torch.Tensor] = None,
        U_mag: Optional[torch.Tensor] = None,
        learned_k: int = 48,
        learned_init_from: Optional[torch.Tensor] = None,
        # routing
        use_horizon_cluster_router: bool = False,
        n_clusters: int = 12,
        cluster_id: Optional[torch.Tensor] = None,
        alpha_init: float = 0.1,
        alpha_max: float = 1.5,
        router_d: int = 32,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps

        self._expert_names: List[str] = []
        self.experts = nn.ModuleDict()

        if use_temporal_residual:
            self.experts["temporal"] = TemporalMambaResidual(
                n_nodes=n_nodes, in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout,
            )
            self._expert_names.append("temporal")

        if use_symmetric_spectral:
            assert U_sym is not None, "use_symmetric_spectral requires U_sym"
            self.experts["sym_spectral"] = SymmetricSpectralExpert(
                n_nodes=n_nodes, U_sym=U_sym,
                in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=biaxis_mode_axis,
            )
            self._expert_names.append("sym_spectral")

        if use_magnetic_spectral:
            assert U_mag is not None, "use_magnetic_spectral requires U_mag"
            self.experts["mag_spectral"] = MagneticSpectralExpert(
                n_nodes=n_nodes, U_mag=U_mag,
                in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=biaxis_mode_axis,
            )
            self._expert_names.append("mag_spectral")

        if use_learned_basis:
            self.experts["learned"] = LearnedBasisExpert(
                n_nodes=n_nodes, k=learned_k, init_from=learned_init_from,
                in_steps=in_steps, out_steps=out_steps,
                d_model=d_model, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=biaxis_mode_axis,
            )
            self._expert_names.append("learned")

        if len(self._expert_names) == 0:
            raise ValueError("DiSR-Mamba must include at least one expert")

        # ---- alpha / router --------------------------------------------
        if use_horizon_cluster_router:
            assert cluster_id is not None, "router needs cluster_id"
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
            # learnable global scalar (initialised near alpha_init)
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
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          delta_y_norm: [B, T_out, N]   weighted residual in normalised space
          per_expert:   list[[B, T_out, N]]
          alpha:        [B, T_out, N] or scalar tensor
          aux:          dict (entropy etc.)
        """
        per_expert = []
        for name in self._expert_names:
            d = self.experts[name](x_norm, tod, dow)
            per_expert.append(d)

        E = len(per_expert)
        if self.router is None:
            # uniform mixing
            stacked = torch.stack(per_expert, dim=-1)                   # [B, T_out, N, E]
            mix = stacked.mean(dim=-1)                                  # [B, T_out, N]
            alpha = self.alpha_global.expand_as(mix) if self.alpha_global is not None else None
            delta = alpha * mix
            aux = {"mean_alpha": alpha.mean().detach() if alpha is not None
                   else torch.tensor(0.0, device=x_norm.device)}
            return {"delta_y_norm": delta,
                    "per_expert": per_expert,
                    "alpha": alpha,
                    "aux": aux,
                    "expert_weights": None}
        else:
            assert x_recent_raw is not None, "router requires x_recent_raw"
            gate, alpha, aux = self.router(tod_in=tod, dow_in=dow,
                                           x_recent_raw=x_recent_raw,
                                           x_recent_norm=x_norm)
            # gate: [B, T_out, N, E]  alpha: [B, T_out, N]
            stacked = torch.stack(per_expert, dim=-1)                   # [B, T_out, N, E]
            mix = (stacked * gate).sum(dim=-1)                          # [B, T_out, N]
            delta = alpha * mix
            return {"delta_y_norm": delta,
                    "per_expert": per_expert,
                    "alpha": alpha,
                    "aux": aux,
                    "expert_weights": gate}


# ---------------------------------------------------------------------------
def build_disr_from_config(cfg: dict,
                            U_sym: Optional[torch.Tensor] = None,
                            U_mag: Optional[torch.Tensor] = None,
                            cluster_id: Optional[torch.Tensor] = None,
                            learned_init_from: Optional[torch.Tensor] = None
                            ) -> DiSRMamba:
    """
    Convenience builder. `cfg` is a dict from YAML; required keys are
    `n_nodes`, `in_steps`, `out_steps`. Expert flags default to False.
    """
    flags = cfg.get("model", {})
    return DiSRMamba(
        n_nodes=int(cfg["n_nodes"]),
        in_steps=int(cfg.get("in_steps", 12)),
        out_steps=int(cfg.get("out_steps", 12)),
        use_temporal_residual=bool(flags.get("use_temporal_residual", True)),
        use_symmetric_spectral=bool(flags.get("use_symmetric_spectral", False)),
        use_magnetic_spectral=bool(flags.get("use_magnetic_spectral", False)),
        use_learned_basis=bool(flags.get("use_learned_basis", False)),
        d_model=int(flags.get("d_model", 64)),
        n_layers=int(flags.get("n_layers", 2)),
        d_state=int(flags.get("d_state", 16)),
        d_conv=int(flags.get("d_conv", 4)),
        expand=int(flags.get("expand", 2)),
        dropout=float(flags.get("dropout", 0.1)),
        biaxis_mode_axis=bool(flags.get("biaxis_mode_axis", True)),
        U_sym=U_sym, U_mag=U_mag,
        learned_k=int(flags.get("learned_k", 48)),
        learned_init_from=learned_init_from,
        use_horizon_cluster_router=bool(flags.get("use_horizon_cluster_router", False)),
        n_clusters=int(flags.get("n_clusters", 12)),
        cluster_id=cluster_id,
        alpha_init=float(flags.get("alpha_init", 0.1)),
        alpha_max=float(flags.get("alpha_max", 1.5)),
        router_d=int(flags.get("router_d", 32)),
    )
