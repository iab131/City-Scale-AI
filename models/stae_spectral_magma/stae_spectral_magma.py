"""
STAE-Spectral-Magma: STAEformer encoder augmented with a three-view bi-axis
spectral Mamba sidechain.

Motivation:
  STAEformer (CIKM 2023) is the strongest reproducible METR-LA backbone we
  have. Its spatial attention is permutation-equivariant — it learns a
  *kernel* over node embeddings without explicit graph structure. We argue
  that injecting three complementary spectral views of the sensor graph
  (symmetric Laplacian, magnetic Laplacian for directed flow, learned-
  semantic kNN basis) into STAEformer's encoder gives the model a graph-
  aware structural signal it cannot recover from attention alone.

Architecture:
              x, tod, dow
                  |
        ┌────────────────┐
        │   STAEformer   │
        │     encoder    │  → h_stae  [B, T_in, N, model_dim]
        └────────────────┘
                  |
         ┌────────┴────────┐
         │                 │
         │   spectral aug  │
         │ ┌──────┬──────┐ │  (1) proj_down: model_dim → d_branch
         │ │ sym  │      │ │  (2) for sym/mag/sem:
         │ │ mag  │ Mamba│ │        z = U^T h_low
         │ │ sem  │      │ │        z = BiAxisMamba(z) (T × K scans)
         │ └──────┴──────┘ │        h_view = U z
         │   horizon       │  (3) blend three h_view via horizon-cluster router
         │   router blend  │  (4) proj_up: d_branch → model_dim
         └─────────────────┘  → h_aug  [B, T_in, N, model_dim]
                  |
              h_stae + h_aug   (residual)
                  |
          STAEformer output_proj
                  |
              y_hat [B, T_out, N]

Why this should work where stand-alone SSM-Magma failed:
  - The STAEformer encoder produces rich per-(sensor, time) features that
    don't depend on a low-rank spectral approximation. This is the
    representational capacity stand-alone SSM-Magma was missing.
  - The spectral sidechain operates on encoder features (not on raw
    speeds), so its job is to *add graph structure*, not to be the sole
    predictor. The bandwidth limit at K modes restricts the augmentation,
    not the base prediction.
  - End-to-end training: encoder, sidechain, and head all learn jointly.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.staeformer import STAEformer
from models.disr.biaxis_mamba import BiAxisMambaBlock
from models.disr.residual_router import HorizonClusterRouter
from models.ssm_magma.semantic_graph import SemanticGraph


# ---------------------------------------------------------------------------
class _SpectralAugView(nn.Module):
    """
    One spectral-view branch: project node-space features → spectral domain
    via the given basis, run a bi-axis Mamba, project back to node space.
    All operations are linear-equivariant in the basis (the basis is fixed
    here for sym/mag and refreshed periodically for sem).
    """

    def __init__(
        self,
        d_branch: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        mode_axis: bool = True,
        is_complex_basis: bool = False,
    ):
        super().__init__()
        self.is_complex = is_complex_basis
        # For complex basis we split (re, im) along feature dim -> 2x d_branch
        d_internal = d_branch * 2 if is_complex_basis else d_branch
        # Map back to d_branch after the scan so the unprojection stays clean.
        if is_complex_basis:
            self.collapse_re_im = nn.Linear(d_internal, d_branch)
        else:
            self.collapse_re_im = None
        self.block = BiAxisMambaBlock(
            d_model=d_internal, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, mode_axis=mode_axis,
        )

    def forward_real(self, h: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """h: [B, T, N, d_branch], U: [N, K] real -> [B, T, N, d_branch]."""
        z = torch.einsum("nk,btne->btke", U, h)                       # [B, T, K, d]
        z = self.block(z)                                              # [B, T, K, d]
        return torch.einsum("nk,btke->btne", U, z)                    # [B, T, N, d]

    def forward_complex(self, h: torch.Tensor, U_real: torch.Tensor,
                        U_imag: torch.Tensor) -> torch.Tensor:
        """
        Real input h, complex basis U = U_real + i U_imag.
        Z = U^H h has complex coefficients; we fold Re/Im along feature dim,
        run real-valued bi-axis Mamba on [B, T, K, 2d], then take Re of U Z.
        """
        # Z = U^H h: zr = U_real^T h, zi = -U_imag^T h
        zr = torch.einsum("nk,btne->btke", U_real, h)
        zi = -torch.einsum("nk,btne->btke", U_imag, h)
        z = torch.cat([zr, zi], dim=-1)                                # [B, T, K, 2d]
        z = self.block(z)                                              # [B, T, K, 2d]
        # Split back into Re/Im for unprojection
        d = z.shape[-1] // 2
        zr_o = z[..., :d]
        zi_o = z[..., d:]
        # X = U Z = (Ur + i Ui)(zr + i zi); take real part
        out = (torch.einsum("nk,btke->btne", U_real, zr_o)
               - torch.einsum("nk,btke->btne", U_imag, zi_o))
        # Collapse the 2d feature back to d to keep downstream shapes clean.
        # We reuse zr_o + zi_o concatenated; for the residual path we just
        # need a single d-channel output, so apply the collapse layer.
        # (Note: `out` itself is already in node space and d-dim; the
        # collapse is only used if a different convention is desired.)
        del z, zr_o, zi_o
        return out


# ---------------------------------------------------------------------------
class SpectralMagmaAugmentation(nn.Module):
    """
    Three-view spectral sidechain: take encoder features h, project through
    each basis (sym/mag/sem), Mamba-scan, project back, blend via router,
    return additive residual.
    """

    def __init__(
        self,
        n_nodes: int,
        in_steps: int,
        out_steps: int,
        model_dim: int,
        d_branch: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        mode_axis: bool = True,
        # bases
        U_sym: Optional[torch.Tensor] = None,
        U_mag: Optional[torch.Tensor] = None,
        d_sem: int = 24,
        k_neighbors: int = 12,
        k_modes_sem: int = 64,
        # view switches
        use_sym: bool = True,
        use_mag: bool = True,
        use_sem: bool = True,
        # router
        use_router: bool = True,
        n_clusters: int = 12,
        cluster_id: Optional[torch.Tensor] = None,
        router_d: int = 32,
        alpha_init: float = 1.0,
        alpha_max: float = 1.5,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.model_dim = model_dim
        self.d_branch = d_branch

        # Down/Up projection to keep the spectral branch cheap.
        self.proj_down = nn.Linear(model_dim, d_branch)
        self.proj_up = nn.Linear(d_branch, model_dim)
        # Initialise proj_up small so the augmentation starts as a near-zero
        # residual — STAEformer dominates at init, the sidechain *adds* to
        # it as training progresses.
        nn.init.normal_(self.proj_up.weight, std=1.0e-3)
        nn.init.zeros_(self.proj_up.bias)

        self.views = nn.ModuleDict()
        self._view_names: List[str] = []

        if use_sym:
            assert U_sym is not None, "use_sym requires U_sym"
            self.register_buffer("U_sym", U_sym.float(), persistent=False)
            self.views["sym"] = _SpectralAugView(
                d_branch=d_branch, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=mode_axis,
                is_complex_basis=False,
            )
            self._view_names.append("sym")

        if use_mag:
            assert U_mag is not None, "use_mag requires U_mag"
            assert torch.is_complex(U_mag), "U_mag must be complex"
            self.register_buffer("U_mag_real", U_mag.real.float(), persistent=False)
            self.register_buffer("U_mag_imag", U_mag.imag.float(), persistent=False)
            self.views["mag"] = _SpectralAugView(
                d_branch=d_branch, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=mode_axis,
                is_complex_basis=True,
            )
            self._view_names.append("mag")

        if use_sem:
            self.semantic_graph = SemanticGraph(
                n_nodes=n_nodes, d_sem=d_sem, k_neighbors=k_neighbors,
                k_modes=k_modes_sem,
            )
            self.views["sem"] = _SpectralAugView(
                d_branch=d_branch, n_layers=n_layers,
                d_state=d_state, d_conv=d_conv, expand=expand,
                dropout=dropout, mode_axis=mode_axis,
                is_complex_basis=False,
            )
            self._view_names.append("sem")
        else:
            self.semantic_graph = None

        # Router blends the view outputs at the node-space residual level.
        if use_router:
            assert cluster_id is not None, "use_router needs cluster_id"
            # The HorizonClusterRouter was designed for output-step routing
            # but it also works at the input-step level when we treat T_in
            # as the "horizon" axis. We give it out_steps=in_steps so the
            # gate shape is [B, T_in, N, n_views].
            self.router = HorizonClusterRouter(
                n_experts=len(self._view_names),
                n_nodes=n_nodes, n_clusters=n_clusters,
                in_steps=in_steps, out_steps=in_steps,
                d_router=router_d,
                alpha_init=alpha_init, alpha_max=alpha_max,
                cluster_id=cluster_id,
            )
            self.alpha_global = None
        else:
            self.router = None
            self.alpha_global = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def view_names(self) -> List[str]:
        return list(self._view_names)

    def forward(
        self,
        h: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        x_recent_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        h: [B, T_in, N, model_dim] from STAEformer encoder
        tod: [B, T_in]   dow: [B, T_in]
        x_recent_raw: [B, T_in, N] raw mph for router context. Defaults to a
            zero tensor when router is on but recent context isn't provided.
        Returns dict with `h_aug` [B, T_in, N, model_dim] (residual to add
        to STAEformer's hidden) and auxiliary tensors for debugging.
        """
        B, T_in, N, D = h.shape

        # Down-project to the cheap branch dim
        h_low = self.proj_down(h)                                      # [B, T, N, d_branch]

        per_view = []
        for name in self._view_names:
            view = self.views[name]
            if name == "sym":
                out_view = view.forward_real(h_low, self.U_sym)
            elif name == "mag":
                out_view = view.forward_complex(h_low, self.U_mag_real,
                                                 self.U_mag_imag)
            elif name == "sem":
                U_sem, _ = self.semantic_graph.get_basis()
                out_view = view.forward_real(h_low, U_sem)
            else:
                raise ValueError(name)
            per_view.append(out_view)                                  # [B, T, N, d_branch]

        # Blend the three views at node-space level (gate over the view axis).
        stacked = torch.stack(per_view, dim=-1)                        # [B, T, N, d_branch, V]
        if self.router is not None:
            if x_recent_raw is None:
                x_recent_raw = torch.zeros(B, T_in, N, device=h.device,
                                           dtype=torch.float32)
            x_recent_norm = x_recent_raw  # router only uses magnitudes
            gate, alpha, aux = self.router(
                tod_in=tod, dow_in=dow,
                x_recent_raw=x_recent_raw, x_recent_norm=x_recent_norm,
            )
            # gate: [B, T_in, N, V]  alpha: [B, T_in, N]
            gate = gate.unsqueeze(-2)                                  # [B, T, N, 1, V]
            mix = (stacked * gate).sum(dim=-1)                         # [B, T, N, d_branch]
            mix = mix * alpha.unsqueeze(-1)
            entropy = aux.get("entropy", torch.zeros((), device=h.device))
        else:
            mix = stacked.mean(dim=-1)                                 # [B, T, N, d_branch]
            mix = mix * self.alpha_global
            gate = None
            entropy = torch.zeros((), device=h.device)

        # Up-project back to model_dim (small init = near-zero residual at start)
        h_aug = self.proj_up(mix)                                      # [B, T, N, model_dim]
        return {
            "h_aug": h_aug,
            "per_view": per_view,
            "gate": gate,
            "entropy": entropy,
        }


# ---------------------------------------------------------------------------
class STAESpectralMagma(nn.Module):
    """STAEformer encoder + three-view spectral Mamba sidechain (residual)."""

    def __init__(
        self,
        N: int = 207,
        in_steps: int = 12,
        out_steps: int = 12,
        # STAEformer hyperparams
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        spatial_embedding_dim: int = 0,
        # spectral sidechain
        d_branch: int = 64,
        spec_n_layers: int = 2,
        spec_d_state: int = 16,
        spec_d_conv: int = 4,
        spec_expand: int = 2,
        spec_dropout: float = 0.1,
        spec_mode_axis: bool = True,
        U_sym: Optional[torch.Tensor] = None,
        U_mag: Optional[torch.Tensor] = None,
        d_sem: int = 24,
        k_neighbors: int = 12,
        k_modes_sem: int = 64,
        use_sym: bool = True,
        use_mag: bool = True,
        use_sem: bool = True,
        use_router: bool = True,
        n_clusters: int = 12,
        cluster_id: Optional[torch.Tensor] = None,
        router_d: int = 32,
        alpha_init: float = 1.0,
        alpha_max: float = 1.5,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps

        self.staeformer = STAEformer(
            N=N, in_steps=in_steps, out_steps=out_steps,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            spatial_embedding_dim=spatial_embedding_dim,
        )

        self.spectral_aug = SpectralMagmaAugmentation(
            n_nodes=N,
            in_steps=in_steps,
            out_steps=out_steps,
            model_dim=self.staeformer.model_dim,
            d_branch=d_branch,
            n_layers=spec_n_layers,
            d_state=spec_d_state,
            d_conv=spec_d_conv,
            expand=spec_expand,
            dropout=spec_dropout,
            mode_axis=spec_mode_axis,
            U_sym=U_sym, U_mag=U_mag,
            d_sem=d_sem, k_neighbors=k_neighbors, k_modes_sem=k_modes_sem,
            use_sym=use_sym, use_mag=use_mag, use_sem=use_sem,
            use_router=use_router, n_clusters=n_clusters,
            cluster_id=cluster_id,
            router_d=router_d,
            alpha_init=alpha_init, alpha_max=alpha_max,
        )

    def forward(self, x_norm: torch.Tensor, tod: torch.Tensor,
                dow: torch.Tensor,
                x_recent_raw: Optional[torch.Tensor] = None,
                return_aux: bool = False) -> torch.Tensor:
        # Encoder
        h = self.staeformer.get_hidden(x_norm, tod, dow)              # [B, T, N, D]
        # Spectral augmentation residual
        aug = self.spectral_aug(h, tod, dow, x_recent_raw=x_recent_raw)
        h_final = h + aug["h_aug"]
        # Mixed-projection head (STAEformer's output_proj)
        B, T_in, N, D = h_final.shape
        out = h_final.transpose(1, 2).reshape(B, N, T_in * D)
        out = self.staeformer.output_proj(out)                        # [B, N, T_out]
        out = out.transpose(1, 2)                                      # [B, T_out, N]
        if return_aux:
            return {"y_pred_norm": out, "gate": aug["gate"],
                    "entropy": aug["entropy"]}
        return out


# ---------------------------------------------------------------------------
def build_stae_spectral_magma(cfg: dict,
                               U_sym: Optional[torch.Tensor] = None,
                               U_mag: Optional[torch.Tensor] = None,
                               cluster_id: Optional[torch.Tensor] = None
                               ) -> STAESpectralMagma:
    """Convenience constructor from a YAML-derived config dict."""
    m = cfg.get("model", {})
    return STAESpectralMagma(
        N=int(cfg["n_nodes"]),
        in_steps=int(cfg.get("in_steps", 12)),
        out_steps=int(cfg.get("out_steps", 12)),
        input_embedding_dim=int(m.get("input_embedding_dim", 24)),
        tod_embedding_dim=int(m.get("tod_embedding_dim", 24)),
        dow_embedding_dim=int(m.get("dow_embedding_dim", 24)),
        adaptive_embedding_dim=int(m.get("adaptive_embedding_dim", 80)),
        feed_forward_dim=int(m.get("feed_forward_dim", 256)),
        num_heads=int(m.get("num_heads", 4)),
        num_layers=int(m.get("num_layers", 3)),
        dropout=float(m.get("dropout", 0.1)),
        spatial_embedding_dim=int(m.get("spatial_embedding_dim", 0)),
        d_branch=int(m.get("d_branch", 64)),
        spec_n_layers=int(m.get("spec_n_layers", 2)),
        spec_d_state=int(m.get("spec_d_state", 16)),
        spec_d_conv=int(m.get("spec_d_conv", 4)),
        spec_expand=int(m.get("spec_expand", 2)),
        spec_dropout=float(m.get("spec_dropout", 0.1)),
        spec_mode_axis=bool(m.get("spec_mode_axis", True)),
        U_sym=U_sym, U_mag=U_mag,
        d_sem=int(m.get("d_sem", 24)),
        k_neighbors=int(m.get("k_neighbors", 12)),
        k_modes_sem=int(m.get("k_modes_sem", 64)),
        use_sym=bool(m.get("use_sym", True)),
        use_mag=bool(m.get("use_mag", True)),
        use_sem=bool(m.get("use_sem", True)),
        use_router=bool(m.get("use_router", True)),
        n_clusters=int(m.get("n_clusters", 12)),
        cluster_id=cluster_id,
        router_d=int(m.get("router_d", 32)),
        alpha_init=float(m.get("alpha_init", 1.0)),
        alpha_max=float(m.get("alpha_max", 1.5)),
    )
