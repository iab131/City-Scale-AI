"""
Hybrid STAEformer + Spectral Mamba.

STAEformer's Transformer encoder handles node-space attention well, but
ignores global graph structure. Our bi-axis spectral Mamba captures
city-wide spectral modes (low-frequency Laplacian eigenvectors) that are
hard to recover via local attention.

Design: parallel branches, fused before the output head.

  x_norm, tod, dow
       │
       ├─── STAEformer encoder ────────────────► h_stae [B, T_in, N, 152]
       │
       └─── Spectral Mamba encoder ────────────► h_spec [B, T_in, N, d_s]
                                                  │
                                                  │  inverse GFT applied per
                                                  │  channel: maps [K, d] -> [N, d]
                                                  │  so output is in NODE space
                                                  ▼
                                              concat([h_stae, h_spec], dim=-1)
                                                       │
                                                       ▼
                                          Linear(T_in * (152+d_s), T_out)
                                                       │
                                                       ▼
                                                y_hat [B, T_out, N]

Why this should help:
  - Local attention (STAEformer) + global spectral (our Mamba) are
    complementary signals. Concatenating before the final flat projection
    gives the readout full access to both.
  - The hybrid is initialized so that the spectral branch starts at zero
    contribution (zero output_proj weights on the spectral channels) — so
    the model at init behaves identically to STAEformer and only grows the
    spectral contribution if it actually helps val loss.

Total params: ~1.26M (STAEformer) + ~0.3M (spec Mamba) = ~1.55M.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from mamba_ssm import Mamba

from models.staeformer import STAEformer
from models.spectral_ssm import BiAxisMambaBlock, ChebFilter


class SpectralBranch(nn.Module):
    """
    Compact spectral Mamba branch:
      x_norm -> GFT projection -> Cheb filter -> embed -> 2 bi-axis Mamba
      blocks -> per-mode embedding -> inverse GFT -> [B, T, N, d_s]
    Output is in NODE SPACE.
    """

    def __init__(self, U: torch.Tensor, evals: torch.Tensor,
                 d_s: int = 32, num_layers: int = 2,
                 cheb_order: int = 3, cheb_channels: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        N, K = U.shape
        self.N = N; self.K = K; self.d_s = d_s
        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)
        self.channel_proj = nn.Linear(cheb_channels, d_s)

        self.mode_emb = nn.Parameter(torch.zeros(K, d_s))
        nn.init.normal_(self.mode_emb, std=0.02)

        self.blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_s, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_s)

    def forward(self, x_norm):
        """
        x_norm: [B, T, N]
        returns: [B, T, N, d_s] in NODE space
        """
        B, T, N = x_norm.shape
        # GFT
        x_hat = x_norm @ self.U                                       # [B, T, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)  # [B, T, K, C]
        h = self.channel_proj(x_filt) + self.mode_emb.view(1, 1, self.K, self.d_s)
        # Bi-axis Mamba on [B, T, K, d_s]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)                                              # [B, T, K, d_s]
        # Inverse GFT: contract over K -> N
        # h_node[b, t, n, d] = sum_k h[b, t, k, d] * U[n, k]
        h_node = torch.einsum("btkd,nk->btnd", h, self.U)             # [B, T, N, d_s]
        return h_node


class HybridSTAEMamba(nn.Module):
    """
    STAEformer + Spectral Mamba parallel branches, fused before output proj.
    """

    def __init__(
        self,
        N: int,
        U: torch.Tensor,
        evals: torch.Tensor,
        # STAEformer args
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        num_dow: int = 7,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        # Spectral branch args
        spec_d: int = 32,
        spec_layers: int = 2,
        spec_cheb_order: int = 3,
        spec_cheb_channels: int = 4,
        # Init scheme
        zero_init_spec_head: bool = True,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.spec_d = spec_d

        # STAEformer encoder (no output proj — we use our own)
        self.stae = STAEformer(
            N=N,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            num_dow=num_dow,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        # We'll bypass stae.output_proj and use our own hybrid head.
        self.stae_model_dim = self.stae.model_dim

        # Spectral branch
        self.spec = SpectralBranch(
            U=U, evals=evals, d_s=spec_d, num_layers=spec_layers,
            cheb_order=spec_cheb_order, cheb_channels=spec_cheb_channels,
            dropout=dropout,
        )

        # Hybrid output projection
        fused_dim = self.stae_model_dim + spec_d
        self.output_proj = nn.Linear(in_steps * fused_dim, out_steps)

        # Zero-init the spectral branch's contribution to the output so the
        # model at init equals STAEformer. The STAEformer side keeps its
        # original Xavier-uniform Linear init.
        if zero_init_spec_head:
            with torch.no_grad():
                # Output proj weight shape: [out_steps, in_steps * fused_dim].
                # The spectral channels occupy the LAST spec_d columns of each
                # in_steps slot — zero those columns out.
                W = self.output_proj.weight                            # [out_steps, in_steps*fused_dim]
                # Reshape to [out_steps, in_steps, fused_dim] for clarity:
                W3 = W.view(out_steps, in_steps, fused_dim)
                W3[:, :, self.stae_model_dim:].zero_()
                # bias is fine (default ~uniform); no need to zero

    def forward(self, x_norm, tod, dow, y_tod=None, y_dow=None):
        """
        x_norm: [B, T_in, N]
        Returns y_hat_norm: [B, T_out, N]
        """
        B, T_in, N = x_norm.shape

        # STAEformer encoder (gives hidden state before output proj)
        h_stae = self.stae.get_hidden(x_norm, tod, dow)               # [B, T_in, N, 152]

        # Spectral branch
        h_spec = self.spec(x_norm)                                    # [B, T_in, N, spec_d]

        # Fuse
        h = torch.cat([h_stae, h_spec], dim=-1)                       # [B, T_in, N, 152+spec_d]

        # Mixed projection: [B, N, T_in * fused_dim] -> Linear -> [B, N, T_out]
        fused_dim = self.stae_model_dim + self.spec_d
        out = h.transpose(1, 2).reshape(B, N, T_in * fused_dim)
        out = self.output_proj(out)                                   # [B, N, T_out]
        out = out.transpose(1, 2)                                     # [B, T_out, N]
        return out


def build_hybrid(N: int, U: torch.Tensor, evals: torch.Tensor, **kwargs) -> HybridSTAEMamba:
    return HybridSTAEMamba(N=N, U=U, evals=evals, **kwargs)
