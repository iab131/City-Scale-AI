"""
Bi-axis Mamba residual block.

For each spectral expert we apply two scans:

  1. *Temporal scan* along the T dimension for each spectral mode k.
  2. *Mode scan* along the K spectral-mode dimension for each time step t.

Their outputs are fused with a sigmoid gate, then decoded back to node
space by the calling module. We keep the block real-valued — complex
spectra are split into [real | imag] channels by the caller.

If `mamba_ssm` is installed (preferred) we use selective SSM blocks; we
otherwise fall back to a bidirectional GRU so the block remains trainable
even on machines without the optional dependency (e.g. local dev). The
two fall-backs are kept architecturally identical from outside.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    # Deep import bypasses mamba_ssm's package __init__ which has a brittle
    # transformers-version dependency on some installs. The Mamba layer itself
    # lives in mamba_ssm.modules.mamba_simple and only needs torch + (optional)
    # causal_conv1d.
    from mamba_ssm.modules.mamba_simple import Mamba as _MambaSSM
    _HAS_MAMBA = True
except Exception:  # pragma: no cover - depends on remote env
    try:
        from mamba_ssm import Mamba as _MambaSSM
        _HAS_MAMBA = True
    except Exception:
        _MambaSSM = None
        _HAS_MAMBA = False


class _MambaLike(nn.Module):
    """
    A drop-in replacement for `mamba_ssm.Mamba` used when the official
    selective-scan kernel is unavailable. Implements a small bidirectional
    GRU with the same input/output convention:

        forward(x)  where x: [B, L, D]  -> y: [B, L, D]
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0):
        super().__init__()
        del d_state, d_conv, expand  # kept for API compatibility
        h = max(d_model // 2, 32)
        self.gru = nn.GRU(input_size=d_model, hidden_size=h,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.drop(self.out(h))


def _make_mamba(d_model: int, d_state: int = 16, d_conv: int = 4,
                expand: int = 2, dropout: float = 0.0,
                force_fallback: bool = False) -> nn.Module:
    """
    Returns a real Mamba layer when CUDA is available and `mamba_ssm` is
    installed, otherwise the bidirectional-GRU stand-in. The official
    `selective_scan_cuda` kernel rejects CPU tensors, so CPU usage (unit
    tests, local development) must use the fallback.
    """
    use_mamba = _HAS_MAMBA and torch.cuda.is_available() and not force_fallback
    if use_mamba:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv,
                         expand=expand)
    return _MambaLike(d_model=d_model, dropout=dropout)


class _ScanBlock(nn.Module):
    """
    A pre-norm residual scan block.

        y = x + Drop(Mamba(LN(x)))

    Used for both the temporal and mode scans of the bi-axis module.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _make_mamba(d_model, d_state=d_state, d_conv=d_conv,
                                 expand=expand, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.drop(self.mamba(self.norm(x)))


class BiAxisMambaBlock(nn.Module):
    """
    Bi-axis Mamba operating on a [B, T, K, D] tensor:
      - temporal scan along T for each (k, d)
      - mode scan along K for each (t, d)
      - gated fusion: out = sig(g) * y_t + (1-sig(g)) * y_k + x

    The block is real-valued; complex spectra should be split into real
    and imaginary channels along D before being passed in.
    """

    def __init__(self, d_model: int, n_layers: int = 1,
                 d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1,
                 mode_axis: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode_axis = mode_axis
        self.temporal_layers = nn.ModuleList([
            _ScanBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        if mode_axis:
            self.mode_layers = nn.ModuleList([
                _ScanBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ])
            self.gate = nn.Linear(d_model * 2, d_model)
        # When mode_axis=False the optional sub-modules are absent from the
        # state_dict (rather than being None-attributes) so that checkpoints
        # round-trip cleanly between the two settings.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, K, D] real
        returns y: [B, T, K, D]
        """
        B, T, K, D = x.shape

        # temporal scan: contract (B, K) into batch, scan over T
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * K, T, D)
        for layer in self.temporal_layers:
            xt = layer(xt)
        y_t = xt.view(B, K, T, D).permute(0, 2, 1, 3).contiguous()

        if not self.mode_axis:
            return y_t

        # mode scan: contract (B, T) into batch, scan over K
        xk = x.contiguous().view(B * T, K, D)
        for layer in self.mode_layers:
            xk = layer(xk)
        y_k = xk.view(B, T, K, D)

        gate = torch.sigmoid(self.gate(torch.cat([y_t, y_k], dim=-1)))
        return gate * y_t + (1.0 - gate) * y_k


class TemporalMambaResidual(nn.Module):
    """
    Plain node-space temporal Mamba residual (no graph operation).

    Takes the input window in normalized space and produces a residual forecast
    in normalized space (caller de-normalizes).

    forward(x_norm)  x_norm: [B, T_in, N]   -> delta_y_norm: [B, T_out, N]
    """

    def __init__(self, n_nodes: int, in_steps: int = 12, out_steps: int = 12,
                 d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1,
                 head_init_std: Optional[float] = 1.0e-3):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.d_model = d_model

        self.in_proj = nn.Linear(1, d_model)
        self.tod_embed = nn.Embedding(288, d_model // 4)
        self.dow_embed = nn.Embedding(7, d_model // 4)
        self.feat_proj = nn.Linear(d_model + d_model // 2, d_model)

        self.layers = nn.ModuleList([
            _ScanBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        # Mixed projection over T_in*D -> T_out, per STAEformer's head.
        # head_init_std set (DiSR default 1e-3): residual≈0 at init so the
        # frozen trunk dominates. head_init_std=None: default PyTorch init —
        # needed when this is the primary predictor (SSM-Magma).
        self.head = nn.Linear(in_steps * d_model, out_steps)
        if head_init_std is not None:
            nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
            nn.init.zeros_(self.head.bias)

    def forward(self, x_norm: torch.Tensor, tod: torch.Tensor,
                dow: torch.Tensor) -> torch.Tensor:
        # x_norm: [B, T, N]    tod: [B, T] in [0,1)    dow: [B, T] int
        B, T, N = x_norm.shape
        h = self.in_proj(x_norm.unsqueeze(-1))                     # [B, T, N, D]
        tod_idx = (tod * 288.0).long().clamp(0, 287)
        dow_idx = dow.long()
        tod_e = self.tod_embed(tod_idx).unsqueeze(2).expand(B, T, N, -1)
        dow_e = self.dow_embed(dow_idx).unsqueeze(2).expand(B, T, N, -1)
        h = torch.cat([h, tod_e, dow_e], dim=-1)
        h = self.feat_proj(h)                                       # [B, T, N, D]

        # contract N into batch, scan over T, then collapse T*D into the head.
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        for layer in self.layers:
            h = layer(h)
        h = h.view(B, N, T * self.d_model)
        out = self.head(h)
        return out.transpose(1, 2).contiguous()
