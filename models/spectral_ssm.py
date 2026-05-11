"""
Spectral State Space Model (SSSM) for METR-LA traffic forecasting.

This is the novel core of the project. Standard "GFT + temporal Mamba"
pipelines (StemGNN, the existing SpectralMambaReal in this repo, etc.)
apply a sequence model along the time axis only - the K spectral modes
are processed as independent channels with no mixing. The model can never
ask "given the city-wide rush-hour mode is active right now, should I
update the localized-congestion modes differently?" - because the modes
never talk to each other.

We fix this with a **bi-axis selective scan**: each block alternates a
Mamba that scans along TIME (per spectral mode) with a Mamba that scans
along MODES (per time step). The mode-axis Mamba is the genuine novel
contribution; the natural ordering of the modes - smallest eigenvalue
first - gives the scan a meaningful "frequency" direction analogous to
how Mamba treats word order in language.

Additional ingredients used by all SOTA METR-LA models that we add here:
  - learnable Chebyshev filter that adaptively reweights the fixed GFT basis
  - learnable spectral-mode embeddings (per-mode positional code)
  - time-of-day / day-of-week embeddings
  - per-horizon decoder that reads the full bi-axis output sequence
  - prediction directly in node space via a *fixed* inverse-GFT readout
    combined with a learned node-space residual head

Inputs (per batch):
  x_norm  [B, T_in, N]    masked-normalized speeds
  tod     [B, T_in]       time-of-day in [0,1)
  dow     [B, T_in]       day-of-week int

Static (registered as buffers, set at construction time):
  U       [N, K]          eigenvectors (smallest-algebraic first)
  evals   [K]             eigenvalues (rescaled into [0,2])

Output:
  y_hat   [B, T_out, N]   node-space predictions (in NORMALIZED space)
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------


class ChebFilter(nn.Module):
    """
    Learnable Chebyshev polynomial filter on the (precomputed) GFT eigenvalues.

    For each input channel c, a polynomial g_c(lam) = sum_p theta_{c,p} T_p(lam)
    is learned. Applied as a per-mode multiplicative gate on the spectral coefs.

    This keeps the fixed Laplacian basis (per the team's "GFT is good" constraint)
    but lets the model adaptively REWEIGHT modes - an empirically large lever in
    every spectral GNN that competes with adaptive-adjacency models.

    Args:
        K (int):     number of spectral modes retained
        order (int): polynomial order, default 3
        channels:    number of independent filters; output is summed across them
    """

    def __init__(self, K: int, order: int = 3, channels: int = 4):
        super().__init__()
        self.K = K
        self.order = order
        self.channels = channels
        # theta[c, p]
        self.theta = nn.Parameter(torch.zeros(channels, order + 1))
        nn.init.normal_(self.theta, std=0.05)
        # encourage the filter to be ~identity at init: first poly coeff = 1
        with torch.no_grad():
            self.theta[:, 0] = 1.0

    def forward(self, evals_scaled: torch.Tensor) -> torch.Tensor:
        """
        evals_scaled: [K]  eigenvalues mapped to [-1, 1]
        returns:      [channels, K] gain per mode per channel
        """
        # Chebyshev recurrence: T_0 = 1, T_1 = x, T_{p+1} = 2 x T_p - T_{p-1}
        Tprev = torch.ones_like(evals_scaled)
        Tcur = evals_scaled
        out = self.theta[:, 0:1] * Tprev.unsqueeze(0)  # [C, K]
        if self.order >= 1:
            out = out + self.theta[:, 1:2] * Tcur.unsqueeze(0)
        for p in range(2, self.order + 1):
            Tnext = 2 * evals_scaled * Tcur - Tprev
            out = out + self.theta[:, p:p + 1] * Tnext.unsqueeze(0)
            Tprev, Tcur = Tcur, Tnext
        return out  # [C, K]


class BiAxisMambaBlock(nn.Module):
    """
    One block of the bi-axis Mamba stack. Operates on tensors of shape
    [B, T, K, D] and produces tensors of the same shape, with two residual
    branches:
      - time-axis Mamba scans along T independently for each (batch, mode)
      - mode-axis Mamba scans along K independently for each (batch, time)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.time_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mode_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, K, D]
        B, T, K, D = x.shape

        # ---- time axis: fold (B, K) -> batch, scan over T ----
        xt = self.norm_t(x)
        xt = xt.permute(0, 2, 1, 3).reshape(B * K, T, D)        # [B*K, T, D]
        xt = self.time_mamba(xt)
        xt = xt.reshape(B, K, T, D).permute(0, 2, 1, 3)         # [B, T, K, D]
        x = x + self.drop(xt)

        # ---- mode axis: fold (B, T) -> batch, scan over K ----
        xk = self.norm_k(x)
        xk = xk.reshape(B * T, K, D)                            # [B*T, K, D]
        xk = self.mode_mamba(xk)
        xk = xk.reshape(B, T, K, D)
        x = x + self.drop(xk)

        return x


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------


class SpectralStateSpaceModel(nn.Module):
    """
    SSSM end-to-end.

    Pipeline:
      1.  Project x_norm (node space) into spectral coefs via fixed U: x_hat = x_norm @ U
      2.  Apply learnable Chebyshev filter (C channels) -> [B, T, K, C]
      3.  Embed + add learnable mode embedding + TOD/DOW embeddings -> [B, T, K, D]
      4.  Bi-axis Mamba stack (L layers)
      5.  Decoder:
          a) Spectral head: predict residual spectral coefs [B, T_out, K]
             and apply fixed U.T to get node-space residual [B, T_out, N]
          b) Direct node-space head: predict node-space residual [B, T_out, N]
          (the two are summed; both pathways are trained jointly)
      6.  Add a "persistence" baseline (last observed normalized speed) so the
          model only needs to learn the DELTA from now.
    """

    def __init__(
        self,
        N: int,
        K: int,
        U: torch.Tensor,                  # [N, K]
        evals: torch.Tensor,              # [K]
        d_model: int = 96,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cheb_order: int = 3,
        cheb_channels: int = 4,
        dropout: float = 0.1,
        input_len: int = 12,
        pred_len: int = 12,
        num_dow: int = 7,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model
        self.input_len = input_len
        self.pred_len = pred_len
        self.cheb_channels = cheb_channels

        # Register fixed spectral basis as buffers (not trainable).
        # Map eigenvalues to [-1, 1] for Chebyshev stability.
        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        # Learnable filter over modes
        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)

        # Channel projection: cheb_channels filtered spectral signals -> d_model
        self.channel_proj = nn.Linear(cheb_channels, d_model)

        # Learnable per-mode positional embedding
        self.mode_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.mode_emb, std=0.02)

        # Time-of-day: small MLP from scalar in [0,1) -> d_model (sinusoidal-ish features)
        self.tod_proj = nn.Linear(2, d_model)  # uses (sin(2pi tod), cos(2pi tod))
        # Day-of-week: lookup table
        self.dow_emb = nn.Embedding(num_dow, d_model)

        # Bi-axis Mamba stack
        self.blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # Decoder heads
        # (a) Spectral residual head: per-mode time aggregator -> T_out scalars per mode
        self.spectral_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )
        # (b) Node-space residual head: mean over (T, K) -> Linear -> [T_out * N]
        self.node_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )
        # Mix the two residual branches with a learnable per-horizon gate
        self.head_gate = nn.Parameter(torch.tensor([0.0]))  # sigmoid -> 0.5 at init

    # ------------------------------------------------------------------
    def encode_time(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        """tod[B, T], dow[B, T] -> [B, T, d_model]"""
        # tod in [0, 1) -> 2 sinusoidal coords
        ang = 2.0 * math.pi * tod
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, T, 2]
        out = self.tod_proj(feat) + self.dow_emb(dow.long())
        return out

    # ------------------------------------------------------------------
    def forward(self, x_norm: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        """
        x_norm:  [B, T_in, N]
        tod:     [B, T_in]
        dow:     [B, T_in]
        returns: [B, T_out, N]   (normalized space)
        """
        B, T, N = x_norm.shape
        assert N == self.N, f"got N={N}, expected {self.N}"

        # 1) GFT projection (fixed)
        #    x_hat: [B, T, K]
        x_hat = x_norm @ self.U  # [B, T, K]

        # 2) Apply learnable Chebyshev gains: produces C "filtered" copies
        #    gains: [C, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)  # [B, T, K, C]

        # 3) Embed channels -> d_model and add mode + time embeddings
        h = self.channel_proj(x_filt)                                 # [B, T, K, D]
        h = h + self.mode_emb.view(1, 1, self.K, self.d_model)        # broadcast
        t_emb = self.encode_time(tod, dow)                            # [B, T, D]
        h = h + t_emb.unsqueeze(2)                                    # broadcast over K

        # 4) Bi-axis Mamba stack
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)                                          # [B, T, K, D]

        # 5a) Spectral residual head:
        #     pool time -> per (B, K) embedding, project to pred_len scalars
        h_time_last = h[:, -1, :, :]                                  # [B, K, D]  use last frame
        spec_resid = self.spectral_head(h_time_last)                  # [B, K, T_out]
        spec_resid = spec_resid.transpose(1, 2)                       # [B, T_out, K]
        node_from_spec = spec_resid @ self.Ut                         # [B, T_out, N]

        # 5b) Node-space residual head: pool modes -> per (B, T) embedding -> [B, T_out, N]
        #     We use mean over the K modes; the result is per-time, per-channel "global" feature.
        #     Then per-sensor predictions come from spec head; here we add per-sensor *bias*
        #     by mixing with a learnable per-sensor projection (linear over the aggregated D).
        h_pool = h.mean(dim=2)                                        # [B, T, D]
        node_resid_pre = self.node_head(h_pool[:, -1, :])             # [B, T_out]
        # broadcast to N sensors via a learnable per-sensor scale (initialized to 0)
        node_resid = node_resid_pre.unsqueeze(-1).expand(B, self.pred_len, N)

        gate = torch.sigmoid(self.head_gate)
        residual = gate * node_from_spec + (1.0 - gate) * node_resid  # [B, T_out, N]

        # 6) Persistence baseline: copy last observation T_out times.
        last_obs = x_norm[:, -1:, :].expand(B, self.pred_len, N)      # [B, T_out, N]
        y_hat = last_obs + residual
        return y_hat


# ----------------------------------------------------------------------------
# v2: encoder-decoder with future-time conditioning and per-sensor bias
# ----------------------------------------------------------------------------


class SpectralStateSpaceModelV2(nn.Module):
    """
    SSSM v2 - addresses three weaknesses of v1 found in the d96_L3 run:

    Weakness 1: The decoder reads only the LAST input timestep ([:, -1, :, :])
                and projects it through a Linear to produce all 12 future frames.
                The encoder's full temporal output is thrown away.
    Fix:        Concatenate the encoded input sequence with T_out learnable
                "decoder slots" (carrying future-time embeddings), then run a
                final temporal+mode-axis Mamba over the joined T_in+T_out
                sequence. The decoder positions can attend (causally, via
                Mamba's scan) to all encoder positions.

    Weakness 2: The decoder has no access to future time-of-day / day-of-week.
                At 60-min horizon this matters: predicting rush-hour onset is
                much easier if you know exactly when the prediction is for.
    Fix:        Future TOD/DOW are added to the decoder slots before the
                final Mamba pass.

    Weakness 3: Inverse-GFT readout produces only a global per-time delta over
                sensors (because the K mode coefficients are then projected by
                a fixed Ut to N sensors). Individual-sensor biases (one sensor
                consistently slower than its neighbors etc) cannot be expressed
                in the spectral domain alone.
    Fix:        Add a learnable per-sensor additive bias (zero-init) AFTER
                inverse-GFT. Same idea as STAEformer's "adaptive embedding"
                but applied minimally as a node-space bias only - the
                spectral-domain part of the architecture is unchanged.
    """

    def __init__(
        self,
        N: int,
        K: int,
        U: torch.Tensor,
        evals: torch.Tensor,
        d_model: int = 128,
        num_layers: int = 4,
        num_dec_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cheb_order: int = 3,
        cheb_channels: int = 4,
        dropout: float = 0.1,
        input_len: int = 12,
        pred_len: int = 12,
        num_dow: int = 7,
        use_node_bias: bool = True,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model
        self.input_len = input_len
        self.pred_len = pred_len
        self.cheb_channels = cheb_channels

        # Fixed GFT basis
        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        # Learnable spectral filter
        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)
        self.channel_proj = nn.Linear(cheb_channels, d_model)

        # Per-mode learnable embedding
        self.mode_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.mode_emb, std=0.02)

        # Time-of-day projection (sin/cos) + day-of-week embedding
        self.tod_proj = nn.Linear(2, d_model)
        self.dow_emb = nn.Embedding(num_dow, d_model)

        # Encoder: bi-axis Mamba
        self.enc_blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Decoder slot embeddings: per-horizon learnable initial query
        self.horizon_emb = nn.Parameter(torch.zeros(pred_len, d_model))
        nn.init.normal_(self.horizon_emb, std=0.02)

        # Decoder: bi-axis Mamba running on concat([enc_out, dec_slots])
        self.dec_blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # Spectral readout: [B, T_out, K, D] -> [B, T_out, K] (one spectral coef per slot)
        self.spec_proj = nn.Linear(d_model, 1)

        # Per-sensor additive bias (node-space). Zero-init so model starts as
        # a pure spectral predictor and only learns per-sensor offsets if useful.
        self.use_node_bias = use_node_bias
        if use_node_bias:
            self.node_bias = nn.Parameter(torch.zeros(pred_len, N))

    # ------------------------------------------------------------------
    def encode_time(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * tod
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, T, 2]
        return self.tod_proj(feat) + self.dow_emb(dow.long())

    # ------------------------------------------------------------------
    def forward(
        self,
        x_norm: torch.Tensor,    # [B, T_in, N]
        tod: torch.Tensor,        # [B, T_in]
        dow: torch.Tensor,        # [B, T_in]
        y_tod: torch.Tensor,      # [B, T_out]
        y_dow: torch.Tensor,      # [B, T_out]
    ) -> torch.Tensor:
        B, T_in, N = x_norm.shape
        T_out = self.pred_len
        K, D = self.K, self.d_model

        # --- encoder ---
        x_hat = x_norm @ self.U                                       # [B, T_in, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)  # [B, T_in, K, C]
        h_enc = self.channel_proj(x_filt)                             # [B, T_in, K, D]
        h_enc = h_enc + self.mode_emb.view(1, 1, K, D)
        t_emb_in = self.encode_time(tod, dow)                         # [B, T_in, D]
        h_enc = h_enc + t_emb_in.unsqueeze(2)                         # broadcast over K

        for blk in self.enc_blocks:
            h_enc = blk(h_enc)
        h_enc = self.enc_norm(h_enc)                                  # [B, T_in, K, D]

        # --- decoder slots ---
        # Build [B, T_out, K, D] slots seeded with: mode_emb + horizon_emb + future TOD/DOW
        t_emb_out = self.encode_time(y_tod, y_dow)                    # [B, T_out, D]
        h_dec = (self.mode_emb.view(1, 1, K, D)
                 + self.horizon_emb.view(1, T_out, 1, D)
                 + t_emb_out.unsqueeze(2))                            # [B, T_out, K, D]

        # --- run decoder Mamba over joined sequence ---
        # Concatenate along time; the decoder positions can causally attend
        # to all encoder positions via the time-axis Mamba's selective scan.
        h = torch.cat([h_enc, h_dec], dim=1)                          # [B, T_in+T_out, K, D]
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_norm(h)
        h_out = h[:, T_in:, :, :]                                     # [B, T_out, K, D]

        # --- spectral readout ---
        spec_pred = self.spec_proj(h_out).squeeze(-1)                 # [B, T_out, K]
        node_from_spec = spec_pred @ self.Ut                          # [B, T_out, N]

        # Persistence baseline
        last_obs = x_norm[:, -1:, :].expand(B, T_out, N)              # [B, T_out, N]
        y_hat = last_obs + node_from_spec
        if self.use_node_bias:
            y_hat = y_hat + self.node_bias.view(1, T_out, N)
        return y_hat


# ----------------------------------------------------------------------------
# v3: keep v1 encoder, add cross-attention decoder + future-time + node bias
# ----------------------------------------------------------------------------


class SpectralStateSpaceModelV3(nn.Module):
    """
    v3 - minimal targeted upgrade of v1.

    v1's bottleneck was the decoder: it read only h[:, -1, :, :] (last encoder
    frame) and projected that single (per-mode) vector to all 12 future
    horizons through a small MLP. That throws away the encoder's full
    temporal output and gives the decoder no information about future time.

    v2 tried to fix this with a Mamba encoder-decoder concat. That underperformed
    because the decoder slots had to *fetch* current-state information from
    the encoder through Mamba's selective scan -- a harder learning problem.

    v3 takes a simpler route: same bi-axis Mamba encoder as v1, but the
    decoder is a cross-attention head where:
      - per-horizon query   = horizon_emb[t] + future_tod[t] + future_dow[t] + mode_emb[k]
      - attention keys/vals = encoder output along time axis (per mode)
      - decoder output      = one spectral coefficient per (B, t, k)
      - readout             = inverse GFT + persistence + per-sensor bias

    The decoder per-mode attention is independent across modes, but the
    encoder's mode-axis Mamba has already mixed modes -- so the per-mode
    cross attention is correct for "look at the right time" while the
    encoder handles "look at the right mode".
    """

    def __init__(
        self,
        N: int,
        K: int,
        U: torch.Tensor,
        evals: torch.Tensor,
        d_model: int = 96,
        num_layers: int = 3,
        n_heads: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cheb_order: int = 3,
        cheb_channels: int = 4,
        dropout: float = 0.1,
        input_len: int = 12,
        pred_len: int = 12,
        num_dow: int = 7,
        use_node_bias: bool = True,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model
        self.input_len = input_len
        self.pred_len = pred_len

        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)
        self.channel_proj = nn.Linear(cheb_channels, d_model)

        self.mode_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.mode_emb, std=0.02)

        self.tod_proj = nn.Linear(2, d_model)
        self.dow_emb = nn.Embedding(num_dow, d_model)

        # Same bi-axis Mamba encoder as v1
        self.enc_blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Per-horizon learnable embedding (positional code for future timesteps)
        self.horizon_emb = nn.Parameter(torch.zeros(pred_len, d_model))
        nn.init.normal_(self.horizon_emb, std=0.02)

        # Cross-attention decoder
        self.dec_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            batch_first=True,
        )
        self.dec_norm1 = nn.LayerNorm(d_model)
        self.dec_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dec_norm2 = nn.LayerNorm(d_model)

        # Spectral readout: D -> 1 spectral coefficient
        self.spec_proj = nn.Linear(d_model, 1)

        # Per-sensor / per-horizon additive bias (zero-init)
        self.use_node_bias = use_node_bias
        if use_node_bias:
            self.node_bias = nn.Parameter(torch.zeros(pred_len, N))

    def encode_time(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * tod
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.tod_proj(feat) + self.dow_emb(dow.long())

    def forward(
        self,
        x_norm: torch.Tensor,    # [B, T_in, N]
        tod: torch.Tensor,        # [B, T_in]
        dow: torch.Tensor,        # [B, T_in]
        y_tod: torch.Tensor,      # [B, T_out]
        y_dow: torch.Tensor,      # [B, T_out]
    ) -> torch.Tensor:
        B, T_in, N = x_norm.shape
        T_out = self.pred_len
        K, D = self.K, self.d_model

        # ----- encoder (same as v1) -----
        x_hat = x_norm @ self.U                                       # [B, T_in, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)
        h = self.channel_proj(x_filt)                                 # [B, T_in, K, D]
        h = h + self.mode_emb.view(1, 1, K, D)
        t_emb_in = self.encode_time(tod, dow)                         # [B, T_in, D]
        h = h + t_emb_in.unsqueeze(2)

        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_norm(h)                                          # [B, T_in, K, D]

        # ----- per-horizon, per-mode queries -----
        # query[b, t, k] = horizon_emb[t] + future_t_emb[b, t] + mode_emb[k]
        future_t_emb = self.encode_time(y_tod, y_dow)                 # [B, T_out, D]
        q = (self.horizon_emb.view(1, T_out, 1, D)
             + future_t_emb.unsqueeze(2)
             + self.mode_emb.view(1, 1, K, D))                        # [B, T_out, K, D]

        # ----- cross-attention: per (b, k), attend over time -----
        # Reshape to [B*K, T_out, D] and [B*K, T_in, D]
        h_kv = h.permute(0, 2, 1, 3).contiguous().view(B * K, T_in, D)
        q_kv = q.permute(0, 2, 1, 3).contiguous().view(B * K, T_out, D)

        attn_out, _ = self.dec_attn(q_kv, h_kv, h_kv, need_weights=False)
        h_dec = self.dec_norm1(q_kv + attn_out)
        h_dec = self.dec_norm2(h_dec + self.dec_ffn(h_dec))            # [B*K, T_out, D]

        h_dec = h_dec.view(B, K, T_out, D).permute(0, 2, 1, 3).contiguous()  # [B, T_out, K, D]

        # ----- spectral readout -----
        spec_pred = self.spec_proj(h_dec).squeeze(-1)                  # [B, T_out, K]
        node_from_spec = spec_pred @ self.Ut                           # [B, T_out, N]

        # ----- persistence + bias -----
        last_obs = x_norm[:, -1:, :].expand(B, T_out, N)
        y_hat = last_obs + node_from_spec
        if self.use_node_bias:
            y_hat = y_hat + self.node_bias.view(1, T_out, N)
        return y_hat


# ----------------------------------------------------------------------------
# v4: keep v1's fast structure, add learnable temporal pool + future-time + bias
# ----------------------------------------------------------------------------


class SpectralStateSpaceModelV4(nn.Module):
    """
    v4 - minimal fix that targets v1's actual bottleneck without sacrificing
    the optimization simplicity that made v1 converge fast.

    v2 (encoder-decoder concat-Mamba) and v3 (cross-attention decoder) both
    require the decoder to *learn* to extract relevant state from random
    queries. They both converged ~3x slower per epoch than v1 in practice.

    v1's secret was that h[:, -1, :, :] is directly the encoder's last frame,
    so the spectral head sees real information at init. v4 keeps this
    structural simplicity but addresses v1's three weaknesses:

      1. Reads only the last encoder frame  -> use a learnable temporal pool
         (softmax weights over T_in) so all encoder frames contribute. With
         zero init the pool degenerates to uniform mean, and with appropriate
         init it can recover "use last frame" behavior. Either way, no
         learning-from-noise needed.
      2. No future TOD/DOW                 -> add per-horizon future-time
         embedding to the pooled per-mode embedding before the spectral head.
      3. No per-sensor bias                -> add a (T_out, N) zero-init bias.

    Decoder cost: ~K * T_out * D MACs per batch element (one Linear), so
    essentially free compared to the Mamba encoder.
    """

    def __init__(
        self,
        N: int,
        K: int,
        U: torch.Tensor,
        evals: torch.Tensor,
        d_model: int = 96,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cheb_order: int = 3,
        cheb_channels: int = 4,
        dropout: float = 0.1,
        input_len: int = 12,
        pred_len: int = 12,
        num_dow: int = 7,
        use_node_bias: bool = True,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model
        self.input_len = input_len
        self.pred_len = pred_len

        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)
        self.channel_proj = nn.Linear(cheb_channels, d_model)

        self.mode_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.mode_emb, std=0.02)

        self.tod_proj = nn.Linear(2, d_model)
        self.dow_emb = nn.Embedding(num_dow, d_model)

        # Same bi-axis Mamba encoder as v1
        self.enc_blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Learnable temporal pool: softmax over T_in frames
        # Init: bias toward the last frame (matches v1's behavior at init)
        pool_init = torch.zeros(input_len)
        pool_init[-1] = 2.0  # last frame dominates softmax at init
        self.t_pool_logits = nn.Parameter(pool_init)

        # Per-horizon embedding
        self.horizon_emb = nn.Parameter(torch.zeros(pred_len, d_model))
        nn.init.normal_(self.horizon_emb, std=0.02)

        # Spectral head: per-(horizon, mode) prediction from query
        self.spec_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.use_node_bias = use_node_bias
        if use_node_bias:
            self.node_bias = nn.Parameter(torch.zeros(pred_len, N))

    def encode_time(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * tod
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.tod_proj(feat) + self.dow_emb(dow.long())

    def forward(
        self,
        x_norm: torch.Tensor,    # [B, T_in, N]
        tod: torch.Tensor,        # [B, T_in]
        dow: torch.Tensor,        # [B, T_in]
        y_tod: torch.Tensor,      # [B, T_out]
        y_dow: torch.Tensor,      # [B, T_out]
    ) -> torch.Tensor:
        B, T_in, N = x_norm.shape
        T_out = self.pred_len
        K, D = self.K, self.d_model

        # ----- encoder (same as v1) -----
        x_hat = x_norm @ self.U                                       # [B, T_in, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)
        h = self.channel_proj(x_filt)                                 # [B, T_in, K, D]
        h = h + self.mode_emb.view(1, 1, K, D)
        t_emb_in = self.encode_time(tod, dow)                         # [B, T_in, D]
        h = h + t_emb_in.unsqueeze(2)

        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_norm(h)                                          # [B, T_in, K, D]

        # ----- learnable temporal pool -----
        t_w = torch.softmax(self.t_pool_logits, dim=0)                # [T_in]
        h_pool = (h * t_w.view(1, T_in, 1, 1)).sum(dim=1)             # [B, K, D]

        # ----- per-horizon queries: pooled + future-time + horizon -----
        future_t = self.encode_time(y_tod, y_dow)                     # [B, T_out, D]
        q = (h_pool.unsqueeze(1)                                       # [B, 1, K, D]
             + future_t.unsqueeze(2)                                   # [B, T_out, 1, D]
             + self.horizon_emb.view(1, T_out, 1, D))                  # [B, T_out, K, D]

        # ----- spectral readout -----
        spec_pred = self.spec_head(q).squeeze(-1)                     # [B, T_out, K]
        node_from_spec = spec_pred @ self.Ut                          # [B, T_out, N]

        # ----- persistence + bias -----
        last_obs = x_norm[:, -1:, :].expand(B, T_out, N)
        y_hat = last_obs + node_from_spec
        if self.use_node_bias:
            y_hat = y_hat + self.node_bias.view(1, T_out, N)
        return y_hat


# ----------------------------------------------------------------------------
# v5: v4 architecture + calendar prior as baseline (replaces persistence)
# ----------------------------------------------------------------------------


class SpectralStateSpaceModelV5(nn.Module):
    """
    v5 - v4 augmented with a *calendar prior* as the prediction baseline.

    All v1-v4 models used last-observation (persistence) as the baseline and
    predicted a residual on top. METR-LA traffic is dominated by deterministic
    daily/weekly periodicity (rush-hour cycles, weekend patterns); persistence
    is a poor baseline at long horizons because it ignores upcoming
    rush-hour onset/release.

    v5 replaces persistence with a per-(sensor, TOD-bin, DOW) mean speed
    computed from training data. This gives the model a strong "what is
    normally happening at the prediction timestamp" signal, so all the
    encoder needs to learn is the *deviation* from typical behavior (which
    is small and easier to predict accurately than the absolute speed).

    The prior also enters as an additional input feature: the encoder sees
    `x_norm - prior_norm` (residualized history) in addition to the raw
    history, so it can detect when *now* deviates from typical, which is
    the cue for when the future will too.
    """

    def __init__(
        self,
        N: int,
        K: int,
        U: torch.Tensor,
        evals: torch.Tensor,
        d_model: int = 96,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cheb_order: int = 3,
        cheb_channels: int = 4,
        dropout: float = 0.1,
        input_len: int = 12,
        pred_len: int = 12,
        num_dow: int = 7,
        use_node_bias: bool = True,
        residualize_input: bool = True,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model
        self.input_len = input_len
        self.pred_len = pred_len
        self.residualize_input = residualize_input

        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)
        # Two input channel groups: raw spectral coefs AND residual-vs-prior coefs
        in_channels = cheb_channels * 2 if residualize_input else cheb_channels
        self.channel_proj = nn.Linear(in_channels, d_model)

        self.mode_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.mode_emb, std=0.02)

        self.tod_proj = nn.Linear(2, d_model)
        self.dow_emb = nn.Embedding(num_dow, d_model)

        self.enc_blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Learnable temporal pool
        pool_init = torch.zeros(input_len)
        pool_init[-1] = 2.0
        self.t_pool_logits = nn.Parameter(pool_init)

        self.horizon_emb = nn.Parameter(torch.zeros(pred_len, d_model))
        nn.init.normal_(self.horizon_emb, std=0.02)

        # Spectral head: predict per-(horizon, mode) coefficient delta
        self.spec_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.use_node_bias = use_node_bias
        if use_node_bias:
            self.node_bias = nn.Parameter(torch.zeros(pred_len, N))

        # Per-horizon learnable mix between persistence and calendar prior baselines.
        # Init biased: early horizons toward persistence (logit > 0),
        # late horizons toward prior (logit < 0). Specifically, linearly interpolate
        # from +2 (sigmoid ~ 0.88, 88% persistence) at t=0 down to -2 at t=T_out-1
        # (sigmoid ~ 0.12, 88% prior). The model can refine from there.
        gate_init = torch.linspace(2.0, -2.0, pred_len)
        self.baseline_gate = nn.Parameter(gate_init)

    def encode_time(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * tod
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.tod_proj(feat) + self.dow_emb(dow.long())

    def forward(
        self,
        x_norm: torch.Tensor,         # [B, T_in, N]
        tod: torch.Tensor,             # [B, T_in]
        dow: torch.Tensor,             # [B, T_in]
        y_tod: torch.Tensor,           # [B, T_out]
        y_dow: torch.Tensor,           # [B, T_out]
        prior_norm_in: torch.Tensor,   # [B, T_in, N]
        prior_norm_out: torch.Tensor,  # [B, T_out, N]
    ) -> torch.Tensor:
        B, T_in, N = x_norm.shape
        T_out = self.pred_len
        K, D = self.K, self.d_model

        # ----- spectral projection: both raw and residual-vs-prior -----
        x_hat = x_norm @ self.U                                       # [B, T_in, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)  # [B, T_in, K, C]

        if self.residualize_input:
            r_hat = (x_norm - prior_norm_in) @ self.U                 # [B, T_in, K]
            r_filt = r_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)
            feat = torch.cat([x_filt, r_filt], dim=-1)                # [B, T_in, K, 2C]
        else:
            feat = x_filt

        h = self.channel_proj(feat)                                   # [B, T_in, K, D]
        h = h + self.mode_emb.view(1, 1, K, D)
        t_emb_in = self.encode_time(tod, dow)                         # [B, T_in, D]
        h = h + t_emb_in.unsqueeze(2)

        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_norm(h)

        # ----- pool + per-horizon queries -----
        t_w = torch.softmax(self.t_pool_logits, dim=0)
        h_pool = (h * t_w.view(1, T_in, 1, 1)).sum(dim=1)             # [B, K, D]

        future_t = self.encode_time(y_tod, y_dow)                     # [B, T_out, D]
        q = (h_pool.unsqueeze(1)
             + future_t.unsqueeze(2)
             + self.horizon_emb.view(1, T_out, 1, D))                 # [B, T_out, K, D]

        spec_pred = self.spec_head(q).squeeze(-1)                     # [B, T_out, K]
        node_from_spec = spec_pred @ self.Ut                          # [B, T_out, N]

        # ----- baseline = learnable per-horizon mix(persistence, calendar prior) -----
        last_obs = x_norm[:, -1:, :].expand(B, T_out, N)
        alpha = torch.sigmoid(self.baseline_gate).view(1, T_out, 1)   # [1, T_out, 1]
        baseline = alpha * last_obs + (1.0 - alpha) * prior_norm_out

        y_hat = baseline + node_from_spec
        if self.use_node_bias:
            y_hat = y_hat + self.node_bias.view(1, T_out, N)
        return y_hat


# ----------------------------------------------------------------------------
# v7: multi-window input (recent + day-ago + week-ago)
# ----------------------------------------------------------------------------


class SpectralStateSpaceModelV7(nn.Module):
    """
    v7 - v4 with multi-window input.

    METR-LA traffic has strong daily and weekly periodicity. A model that
    sees only the last 12 timesteps (1 hour) has to LEARN this periodicity
    from data. v7 gives it directly: each sample includes the same
    12-timestep window from 1 day ago and 7 days ago, concatenated into a
    36-timestep "augmented input". The model can then condition on
    "what was happening at this exact (TOD, DOW) one day / one week ago".

    Mamba's selective scan along the time axis can naturally weight the
    informative windows. We add a 3-way window-id embedding (0=recent,
    1=day_ago, 2=week_ago) so the model can distinguish them.
    """

    def __init__(
        self,
        N: int,
        K: int,
        U: torch.Tensor,
        evals: torch.Tensor,
        d_model: int = 96,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cheb_order: int = 3,
        cheb_channels: int = 4,
        dropout: float = 0.1,
        input_len: int = 12,
        pred_len: int = 12,
        num_dow: int = 7,
        num_windows: int = 3,         # 1=only recent, 3=recent+day+week
        use_node_bias: bool = True,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model
        self.input_len = input_len
        self.pred_len = pred_len
        self.num_windows = num_windows
        self.total_in = input_len * num_windows

        evals_max = float(evals.max().item() + 1e-6)
        evals_scaled = (2.0 * evals / evals_max) - 1.0
        self.register_buffer("U", U.contiguous())
        self.register_buffer("Ut", U.t().contiguous())
        self.register_buffer("evals_scaled", evals_scaled.contiguous())

        self.cheb = ChebFilter(K=K, order=cheb_order, channels=cheb_channels)
        self.channel_proj = nn.Linear(cheb_channels, d_model)

        self.mode_emb = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.mode_emb, std=0.02)

        self.tod_proj = nn.Linear(2, d_model)
        self.dow_emb = nn.Embedding(num_dow, d_model)
        self.win_emb = nn.Embedding(num_windows, d_model)

        self.enc_blocks = nn.ModuleList([
            BiAxisMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Learnable temporal pool over the total_in steps
        # Init bias toward the end of the sequence (recent window's last step)
        pool_init = torch.zeros(self.total_in)
        pool_init[-1] = 2.0
        self.t_pool_logits = nn.Parameter(pool_init)

        self.horizon_emb = nn.Parameter(torch.zeros(pred_len, d_model))
        nn.init.normal_(self.horizon_emb, std=0.02)

        self.spec_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.use_node_bias = use_node_bias
        if use_node_bias:
            self.node_bias = nn.Parameter(torch.zeros(pred_len, N))

    def encode_time(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        ang = 2.0 * math.pi * tod
        feat = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.tod_proj(feat) + self.dow_emb(dow.long())

    def forward(
        self,
        x_norm: torch.Tensor,    # [B, total_in, N]
        tod: torch.Tensor,        # [B, total_in]
        dow: torch.Tensor,        # [B, total_in]
        win_id: torch.Tensor,     # [B, total_in]
        y_tod: torch.Tensor,      # [B, T_out]
        y_dow: torch.Tensor,      # [B, T_out]
    ) -> torch.Tensor:
        B, T_total, N = x_norm.shape
        T_out = self.pred_len
        K, D = self.K, self.d_model

        # ----- encoder -----
        x_hat = x_norm @ self.U                                       # [B, T, K]
        gains = self.cheb(self.evals_scaled)                          # [C, K]
        x_filt = x_hat.unsqueeze(-1) * gains.t().unsqueeze(0).unsqueeze(0)
        h = self.channel_proj(x_filt)                                 # [B, T, K, D]
        h = h + self.mode_emb.view(1, 1, K, D)

        t_emb_in = self.encode_time(tod, dow)                         # [B, T, D]
        w_emb_in = self.win_emb(win_id.long())                        # [B, T, D]
        h = h + (t_emb_in + w_emb_in).unsqueeze(2)

        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_norm(h)

        # ----- temporal pool -----
        t_w = torch.softmax(self.t_pool_logits, dim=0)                # [T_total]
        h_pool = (h * t_w.view(1, T_total, 1, 1)).sum(dim=1)          # [B, K, D]

        # ----- per-horizon queries -----
        future_t = self.encode_time(y_tod, y_dow)                     # [B, T_out, D]
        q = (h_pool.unsqueeze(1)
             + future_t.unsqueeze(2)
             + self.horizon_emb.view(1, T_out, 1, D))                 # [B, T_out, K, D]

        spec_pred = self.spec_head(q).squeeze(-1)                     # [B, T_out, K]
        node_from_spec = spec_pred @ self.Ut                          # [B, T_out, N]

        # ----- persistence baseline = last step of RECENT window -----
        # In multi-window concat order: [..., week, day, recent]
        # The very last step (index -1) is recent's last frame = "now"
        last_obs = x_norm[:, -1:, :].expand(B, T_out, N)
        y_hat = last_obs + node_from_spec
        if self.use_node_bias:
            y_hat = y_hat + self.node_bias.view(1, T_out, N)
        return y_hat


# ----------------------------------------------------------------------------
# Utility: build model from cached artifacts
# ----------------------------------------------------------------------------


def build_model(K: int, U_np, evals_np, version: str = "v1", **kwargs):
    import numpy as np
    if isinstance(U_np, np.ndarray):
        U = torch.from_numpy(U_np).float()
    else:
        U = U_np.float()
    if isinstance(evals_np, np.ndarray):
        ev = torch.from_numpy(evals_np).float()
    else:
        ev = evals_np.float()
    N = U.shape[0]
    if version == "v2":
        return SpectralStateSpaceModelV2(N=N, K=K, U=U, evals=ev, **kwargs)
    if version == "v3":
        return SpectralStateSpaceModelV3(N=N, K=K, U=U, evals=ev, **kwargs)
    if version == "v4":
        return SpectralStateSpaceModelV4(N=N, K=K, U=U, evals=ev, **kwargs)
    if version == "v5":
        return SpectralStateSpaceModelV5(N=N, K=K, U=U, evals=ev, **kwargs)
    if version == "v7":
        return SpectralStateSpaceModelV7(N=N, K=K, U=U, evals=ev, **kwargs)
    return SpectralStateSpaceModel(N=N, K=K, U=U, evals=ev, **kwargs)
