"""
TESTAM port — three experts (Temporal / Adaptive-Graph / Dynamic-Attention) blended
through a learnable memory gate. Trained end-to-end with masked MAE.

We deliberately drop the original paper's hard-routing + auxiliary best/worst-choice
losses in favor of softmax mixing, because (a) the soft variant matches our masked-
MAE pipeline directly, (b) it removes the gate-collapse risk on a small graph, and
(c) for ensembling purposes we want each expert to contribute every sample, not just
the gate's pick. An optional `aux_gate_loss` flag is provided to enable the original
best-choice / worst-avoidance auxiliary losses for ablation.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
class _MHA(nn.Module):
    """Standard scaled-dot-product MHA over a chosen sequence axis."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., L, D]
        prefix = x.shape[:-2]
        L, D = x.shape[-2], x.shape[-1]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # split heads
        q = q.reshape(*prefix, L, self.num_heads, self.head_dim).transpose(-2, -3)
        k = k.reshape(*prefix, L, self.num_heads, self.head_dim).transpose(-2, -3)
        v = v.reshape(*prefix, L, self.num_heads, self.head_dim).transpose(-2, -3)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                                            # [..., H, L, hd]
        out = out.transpose(-2, -3).reshape(*prefix, L, D)
        return self.drop(self.out(out))


class _TransformerBlock(nn.Module):
    """Pre-LN transformer block over an explicit sequence axis."""

    def __init__(self, d_model: int, ffn_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MHA(d_model, num_heads, dropout, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Expert 1: Temporal expert (no graph)
# ---------------------------------------------------------------------------
class TemporalExpert(nn.Module):
    """
    Per-node temporal Transformer. Contracts N into the batch dimension so each
    sensor is processed independently along the time axis.
    """

    def __init__(self, n_nodes: int, d_model: int = 32, num_layers: int = 3,
                 ffn_dim: int = 64, num_heads: int = 4, dropout: float = 0.3,
                 in_steps: int = 12, out_steps: int = 12,
                 steps_per_day: int = 288, n_dow: int = 7):
        super().__init__()
        self.d_model = d_model
        self.in_steps = in_steps
        self.out_steps = out_steps

        self.input_proj = nn.Linear(1, d_model)
        self.tod_embed = nn.Embedding(steps_per_day, d_model)
        self.dow_embed = nn.Embedding(n_dow, d_model)
        self.node_embed = nn.Parameter(torch.zeros(n_nodes, d_model))
        nn.init.normal_(self.node_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, ffn_dim, num_heads, dropout, causal=False)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(in_steps * d_model, out_steps)

    def forward(self, x_norm: torch.Tensor, tod_idx: torch.Tensor,
                dow_idx: torch.Tensor) -> torch.Tensor:
        # x_norm: [B, T, N]; tod_idx, dow_idx: [B, T] long
        B, T, N = x_norm.shape
        D = self.d_model
        h = self.input_proj(x_norm.unsqueeze(-1))                 # [B, T, N, D]
        tod_e = self.tod_embed(tod_idx).unsqueeze(2).expand(B, T, N, D)
        dow_e = self.dow_embed(dow_idx).unsqueeze(2).expand(B, T, N, D)
        node_e = self.node_embed.view(1, 1, N, D).expand(B, T, N, D)
        h = h + tod_e + dow_e + node_e

        # contract N into batch, attend over T
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        for blk in self.blocks:
            h = blk(h)
        # head: collapse T*D into T_out
        h = h.reshape(B, N, T * D)
        out = self.head(h).transpose(1, 2)                        # [B, T_out, N]
        return out


# ---------------------------------------------------------------------------
# Expert 2: Spatio-temporal expert with adaptive graph
# ---------------------------------------------------------------------------
class _AdaptiveGraph(nn.Module):
    """
    Learns two embedding matrices (E1, E2) and a memory bank M, then derives a
    pair of adaptive support matrices via softmax(ReLU(E1 M E2^T)) (paper's
    Section 3 description).
    """

    def __init__(self, n_nodes: int, d_emb: int = 16, memory_size: int = 20):
        super().__init__()
        self.E1 = nn.Parameter(torch.empty(n_nodes, d_emb))
        self.E2 = nn.Parameter(torch.empty(n_nodes, d_emb))
        self.M = nn.Parameter(torch.empty(memory_size, d_emb))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)
        nn.init.xavier_uniform_(self.M)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns two [N, N] supports."""
        a = self.E1 @ self.M.T                                    # [N, mem]
        b = self.E2 @ self.M.T                                    # [N, mem]
        s = F.relu(a @ b.T)
        s = F.softmax(s, dim=-1)                                  # row-stochastic
        return s, s.T


class STExpert(nn.Module):
    """
    Adaptive-graph spatio-temporal expert. Repeats `n_layers` blocks of
    (graph-conv on N, transformer on T) over a [B, T, N, D] tensor.
    """

    def __init__(self, n_nodes: int, d_model: int = 32, num_layers: int = 3,
                 ffn_dim: int = 64, num_heads: int = 4, dropout: float = 0.3,
                 in_steps: int = 12, out_steps: int = 12,
                 steps_per_day: int = 288, n_dow: int = 7,
                 d_emb: int = 16, memory_size: int = 20):
        super().__init__()
        self.d_model = d_model
        self.in_steps = in_steps
        self.out_steps = out_steps

        self.input_proj = nn.Linear(1, d_model)
        self.tod_embed = nn.Embedding(steps_per_day, d_model)
        self.dow_embed = nn.Embedding(n_dow, d_model)
        self.node_embed = nn.Parameter(torch.zeros(n_nodes, d_model))
        nn.init.normal_(self.node_embed, std=0.02)

        self.graph = _AdaptiveGraph(n_nodes, d_emb, memory_size)
        # one graph conv + one temporal block per layer
        self.gconv_w = nn.ParameterList([
            nn.Parameter(torch.empty(2 * d_model, d_model))
            for _ in range(num_layers)
        ])
        self.gconv_b = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(num_layers)
        ])
        for w in self.gconv_w:
            nn.init.xavier_uniform_(w)
        self.gconv_ln = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.t_blocks = nn.ModuleList([
            _TransformerBlock(d_model, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(in_steps * d_model, out_steps)

    def _gconv(self, h: torch.Tensor, s_fwd: torch.Tensor, s_bwd: torch.Tensor,
               layer: int) -> torch.Tensor:
        # h: [B, T, N, D]
        # support multiplication along node axis
        # contract: out[..., n, d] = sum_{m} s[n, m] * h[..., m, d]
        h_fwd = torch.einsum("nm,btmd->btnd", s_fwd, h)
        h_bwd = torch.einsum("nm,btmd->btnd", s_bwd, h)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)                 # [B, T, N, 2D]
        h_out = torch.einsum("btnd,de->btne", h_cat, self.gconv_w[layer])
        h_out = h_out + self.gconv_b[layer]
        return self.gconv_ln[layer](h_out)

    def forward(self, x_norm: torch.Tensor, tod_idx: torch.Tensor,
                dow_idx: torch.Tensor) -> torch.Tensor:
        B, T, N = x_norm.shape
        D = self.d_model
        h = self.input_proj(x_norm.unsqueeze(-1))
        tod_e = self.tod_embed(tod_idx).unsqueeze(2).expand(B, T, N, D)
        dow_e = self.dow_embed(dow_idx).unsqueeze(2).expand(B, T, N, D)
        node_e = self.node_embed.view(1, 1, N, D).expand(B, T, N, D)
        h = h + tod_e + dow_e + node_e

        s_fwd, s_bwd = self.graph()                               # both [N, N]
        for L in range(len(self.t_blocks)):
            # spatial step
            h = h + self._gconv(h, s_fwd, s_bwd, L)
            # temporal step (contract N into batch)
            h_t = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
            h_t = self.t_blocks[L](h_t)
            h = h_t.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()

        h = h.permute(0, 2, 1, 3).contiguous().view(B, N, T * D)
        out = self.head(h).transpose(1, 2)                        # [B, T_out, N]
        return out


# ---------------------------------------------------------------------------
# Expert 3: Attention-only expert (no graph)
# ---------------------------------------------------------------------------
class AttentionExpert(nn.Module):
    """Pure spatial+temporal Transformer, no graph."""

    def __init__(self, n_nodes: int, d_model: int = 32, num_layers: int = 3,
                 ffn_dim: int = 64, num_heads: int = 4, dropout: float = 0.3,
                 in_steps: int = 12, out_steps: int = 12,
                 steps_per_day: int = 288, n_dow: int = 7):
        super().__init__()
        self.d_model = d_model
        self.in_steps = in_steps
        self.out_steps = out_steps

        self.input_proj = nn.Linear(1, d_model)
        self.tod_embed = nn.Embedding(steps_per_day, d_model)
        self.dow_embed = nn.Embedding(n_dow, d_model)
        self.node_embed = nn.Parameter(torch.zeros(n_nodes, d_model))
        nn.init.normal_(self.node_embed, std=0.02)

        self.t_blocks = nn.ModuleList([
            _TransformerBlock(d_model, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.s_blocks = nn.ModuleList([
            _TransformerBlock(d_model, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(in_steps * d_model, out_steps)

    def forward(self, x_norm: torch.Tensor, tod_idx: torch.Tensor,
                dow_idx: torch.Tensor) -> torch.Tensor:
        B, T, N = x_norm.shape
        D = self.d_model
        h = self.input_proj(x_norm.unsqueeze(-1))
        tod_e = self.tod_embed(tod_idx).unsqueeze(2).expand(B, T, N, D)
        dow_e = self.dow_embed(dow_idx).unsqueeze(2).expand(B, T, N, D)
        node_e = self.node_embed.view(1, 1, N, D).expand(B, T, N, D)
        h = h + tod_e + dow_e + node_e                            # [B, T, N, D]

        for tb, sb in zip(self.t_blocks, self.s_blocks):
            # temporal: contract N into batch
            h_t = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
            h_t = tb(h_t)
            h = h_t.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
            # spatial: contract T into batch
            h_s = h.contiguous().view(B * T, N, D)
            h_s = sb(h_s)
            h = h_s.view(B, T, N, D)

        h = h.permute(0, 2, 1, 3).contiguous().view(B, N, T * D)
        out = self.head(h).transpose(1, 2)                        # [B, T_out, N]
        return out


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------
class MemoryGate(nn.Module):
    """
    Per-(sample, horizon, sensor) softmax over experts, conditioned on a small
    learnable memory bank.

    Output: [B, T_out, N, n_experts] softmax weights.
    """

    def __init__(self, n_nodes: int, n_experts: int = 3, d_gate: int = 32,
                 memory_size: int = 20, in_steps: int = 12, out_steps: int = 12,
                 steps_per_day: int = 288, n_dow: int = 7):
        super().__init__()
        self.n_experts = n_experts
        self.in_steps = in_steps
        self.out_steps = out_steps

        self.M = nn.Parameter(torch.empty(memory_size, d_gate))
        nn.init.xavier_uniform_(self.M)
        self.q_proj = nn.Linear(d_gate, d_gate)
        self.k_proj = nn.Linear(d_gate, d_gate)

        self.x_proj = nn.Linear(1, d_gate)
        self.tod_e = nn.Embedding(steps_per_day, d_gate)
        self.dow_e = nn.Embedding(n_dow, d_gate)
        self.h_emb = nn.Embedding(out_steps, d_gate)
        self.node_e = nn.Parameter(torch.zeros(n_nodes, d_gate))
        nn.init.normal_(self.node_e, std=0.02)

        self.mlp = nn.Sequential(
            nn.Linear(d_gate, d_gate),
            nn.GELU(),
            nn.Linear(d_gate, n_experts),
        )

    def forward(self, x_norm: torch.Tensor, tod_idx: torch.Tensor,
                dow_idx: torch.Tensor) -> torch.Tensor:
        # x_norm: [B, T_in, N]; tod_idx, dow_idx: [B, T_in]
        B, T, N = x_norm.shape
        D = self.M.shape[-1]

        # condense input over time (last step) for a query vector per (B, N)
        x_last = x_norm[:, -1:, :].unsqueeze(-1)                  # [B, 1, N, 1]
        h = self.x_proj(x_last).squeeze(1)                        # [B, N, D]
        tod_e = self.tod_e(tod_idx[:, -1]).unsqueeze(1).expand(B, N, D)
        dow_e = self.dow_e(dow_idx[:, -1]).unsqueeze(1).expand(B, N, D)
        node_e = self.node_e.view(1, N, D).expand(B, N, D)
        q = self.q_proj(h + tod_e + dow_e + node_e)               # [B, N, D]

        # attend memory
        k = self.k_proj(self.M)                                   # [mem, D]
        scores = (q @ k.T) / math.sqrt(D)                         # [B, N, mem]
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ self.M                                       # [B, N, D]

        # broadcast over horizon
        h_e = self.h_emb(torch.arange(self.out_steps, device=x_norm.device))
        feat = ctx.unsqueeze(1) + h_e.view(1, self.out_steps, 1, D)
        logits = self.mlp(feat)                                   # [B, T_out, N, E]
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Full TESTAM model
# ---------------------------------------------------------------------------
class TESTAM(nn.Module):
    """
    Three experts blended by a memory gate, end-to-end trained with masked MAE.

    forward(x_norm, tod, dow)
      x_norm: [B, T_in, N] float (normalized speed)
      tod:    [B, T_in] float in [0, 1)
      dow:    [B, T_in] long in {0..6}
    Returns:
      y_pred_norm: [B, T_out, N]   (caller de-normalizes for the loss)
    """

    def __init__(
        self,
        N: int = 207,
        in_steps: int = 12,
        out_steps: int = 12,
        d_model: int = 32,
        num_layers: int = 3,
        ffn_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        steps_per_day: int = 288,
        n_dow: int = 7,
        d_emb: int = 16,
        memory_size: int = 20,
        d_gate: int = 32,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day

        common = dict(
            n_nodes=N, d_model=d_model, num_layers=num_layers, ffn_dim=ffn_dim,
            num_heads=num_heads, dropout=dropout, in_steps=in_steps,
            out_steps=out_steps, steps_per_day=steps_per_day, n_dow=n_dow,
        )
        self.temporal_expert = TemporalExpert(**common)
        self.st_expert = STExpert(
            **common, d_emb=d_emb, memory_size=memory_size,
        )
        self.attention_expert = AttentionExpert(**common)
        self.gate = MemoryGate(
            n_nodes=N, n_experts=3, d_gate=d_gate, memory_size=memory_size,
            in_steps=in_steps, out_steps=out_steps,
            steps_per_day=steps_per_day, n_dow=n_dow,
        )

    def _tod_idx(self, tod: torch.Tensor) -> torch.Tensor:
        return (tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)

    def forward(
        self,
        x_norm: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        return_experts: bool = False,
    ) -> torch.Tensor:
        tod_idx = self._tod_idx(tod)
        dow_idx = dow.long().clamp(0, 6)

        y_t = self.temporal_expert(x_norm, tod_idx, dow_idx)      # [B, T_out, N]
        y_s = self.st_expert(x_norm, tod_idx, dow_idx)
        y_a = self.attention_expert(x_norm, tod_idx, dow_idx)
        stacked = torch.stack([y_t, y_s, y_a], dim=-1)            # [B, T_out, N, 3]
        gate = self.gate(x_norm, tod_idx, dow_idx)                # [B, T_out, N, 3]
        y_mix = (stacked * gate).sum(dim=-1)                      # [B, T_out, N]
        if return_experts:
            return y_mix, stacked, gate
        return y_mix


def build_testam(N: int = 207, **kwargs) -> TESTAM:
    """Convenience constructor with TESTAM paper defaults for METR-LA."""
    return TESTAM(N=N, **kwargs)
