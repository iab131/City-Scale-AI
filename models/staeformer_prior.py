"""
STAEformer with calendar-prior feature stream.

In addition to the standard input embeddings (feature/TOD/DOW/adaptive),
adds a per-(time, sensor) "calendar prior" feature — the historical mean
speed at that (TOD-bin, DOW, sensor) triple from training data. The model
learns to refine the prior.

Different from spectral_ssm V5 which used the prior as a BASELINE (replacing
persistence). Here it's an INPUT FEATURE, so it acts as guidance rather
than substitution.

Input:
  x_norm     [B, T_in, N]      normalized speeds
  tod        [B, T_in]         time-of-day
  dow        [B, T_in]         day-of-week
  prior_norm [B, T_in, N]      normalized calendar-prior speeds

Output:
  y_norm     [B, T_out, N]
"""

import torch
import torch.nn as nn
from models.staeformer import STAEformer, SelfAttentionLayer


class STAEformerWithPrior(nn.Module):
    def __init__(
        self, N: int,
        in_steps: int = 12, out_steps: int = 12,
        steps_per_day: int = 288, num_dow: int = 7,
        input_embedding_dim: int = 24,
        prior_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day

        self.model_dim = (
            input_embedding_dim + prior_embedding_dim
            + tod_embedding_dim + dow_embedding_dim
            + adaptive_embedding_dim
        )

        self.input_proj = nn.Linear(1, input_embedding_dim)
        self.prior_proj = nn.Linear(1, prior_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(num_dow, dow_embedding_dim)
        self.adaptive_embedding = nn.Parameter(torch.empty(in_steps, N, adaptive_embedding_dim))
        nn.init.xavier_uniform_(self.adaptive_embedding)

        self.temporal_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps)

    def forward(self, x_norm, tod, dow, prior_norm, y_tod=None, y_dow=None):
        B, T_in, N = x_norm.shape
        feats = []
        feats.append(self.input_proj(x_norm.unsqueeze(-1)))
        feats.append(self.prior_proj(prior_norm.unsqueeze(-1)))
        tod_idx = (tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
        feats.append(self.tod_embedding(tod_idx).unsqueeze(2).expand(B, T_in, N, -1))
        feats.append(self.dow_embedding(dow.long()).unsqueeze(2).expand(B, T_in, N, -1))
        feats.append(self.adaptive_embedding.unsqueeze(0).expand(B, T_in, N, -1))
        h = torch.cat(feats, dim=-1)
        for layer in self.temporal_layers:
            h = layer(h, dim=1)
        for layer in self.spatial_layers:
            h = layer(h, dim=2)
        out = h.transpose(1, 2).reshape(B, N, T_in * self.model_dim)
        return self.output_proj(out).transpose(1, 2)
