"""
STAEformer (CIKM 2023) reproduction.

Faithful reimplementation of:
  Liu et al., "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer
  SOTA for Traffic Forecasting", CIKM 2023 (arXiv:2308.10425).

Used as the SOTA-class backbone for our hybrid path. Reproduces:
  - feature/TOD/DOW/adaptive embeddings concatenated to model_dim=152
  - 3-layer temporal Transformer (post-LN, FFN 152->256->152)
  - 3-layer spatial Transformer (same block, different attention axis)
  - mixed projection output (flatten T*D -> Linear -> T*1)

Reference code:
  https://github.com/XDZhelheim/STAEformer

This module is wired to receive separate tensors (x_norm, tod, dow) so that
it composes cleanly with our data pipeline (preprocess_v2.py + dataset_v2.py).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention with the STAEformer's specific "heads-on-batch-dim"
    implementation. Functionally identical to nn.MultiheadAttention.
    """

    def __init__(self, model_dim: int, num_heads: int, mask: bool = False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert model_dim % num_heads == 0
        self.mask = mask
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, q, k, v):
        # q, k, v: [..., L, model_dim]
        # Split heads on the batch dim (STAEformer convention).
        Q = torch.cat(torch.split(self.fc_q(q), self.head_dim, dim=-1), dim=0)
        K = torch.cat(torch.split(self.fc_k(k), self.head_dim, dim=-1), dim=0)
        V = torch.cat(torch.split(self.fc_v(v), self.head_dim, dim=-1), dim=0)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.mask:
            mask = torch.ones_like(scores, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = torch.cat(torch.split(out, q.size(0), dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    """Standard post-LN Transformer block with self-attention."""

    def __init__(self, model_dim: int, ffn_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, mask: bool = False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        """
        x: [..., L, model_dim]
        dim: which axis is the sequence axis. Will transpose so it's -2.
        """
        x = x.transpose(dim, -2)
        h = self.attn(x, x, x)
        h = self.drop1(h)
        x = self.ln1(x + h)
        h = self.ffn(x)
        h = self.drop2(h)
        x = self.ln2(x + h)
        x = x.transpose(dim, -2)
        return x


class STAEformer(nn.Module):
    """
    STAEformer reference implementation, taking our pipeline's tensor format.

    Inputs:
        x_norm: [B, T_in, N]      masked-normalized speeds
        tod:    [B, T_in]         time-of-day in [0, 1)
        dow:    [B, T_in]         day-of-week int in {0..6}

    Returns:
        y_hat_norm: [B, T_out, N]  in normalized space (caller de-normalizes)
    """

    def __init__(
        self,
        N: int,
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
        spatial_embedding_dim: int = 0,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day

        # Per the reference: model_dim = sum of all embedding dims
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        # Embeddings
        self.input_proj = nn.Linear(1, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim) if tod_embedding_dim else None
        self.dow_embedding = nn.Embedding(num_dow, dow_embedding_dim) if dow_embedding_dim else None
        self.spatial_embedding_dim = spatial_embedding_dim
        if spatial_embedding_dim:
            self.spatial_embedding = nn.Parameter(torch.empty(N, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.spatial_embedding)
        else:
            self.register_parameter("spatial_embedding", None)

        # Adaptive embedding — STAEformer's headline novelty.
        self.adaptive_embedding_dim = adaptive_embedding_dim
        if adaptive_embedding_dim:
            self.adaptive_embedding = nn.Parameter(torch.empty(in_steps, N, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.register_parameter("adaptive_embedding", None)

        # Temporal Transformer
        self.temporal_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Spatial Transformer
        self.spatial_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Mixed projection head: flatten T_in*D -> Linear -> T_out
        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps)

    def get_hidden(self, x_norm, tod, dow):
        """
        Returns the post-encoder hidden state [B, T_in, N, model_dim]
        (useful for hybrid model — see SpectralStateSpaceHybrid).
        """
        B, T_in, N = x_norm.shape
        features = []
        # Feature embedding
        features.append(self.input_proj(x_norm.unsqueeze(-1)))                # [B, T, N, d_f]
        # TOD embedding (broadcast over N)
        if self.tod_embedding is not None:
            tod_idx = (tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            tod_emb = self.tod_embedding(tod_idx)                              # [B, T, d_tod]
            features.append(tod_emb.unsqueeze(2).expand(B, T_in, N, -1))
        if self.dow_embedding is not None:
            dow_emb = self.dow_embedding(dow.long())                           # [B, T, d_dow]
            features.append(dow_emb.unsqueeze(2).expand(B, T_in, N, -1))
        if self.spatial_embedding is not None:
            features.append(self.spatial_embedding.unsqueeze(0).unsqueeze(0)
                            .expand(B, T_in, N, -1))
        if self.adaptive_embedding is not None:
            features.append(self.adaptive_embedding.unsqueeze(0).expand(B, T_in, N, -1))

        h = torch.cat(features, dim=-1)                                        # [B, T, N, model_dim]

        # Temporal Transformer (dim=1 is time)
        for layer in self.temporal_layers:
            h = layer(h, dim=1)
        # Spatial Transformer (dim=2 is nodes)
        for layer in self.spatial_layers:
            h = layer(h, dim=2)

        return h                                                               # [B, T_in, N, model_dim]

    def forward(self, x_norm, tod, dow, y_tod=None, y_dow=None):
        """
        x_norm: [B, T_in, N]   (we accept y_tod/y_dow for API compatibility
        but STAEformer does not use them — it predicts T_out steps from T_in
        via a flat Linear over the encoder's pooled features.)
        """
        B, T_in, N = x_norm.shape
        h = self.get_hidden(x_norm, tod, dow)                                  # [B, T_in, N, D]
        # Mixed projection: rearrange to [B, N, T_in * D] -> Linear -> [B, N, T_out]
        out = h.transpose(1, 2).reshape(B, N, T_in * self.model_dim)
        out = self.output_proj(out)                                            # [B, N, T_out]
        out = out.transpose(1, 2)                                              # [B, T_out, N]
        return out


def build_staeformer(N: int = 207, **kwargs) -> STAEformer:
    """Convenience constructor with STAEformer's default METR-LA config."""
    return STAEformer(N=N, **kwargs)
