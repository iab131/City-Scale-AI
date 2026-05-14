"""
STMAE — Spatial-Temporal Masked AutoEncoder for pretraining (vendored).

Inspired by STD-MAE (IJCAI 2024) and STMAE (CIKM 2024), but implemented
directly on our preprocess_v2 pipeline so we sidestep the easytorch /
torchvision dependency hell that derailed the previous attempt.

Two encoders are trained separately:
  - TMAE: long-term temporal context. Mask 75% of *time patches* per node,
          reconstruct from a per-node Transformer over the remaining patches.
  - SMAE: long-term spatial context. Mask 75% of *nodes* per time patch,
          reconstruct from a per-time-patch Transformer over remaining nodes.

For finetuning, the (frozen) encoders are bolted onto STAEformer as
additional per-node feature streams. See models/staeformer_pretrained.py.

Patch convention:
  Long history shape: [B, N, T_long]   (T_long = patch_size * num_patches,
                                        typically 12 * 168 = 2016 ≈ 1 week)
  Patches:            [B, N, P, patch_size]
  Embedded:           [B, N, P, d]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Linear embed each contiguous `patch_size` slice into `embed_dim`."""

    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        """
        x: [B, N, T_long]   T_long must be divisible by patch_size
        returns: [B, N, P, d]
        """
        B, N, T = x.shape
        P = T // self.patch_size
        assert T == P * self.patch_size, f"T={T} not divisible by patch_size={self.patch_size}"
        x = x.view(B, N, P, self.patch_size)
        return self.proj(x)


class SinCosPosEmb(nn.Module):
    """Fixed sin/cos positional encoding. Cached for given max_len."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                        * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, n_pos: int):
        return self.pe[:n_pos]


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block (GELU, no bias on LN, dropout)."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, D]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# ----------------------------------------------------------------------------
# TMAE — Temporal Masked AutoEncoder
# ----------------------------------------------------------------------------


class TMAE(nn.Module):
    """
    Temporal masked autoencoder.

    Each node is processed independently. For input [B, N, T_long]:
      1. Patchify into [B, N, P, patch_size] then embed to [B, N, P, d].
      2. Add temporal positional embedding (shared across nodes).
      3. Randomly mask M = floor(P * mask_ratio) patches (same per-sample mask).
      4. Encoder Transformer runs on visible patches → [B, N, P_vis, d].
      5. Decoder receives encoder output + learnable mask tokens (placed at
         masked positions, with positional embedding) → reconstruction
         logits [B, N, P, patch_size].
      6. Loss is MAE on reconstructed *masked* patches only (in normalized space).
    """

    def __init__(
        self,
        patch_size: int = 12,
        max_patches: int = 168,            # 12 * 168 = 2016 ≈ 1 week
        embed_dim: int = 96,
        num_heads: int = 4,
        ffn_mult: int = 4,
        encoder_depth: int = 4,
        decoder_depth: int = 1,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.pos_emb = SinCosPosEmb(max_patches, embed_dim)

        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_mult, dropout)
            for _ in range(encoder_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.enc2dec = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_mult, dropout)
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(embed_dim)
        self.reconstruct = nn.Linear(embed_dim, patch_size)

    def _random_mask(self, P: int, device, generator=None):
        """
        Returns (vis_idx, masked_idx) — both lists of indices into [0, P).
        Same mask applied to every (B, N) instance in the batch (per-batch
        mask, not per-node) — this matches STD-MAE's convention.
        """
        n_mask = int(P * self.mask_ratio)
        perm = torch.randperm(P, device=device, generator=generator)
        return perm[n_mask:].sort().values, perm[:n_mask].sort().values

    def encode(self, x, mask: bool = True, generator=None):
        """
        x: [B, N, T_long]
        returns:
          h_full or h_vis: encoded patches
          vis_idx, masked_idx: index tensors
        """
        B, N, T = x.shape
        h = self.patch_embed(x)                            # [B, N, P, d]
        B, N, P, D = h.shape
        pos = self.pos_emb(P).view(1, 1, P, D).to(h.dtype) # [1, 1, P, d]
        h = h + pos

        if mask:
            vis_idx, masked_idx = self._random_mask(P, x.device, generator)
            h_vis = h[:, :, vis_idx, :]                     # [B, N, P_vis, d]
            # Run encoder per node — fold N into batch
            P_vis = h_vis.shape[2]
            h_vis = h_vis.reshape(B * N, P_vis, D)
            for layer in self.encoder:
                h_vis = layer(h_vis)
            h_vis = self.enc_norm(h_vis).view(B, N, P_vis, D)
            return h_vis, vis_idx, masked_idx
        else:
            # No masking — used at inference / for downstream tasks
            h_full = h.reshape(B * N, P, D)
            for layer in self.encoder:
                h_full = layer(h_full)
            h_full = self.enc_norm(h_full).view(B, N, P, D)
            return h_full, None, None

    def decode(self, h_vis, vis_idx, masked_idx, P: int):
        """
        h_vis:      [B, N, P_vis, d]
        returns:    [B, N, P, patch_size] reconstruction
        """
        B, N, P_vis, D = h_vis.shape
        h_vis = self.enc2dec(h_vis)
        pos = self.pos_emb(P).view(1, 1, P, D).to(h_vis.dtype)

        # Place visible tokens at their positions, mask tokens at masked positions
        h_full = torch.zeros(B, N, P, D, device=h_vis.device, dtype=h_vis.dtype)
        h_full[:, :, vis_idx, :] = h_vis + pos[:, :, vis_idx, :]
        h_full[:, :, masked_idx, :] = self.mask_token.to(h_vis.dtype).view(1, 1, 1, D) + pos[:, :, masked_idx, :]

        # Decoder per-node
        h_full = h_full.reshape(B * N, P, D)
        for layer in self.decoder:
            h_full = layer(h_full)
        h_full = self.dec_norm(h_full).view(B, N, P, D)
        return self.reconstruct(h_full)                     # [B, N, P, patch_size]

    def forward(self, x):
        """Pretraining forward — returns reconstruction (only masked tokens),
        the ground-truth masked targets, and the masked indices.
        x: [B, N, T_long]
        """
        h_vis, vis_idx, masked_idx = self.encode(x, mask=True)
        P = x.shape[2] // self.patch_size
        recon = self.decode(h_vis, vis_idx, masked_idx, P)  # [B, N, P, patch_size]
        # Ground truth: re-patchify the input
        B, N, T = x.shape
        gt = x.view(B, N, P, self.patch_size)
        recon_m = recon[:, :, masked_idx, :]                 # [B, N, M, patch_size]
        gt_m = gt[:, :, masked_idx, :]
        return recon_m, gt_m, masked_idx


# ----------------------------------------------------------------------------
# SMAE — Spatial Masked AutoEncoder (mask nodes instead of time patches)
# ----------------------------------------------------------------------------


class SMAE(nn.Module):
    """
    Spatial masked autoencoder. For input [B, N, T_long]:
      1. Patchify time → [B, N, P, patch_size] then embed to [B, N, P, d].
      2. Add positional embedding along *node* axis (learnable) AND temporal.
      3. For each time patch p, mask 75% of the N nodes uniformly (same mask
         across the P patches per sample).
      4. Encoder runs Transformer over the *visible nodes* per time patch
         → [B, N_vis, P, d].
      5. Decoder fills mask tokens, applies Transformer, reconstructs
         patch values for all nodes → [B, N, P, patch_size].
    """

    def __init__(
        self,
        N: int,
        patch_size: int = 12,
        max_patches: int = 168,
        embed_dim: int = 96,
        num_heads: int = 4,
        ffn_mult: int = 4,
        encoder_depth: int = 4,
        decoder_depth: int = 1,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.N = N
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_patches = max_patches
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.pos_emb_t = SinCosPosEmb(max_patches, embed_dim)
        # Learnable per-node embedding (acts like positional encoding for nodes)
        self.pos_emb_n = nn.Parameter(torch.zeros(N, embed_dim))
        nn.init.normal_(self.pos_emb_n, std=0.02)

        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_mult, dropout)
            for _ in range(encoder_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.enc2dec = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_mult, dropout)
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(embed_dim)
        self.reconstruct = nn.Linear(embed_dim, patch_size)

    def _random_node_mask(self, device, generator=None):
        n_mask = int(self.N * self.mask_ratio)
        perm = torch.randperm(self.N, device=device, generator=generator)
        return perm[n_mask:].sort().values, perm[:n_mask].sort().values

    def encode(self, x, mask: bool = True, generator=None):
        """
        x: [B, N, T_long]
        returns: h_vis [B, N_vis, P, d], vis_node_idx, masked_node_idx
        """
        B, N, T = x.shape
        h = self.patch_embed(x)                            # [B, N, P, d]
        P, D = h.shape[2], h.shape[3]
        dt = h.dtype
        h = h + self.pos_emb_t(P).view(1, 1, P, D).to(dt) + self.pos_emb_n.to(dt).view(1, N, 1, D)

        if mask:
            vis_idx, masked_idx = self._random_node_mask(x.device, generator)
            h_vis = h[:, vis_idx, :, :]                     # [B, N_vis, P, d]
            # Encoder: attention over visible nodes per time patch — fold P into batch
            N_vis = h_vis.shape[1]
            h_vis = h_vis.permute(0, 2, 1, 3).reshape(B * P, N_vis, D)
            for layer in self.encoder:
                h_vis = layer(h_vis)
            h_vis = self.enc_norm(h_vis)
            h_vis = h_vis.view(B, P, N_vis, D).permute(0, 2, 1, 3)  # [B, N_vis, P, d]
            return h_vis, vis_idx, masked_idx
        else:
            # Encode full unmasked input
            h_full = h.permute(0, 2, 1, 3).reshape(B * P, N, D)
            for layer in self.encoder:
                h_full = layer(h_full)
            h_full = self.enc_norm(h_full).view(B, P, N, D).permute(0, 2, 1, 3)
            return h_full, None, None

    def decode(self, h_vis, vis_idx, masked_idx, P: int):
        """h_vis: [B, N_vis, P, d] → reconstruction [B, N, P, patch_size]"""
        B, N_vis, _, D = h_vis.shape
        N = self.N
        h_vis = self.enc2dec(h_vis)
        dt = h_vis.dtype
        pos_t = self.pos_emb_t(P).view(1, 1, P, D).to(dt)
        pos_n = self.pos_emb_n.to(dt)
        mtok = self.mask_token.to(dt)

        h_full = torch.zeros(B, N, P, D, device=h_vis.device, dtype=dt)
        h_full[:, vis_idx, :, :] = h_vis + pos_t + pos_n[vis_idx].view(1, -1, 1, D)
        h_full[:, masked_idx, :, :] = (
            mtok.view(1, 1, 1, D)
            + pos_t
            + pos_n[masked_idx].view(1, -1, 1, D)
        )

        h_full = h_full.permute(0, 2, 1, 3).reshape(B * P, N, D)
        for layer in self.decoder:
            h_full = layer(h_full)
        h_full = self.dec_norm(h_full).view(B, P, N, D).permute(0, 2, 1, 3)
        return self.reconstruct(h_full)                     # [B, N, P, patch_size]

    def forward(self, x):
        h_vis, vis_idx, masked_idx = self.encode(x, mask=True)
        P = x.shape[2] // self.patch_size
        recon = self.decode(h_vis, vis_idx, masked_idx, P)
        B, N, T = x.shape
        gt = x.view(B, N, P, self.patch_size)
        recon_m = recon[:, masked_idx, :, :]                 # [B, N_masked, P, patch_size]
        gt_m = gt[:, masked_idx, :, :]
        return recon_m, gt_m, masked_idx
