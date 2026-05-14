"""
STAEformer with frozen pretrained STMAE encoder(s) bolted on.

The pretrained encoder(s) take long history [B, N, T_long] → [B, N, P, d].
We take the *last patch* representation [B, N, d] (the most-recent-week summary
per node) and feed it as an additional per-node feature stream into
STAEformer's embedding concat. The pretrained encoder weights are loaded
and frozen; only STAEformer itself + an adapter Linear are trained.

Inputs at finetuning time:
  x_norm     [B, T_in=12, N]
  tod        [B, T_in]
  dow        [B, T_in]
  long_hist  [B, N, T_long=2016]   normalized speeds for the preceding 1 week

Output:
  y_norm     [B, T_out=12, N]
"""

from __future__ import annotations
import torch
import torch.nn as nn

from models.staeformer import STAEformer, SelfAttentionLayer
from models.stmae import TMAE, SMAE


class STAEformerPretrained(nn.Module):
    """
    Composition:
      - tmae (frozen)   long_hist → [B, N, d_tmae] (last patch)
      - smae (frozen)   long_hist → [B, N, d_smae] (last patch)  [optional]
      - adapter         [B, N, d_tmae + d_smae] → [B, N, d_pre] via Linear
      - h_pre then broadcast over T_in and concatenated to STAEformer's
        feature/TOD/DOW/adaptive embeddings → model_dim + d_pre
      - 3× temporal + 3× spatial Transformer over [B, T_in, N, M+d_pre]
      - Mixed projection: flatten T_in*(M+d_pre) → Linear → T_out
    """

    def __init__(
        self,
        N: int,
        tmae_ckpt_path: str = None,
        smae_ckpt_path: str = None,
        # STAEformer dims (default per CIKM'23 METR-LA)
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
        # Pretrained adapter
        d_pre: int = 32,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
        self.N = N
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.freeze_pretrained = freeze_pretrained
        self.use_tmae = tmae_ckpt_path is not None
        self.use_smae = smae_ckpt_path is not None
        assert self.use_tmae or self.use_smae, "must provide at least one pretrained encoder"

        # ---- Load pretrained encoder(s) ----
        d_combined = 0
        if self.use_tmae:
            ck = torch.load(tmae_ckpt_path, map_location="cpu", weights_only=False)
            a = ck["args"]
            self.tmae = TMAE(
                patch_size=a["patch_size"], max_patches=a["max_patches"],
                embed_dim=a["embed_dim"], num_heads=a["num_heads"],
                ffn_mult=a["ffn_mult"],
                encoder_depth=a["encoder_depth"], decoder_depth=a["decoder_depth"],
                mask_ratio=a["mask_ratio"], dropout=a["dropout"],
            )
            self.tmae.load_state_dict(ck["model"])
            if freeze_pretrained:
                for p in self.tmae.parameters():
                    p.requires_grad = False
                self.tmae.eval()
            d_combined += a["embed_dim"]
            self.tmae_d = a["embed_dim"]

        if self.use_smae:
            ck = torch.load(smae_ckpt_path, map_location="cpu", weights_only=False)
            a = ck["args"]
            self.smae = SMAE(
                N=N,
                patch_size=a["patch_size"], max_patches=a["max_patches"],
                embed_dim=a["embed_dim"], num_heads=a["num_heads"],
                ffn_mult=a["ffn_mult"],
                encoder_depth=a["encoder_depth"], decoder_depth=a["decoder_depth"],
                mask_ratio=a["mask_ratio"], dropout=a["dropout"],
            )
            self.smae.load_state_dict(ck["model"])
            if freeze_pretrained:
                for p in self.smae.parameters():
                    p.requires_grad = False
                self.smae.eval()
            d_combined += a["embed_dim"]
            self.smae_d = a["embed_dim"]

        # Adapter from pretrained embeddings to d_pre per node
        self.pre_adapter = nn.Linear(d_combined, d_pre)
        self.d_pre = d_pre

        # ---- STAEformer-style components, but with widened model_dim ----
        self.model_dim = (
            input_embedding_dim + tod_embedding_dim + dow_embedding_dim
            + adaptive_embedding_dim + d_pre
        )
        self.steps_per_day = steps_per_day

        self.input_proj = nn.Linear(1, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim) if tod_embedding_dim else None
        self.dow_embedding = nn.Embedding(num_dow, dow_embedding_dim) if dow_embedding_dim else None
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

    def _encode_long(self, long_hist: torch.Tensor) -> torch.Tensor:
        """long_hist: [B, N, T_long] → [B, N, d_combined] (BEFORE adapter)"""
        feats = []
        if self.use_tmae:
            with torch.no_grad() if self.freeze_pretrained else torch.enable_grad():
                h, _, _ = self.tmae.encode(long_hist, mask=False)   # [B, N, P, d]
                last = h[:, :, -1, :]                                # [B, N, d_tmae]
            feats.append(last)
        if self.use_smae:
            with torch.no_grad() if self.freeze_pretrained else torch.enable_grad():
                h, _, _ = self.smae.encode(long_hist, mask=False)   # [B, N, P, d]
                last = h[:, :, -1, :]                                # [B, N, d_smae]
            feats.append(last)
        return torch.cat(feats, dim=-1)                              # [B, N, d_combined]

    def encode_pre(self, long_hist: torch.Tensor) -> torch.Tensor:
        """Public encoder forward — returns the COMBINED encoder output BEFORE
        the adapter. Used for precomputation: cache this output once across
        epochs, then call `forward_with_pre` instead of forward."""
        return self._encode_long(long_hist)

    def forward_with_pre(self, x_norm, tod, dow, pre_combined, y_tod=None, y_dow=None):
        """Same as forward() but `pre_combined` is the cached encoder output
        [B, N, d_combined] from a previous call to encode_pre(); we just apply
        the trainable adapter + STAEformer."""
        B, T_in, N = x_norm.shape
        h_pre = self.pre_adapter(pre_combined)                       # [B, N, d_pre]
        return self._stae_forward(x_norm, tod, dow, h_pre)

    def _stae_forward(self, x_norm, tod, dow, h_pre):
        """Internal: STAEformer-style forward given an already-computed h_pre."""
        B, T_in, N = x_norm.shape
        feats = [self.input_proj(x_norm.unsqueeze(-1))]               # [B, T, N, d_f]
        if self.tod_embedding is not None:
            tod_idx = (tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            tod_emb = self.tod_embedding(tod_idx)
            feats.append(tod_emb.unsqueeze(2).expand(B, T_in, N, -1))
        if self.dow_embedding is not None:
            dow_emb = self.dow_embedding(dow.long())
            feats.append(dow_emb.unsqueeze(2).expand(B, T_in, N, -1))
        feats.append(self.adaptive_embedding.unsqueeze(0).expand(B, T_in, N, -1))
        feats.append(h_pre.unsqueeze(1).expand(B, T_in, N, self.d_pre))

        h = torch.cat(feats, dim=-1)
        for layer in self.temporal_layers:
            h = layer(h, dim=1)
        for layer in self.spatial_layers:
            h = layer(h, dim=2)
        out = h.transpose(1, 2).reshape(B, N, T_in * self.model_dim)
        out = self.output_proj(out)
        return out.transpose(1, 2)

    def forward(self, x_norm, tod, dow, long_hist, y_tod=None, y_dow=None):
        """Standard forward: re-encodes the long history every call.
        Use forward_with_pre() with cached encode_pre() output for speed."""
        pre_combined = self._encode_long(long_hist)                   # [B, N, d_combined]
        h_pre = self.pre_adapter(pre_combined)                        # [B, N, d_pre]
        return self._stae_forward(x_norm, tod, dow, h_pre)

        return self._stae_forward(x_norm, tod, dow, h_pre)
