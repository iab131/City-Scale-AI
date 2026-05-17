"""
Frozen-trunk wrapper around STAEformer.

Loads a pretrained STAEformer checkpoint, freezes all parameters by default,
and provides a clean `forward_base(x_norm, tod, dow)` that returns the
predicted normalized speeds Y_base in the same format the rest of DiSR-Mamba
expects.

Also supports "light joint fine-tuning" (Stage F) by selectively unfreezing
only the output projection and the last spatial layer.
"""

from __future__ import annotations
from typing import Optional, Sequence

import os
import sys
import torch
import torch.nn as nn

# import STAEformer from the project's models/ package
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.append(_PROJ_ROOT)

from models.staeformer import STAEformer  # noqa: E402


class STAEFrozenWrapper(nn.Module):
    """
    Wraps a STAEformer model. Default: fully frozen and in eval mode.
    Use `partial_unfreeze` to enable Stage F light fine-tuning.
    """

    def __init__(
        self,
        model: STAEformer,
        freeze: bool = True,
        keep_dropout_off: bool = True,
    ):
        super().__init__()
        self.staeformer = model
        self.keep_dropout_off = keep_dropout_off
        if freeze:
            for p in self.staeformer.parameters():
                p.requires_grad_(False)
            self.staeformer.eval()
        else:
            self.staeformer.train()
        self._frozen = freeze

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Keep the trunk in eval whenever it's frozen (and dropout is off)."""
        super().train(mode)
        if self._frozen and self.keep_dropout_off:
            self.staeformer.eval()
        return self

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: torch.device,
        freeze: bool = True,
    ) -> "STAEFrozenWrapper":
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ckpt["args"]
        model = STAEformer(
            N=a.get("N", 207),
            in_steps=a["in_steps"], out_steps=a["out_steps"],
            input_embedding_dim=a["input_embedding_dim"],
            tod_embedding_dim=a["tod_embedding_dim"],
            dow_embedding_dim=a["dow_embedding_dim"],
            adaptive_embedding_dim=a["adaptive_embedding_dim"],
            feed_forward_dim=a["feed_forward_dim"],
            num_heads=a["num_heads"], num_layers=a["num_layers"],
            dropout=a["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model"])
        return cls(model, freeze=freeze)

    # ------------------------------------------------------------------
    def partial_unfreeze(self, last_n_spatial: int = 1, include_output: bool = True):
        """
        Stage F: unfreeze only:
          - the output projection (`output_proj`)
          - the last `last_n_spatial` spatial-transformer layers

        Everything else stays frozen.
        """
        self._frozen = False
        for p in self.staeformer.parameters():
            p.requires_grad_(False)
        if include_output:
            for p in self.staeformer.output_proj.parameters():
                p.requires_grad_(True)
        if last_n_spatial > 0:
            for layer in self.staeformer.spatial_layers[-last_n_spatial:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    # ------------------------------------------------------------------
    def forward_base(self, x_norm: torch.Tensor, tod: torch.Tensor,
                     dow: torch.Tensor) -> torch.Tensor:
        """
        Returns STAEformer's normalized prediction Y_base ∈ [B, T_out, N].
        Runs without grad if the trunk is frozen.
        """
        if self._frozen:
            with torch.no_grad():
                return self.staeformer(x_norm, tod, dow)
        return self.staeformer(x_norm, tod, dow)

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.forward_base(*args, **kwargs)
