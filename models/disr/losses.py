"""
Loss functions for DiSR-Mamba.

All metrics use the project's standard masked convention: the mask is 1.0 at
valid (non-zero) sensor readings and 0.0 at missing, and the loss is
normalized by the mean of the mask so it remains scale-invariant to the
fraction of missing entries.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def masked_mae(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    m_mean = mask.mean().clamp(min=eps)
    return ((pred - true).abs() * mask).mean() / m_mean


def masked_rmse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    m_mean = mask.mean().clamp(min=eps)
    return torch.sqrt(((pred - true) ** 2 * mask).mean() / m_mean)


def masked_mape(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    m = mask * (true.abs() > 1e-3).float()
    m_mean = m.mean().clamp(min=eps)
    return ((pred - true).abs() / true.abs().clamp(min=eps) * m).mean() / m_mean


def per_horizon_metrics(pred: torch.Tensor, true: torch.Tensor,
                        mask: torch.Tensor) -> dict:
    """Standard per-horizon table (15/30/60 min + avg)."""
    out = {
        "avg_mae":  masked_mae(pred, true, mask).item(),
        "avg_rmse": masked_rmse(pred, true, mask).item(),
        "avg_mape": masked_mape(pred, true, mask).item(),
    }
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p_t, y_t, m_t = pred[:, t:t+1], true[:, t:t+1], mask[:, t:t+1]
            out[f"mae_{tag}"]  = masked_mae(p_t, y_t, m_t).item()
            out[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            out[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return out


def per_horizon_mae(pred: torch.Tensor, true: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Returns MAE tensor of shape [T_out] (per-horizon)."""
    diff = (pred - true).abs() * mask
    # mean across batch and nodes; per-horizon
    m_mean = mask.mean(dim=(0, 2)).clamp(min=1e-6)
    return diff.mean(dim=(0, 2)) / m_mean


def per_speed_regime_mae(pred: torch.Tensor, true: torch.Tensor,
                         mask: torch.Tensor) -> dict:
    """
    Reports masked MAE for four speed buckets defined on the ground-truth:
      < 20 mph, 20-40 mph, 40-60 mph, >= 60 mph.
    """
    buckets = [
        ("lt20",  true < 20.0),
        ("20_40", (true >= 20.0) & (true < 40.0)),
        ("40_60", (true >= 40.0) & (true < 60.0)),
        ("ge60",  true >= 60.0),
    ]
    out = {}
    for name, sel in buckets:
        m = mask * sel.float()
        m_sum = m.sum().clamp(min=1.0)
        out[f"mae_{name}"] = float(((pred - true).abs() * m).sum() / m_sum)
        out[f"count_{name}"] = int(m.sum().item())
    return out


def congestion_mask(
    y_true: torch.Tensor,
    x_recent: Optional[torch.Tensor] = None,
    speed_thr: float = 20.0,
    delta_thr: float = 5.0,
) -> torch.Tensor:
    """
    Build a soft selection mask for "hard" regions used by congestion-aware
    loss. Returns the union of:
      - low-speed mask: y_true < speed_thr
      - high-volatility mask: |y_true - x_recent_last| > delta_thr (if x_recent given)

    `x_recent` should be the input window's last time step broadcast over T_out,
    in the same (raw mph) units as y_true. Shape [B, T_out, N] or [B, 1, N].
    """
    low_speed = (y_true < speed_thr).float()
    if x_recent is None:
        return low_speed
    if x_recent.dim() == 3 and x_recent.shape[1] == 1:
        x_recent = x_recent.expand_as(y_true)
    elif x_recent.dim() == 2:
        x_recent = x_recent.unsqueeze(1).expand_as(y_true)
    volatility = ((y_true - x_recent).abs() > delta_thr).float()
    return torch.clamp(low_speed + volatility, max=1.0)


def disr_composite_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_base: torch.Tensor,
    delta_y: torch.Tensor,
    y_mask: torch.Tensor,
    x_recent: Optional[torch.Tensor] = None,
    lambda_res: float = 0.2,
    lambda_cong: float = 0.1,
    speed_thr: float = 20.0,
    delta_thr: float = 5.0,
    router_entropy: Optional[torch.Tensor] = None,
    lambda_entropy: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Composite training loss:

        L_main = masked_MAE(y_pred,  y_true)
        L_res  = masked_MAE(delta_y, y_true - y_base)
        L_cong = masked_MAE(y_pred, y_true) over congestion mask
        L      = L_main + λ_res * L_res + λ_cong * L_cong + λ_ent * (-H(router))

    All inputs in raw mph except delta_y which is in raw mph as well (the model
    de-normalizes inside its forward).
    """
    L_main = masked_mae(y_pred, y_true, y_mask)
    residual_target = y_true - y_base
    L_res = masked_mae(delta_y, residual_target, y_mask)

    cong = congestion_mask(y_true, x_recent=x_recent,
                           speed_thr=speed_thr, delta_thr=delta_thr)
    cong_mask = y_mask * cong
    cm_sum = cong_mask.sum().clamp(min=1.0)
    L_cong = ((y_pred - y_true).abs() * cong_mask).sum() / cm_sum

    L = L_main + lambda_res * L_res + lambda_cong * L_cong

    parts = {
        "L_main": float(L_main.detach()),
        "L_res":  float(L_res.detach()),
        "L_cong": float(L_cong.detach()),
        "cong_frac": float(cong.mean().detach()),
    }
    if router_entropy is not None and lambda_entropy > 0.0:
        # router_entropy is negative entropy of the expert mixture; we want
        # high entropy at the start to avoid collapse, so we subtract it.
        L = L - lambda_entropy * router_entropy
        parts["router_entropy"] = float(router_entropy.detach())
    return L, parts
