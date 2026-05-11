"""
v2 dataset: returns the windowed inputs needed for the Spectral State Space Model.

Each sample contains:
  x_node      [T_in, N]    raw (unnormalized) input speeds (used for residual / decoder context)
  x_norm      [T_in, N]    masked-normalized input speeds
  tod         [T_in]       time-of-day in [0, 1) for input window
  dow         [T_in]       day-of-week int for input window
  y_node      [T_out, N]   raw target speeds (loss is computed against this)
  y_mask      [T_out, N]   1 where y_node != 0 (valid reading)
  y_tod       [T_out]      future time-of-day (decoder conditioning)
  y_dow       [T_out]      future day-of-week
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SSSMDataset(Dataset):
    def __init__(self, X, X_norm, tod, dow, missing_mask, input_len=12, pred_len=12,
                 prior=None, prior_norm=None):
        """
        All inputs are full-length arrays; we slide windows over them.
        Pass already-split arrays (X_train_part, etc.) for split-specific datasets.
        prior / prior_norm are optional (calendar prior in raw / normalized space).
        """
        self.X = X
        self.X_norm = X_norm
        self.tod = tod
        self.dow = dow
        self.missing = missing_mask
        self.input_len = input_len
        self.pred_len = pred_len
        self.prior = prior
        self.prior_norm = prior_norm

    def __len__(self):
        return max(0, len(self.X) - self.input_len - self.pred_len + 1)

    def __getitem__(self, idx):
        i, j = idx, idx + self.input_len
        k = j + self.pred_len
        sample = {
            "x_node": torch.from_numpy(self.X[i:j]),                     # [T_in, N]
            "x_norm": torch.from_numpy(self.X_norm[i:j]),                # [T_in, N]
            "tod":    torch.from_numpy(self.tod[i:j]),                   # [T_in]
            "dow":    torch.from_numpy(self.dow[i:j]),                   # [T_in]
            "y_node": torch.from_numpy(self.X[j:k]),                     # [T_out, N]
            "y_mask": torch.from_numpy(self.missing[j:k]),               # [T_out, N]
            "y_tod":  torch.from_numpy(self.tod[j:k]),                   # [T_out]
            "y_dow":  torch.from_numpy(self.dow[j:k]),                   # [T_out]
        }
        if self.prior_norm is not None:
            sample["prior_norm_in"]  = torch.from_numpy(self.prior_norm[i:j])  # [T_in, N]
            sample["prior_norm_out"] = torch.from_numpy(self.prior_norm[j:k])  # [T_out, N]
        if self.prior is not None:
            sample["prior_out"] = torch.from_numpy(self.prior[j:k])           # [T_out, N]
        return sample


def split_train_val_test(arrays, train_frac=0.7, val_frac=0.1):
    """Slice each array along axis 0 into train / val / test parts."""
    T = len(arrays[0])
    n_train = int(train_frac * T)
    n_val = int(val_frac * T)
    out = []
    for a in arrays:
        out.append((a[:n_train], a[n_train:n_train + n_val], a[n_train + n_val:]))
    return out


# ---------------------------------------------------------------------------
# Multi-window dataset (for v7+): for each sample at prediction start t0,
# return recent, day-ago, week-ago input windows.
# ---------------------------------------------------------------------------


class MultiWindowSSSMDataset(Dataset):
    """
    Multi-window dataset. Holds the FULL (un-split) arrays and indexes
    samples by their prediction-start t0. Input windows are looked up from
    history using fixed offsets:

      recent:    X[t0-12 : t0]
      day_ago:   X[t0-12-288 : t0-288]
      week_ago:  X[t0-12-2016 : t0-2016]
      target:    X[t0 : t0+12]

    Each split is defined by a (t0_start, t0_end) range over valid t0.
    Minimum t0 is 12+2016 = 2028 (need 7-day history + input window).
    """

    DAILY_OFFSET = 288
    WEEKLY_OFFSET = 7 * 288

    def __init__(self, X, X_norm, tod, dow, missing_mask,
                 t0_start, t0_end,
                 input_len=12, pred_len=12,
                 prior=None, prior_norm=None,
                 use_daily=True, use_weekly=True):
        self.X = X
        self.X_norm = X_norm
        self.tod = tod
        self.dow = dow
        self.missing = missing_mask
        self.input_len = input_len
        self.pred_len = pred_len
        self.t0_start = t0_start
        self.t0_end = t0_end
        self.prior = prior
        self.prior_norm = prior_norm
        self.use_daily = use_daily
        self.use_weekly = use_weekly

    def __len__(self):
        return max(0, self.t0_end - self.t0_start)

    def _slice(self, arr, a, b):
        return torch.from_numpy(arr[a:b])

    def __getitem__(self, idx):
        t0 = self.t0_start + idx
        L = self.input_len
        D = self.DAILY_OFFSET
        W = self.WEEKLY_OFFSET
        P = self.pred_len

        # Recent
        r_lo, r_hi = t0 - L, t0
        d_lo, d_hi = t0 - L - D, t0 - D
        w_lo, w_hi = t0 - L - W, t0 - W

        # Recent window
        x_norm_r = self._slice(self.X_norm, r_lo, r_hi)         # [L, N]
        tod_r = self._slice(self.tod, r_lo, r_hi)
        dow_r = self._slice(self.dow, r_lo, r_hi)
        win_id_r = torch.zeros(L, dtype=torch.long)              # window id 0
        x_norm = x_norm_r
        tod_in = tod_r
        dow_in = dow_r
        win_id = win_id_r

        if self.use_daily:
            x_norm_d = self._slice(self.X_norm, d_lo, d_hi)
            tod_d = self._slice(self.tod, d_lo, d_hi)
            dow_d = self._slice(self.dow, d_lo, d_hi)
            win_id_d = torch.ones(L, dtype=torch.long)           # window id 1
            x_norm = torch.cat([x_norm_d, x_norm], dim=0)
            tod_in = torch.cat([tod_d, tod_in], dim=0)
            dow_in = torch.cat([dow_d, dow_in], dim=0)
            win_id = torch.cat([win_id_d, win_id], dim=0)

        if self.use_weekly:
            x_norm_w = self._slice(self.X_norm, w_lo, w_hi)
            tod_w = self._slice(self.tod, w_lo, w_hi)
            dow_w = self._slice(self.dow, w_lo, w_hi)
            win_id_w = torch.full((L,), 2, dtype=torch.long)     # window id 2
            x_norm = torch.cat([x_norm_w, x_norm], dim=0)
            tod_in = torch.cat([tod_w, tod_in], dim=0)
            dow_in = torch.cat([dow_w, dow_in], dim=0)
            win_id = torch.cat([win_id_w, win_id], dim=0)

        sample = {
            "x_norm": x_norm,                                    # [L*K_wins, N]
            "tod":    tod_in,
            "dow":    dow_in,
            "win_id": win_id,
            "y_node": self._slice(self.X, t0, t0 + P),
            "y_mask": self._slice(self.missing, t0, t0 + P),
            "y_tod":  self._slice(self.tod, t0, t0 + P),
            "y_dow":  self._slice(self.dow, t0, t0 + P),
        }
        if self.prior_norm is not None:
            sample["prior_norm_out"] = self._slice(self.prior_norm, t0, t0 + P)
        return sample


def split_t0_range(T, train_frac=0.7, val_frac=0.1, min_t0=2028, max_t0_lookback=12):
    """
    Returns (train, val, test) tuples of (t0_start, t0_end) for use with
    MultiWindowSSSMDataset. min_t0 = input_len + weekly_offset = 12 + 2016.
    """
    t0_max = T - max_t0_lookback
    t0_train_end = int(train_frac * T)
    t0_val_end = int((train_frac + val_frac) * T)
    return (
        (max(min_t0, 0), t0_train_end),
        (t0_train_end, t0_val_end),
        (t0_val_end, t0_max),
    )
