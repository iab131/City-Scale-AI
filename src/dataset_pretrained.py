"""
Dataset that returns both:
  - the standard short-window inputs (x_norm, tod, dow, y_node, y_mask, y_tod, y_dow)
  - a long-history window [N, T_long] ending at the same point as x_norm

Works by holding the full unsplit X_norm/tod/dow/missing arrays and indexing
absolute timesteps. Each split (train/val/test) supplies a (t0_start, t0_end)
range over valid prediction-start indices.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainedSSSMDataset(Dataset):
    """
    For prediction-start t0, the sample consists of:
      x_node      X[t0 - T_in : t0]                    raw input speeds
      x_norm      X_norm[t0 - T_in : t0]               normalized
      tod, dow    same window
      y_node      X[t0 : t0 + T_out]                   targets in raw mph
      y_mask      missing[t0 : t0 + T_out]
      y_tod, y_dow corresponding future TOD/DOW
      long_hist   X_norm[t0 - T_long : t0].T           [N, T_long]   one week of context

    The valid range is [max(T_long, T_in), T - T_out].
    """

    def __init__(self, X, X_norm, tod, dow, missing_mask,
                 t0_start: int, t0_end: int,
                 T_in: int = 12, T_out: int = 12, T_long: int = 2016):
        self.X = X
        self.X_norm = X_norm
        self.tod = tod
        self.dow = dow
        self.missing = missing_mask
        self.T_in = T_in
        self.T_out = T_out
        self.T_long = T_long
        self.t0_start = t0_start
        self.t0_end = t0_end

    def __len__(self):
        return max(0, self.t0_end - self.t0_start)

    def __getitem__(self, idx):
        t0 = self.t0_start + idx
        i = t0 - self.T_in            # start of input window
        j = t0                         # end of input / start of target
        k = t0 + self.T_out            # end of target
        l = t0 - self.T_long           # start of long history

        x_norm_short = self.X_norm[i:j]                          # [T_in, N]
        long_hist = self.X_norm[l:j].T                            # [N, T_long]

        sample = {
            "x_node":   torch.from_numpy(np.ascontiguousarray(self.X[i:j])),
            "x_norm":   torch.from_numpy(np.ascontiguousarray(x_norm_short)),
            "tod":      torch.from_numpy(np.ascontiguousarray(self.tod[i:j])),
            "dow":      torch.from_numpy(np.ascontiguousarray(self.dow[i:j])),
            "y_node":   torch.from_numpy(np.ascontiguousarray(self.X[j:k])),
            "y_mask":   torch.from_numpy(np.ascontiguousarray(self.missing[j:k])),
            "y_tod":    torch.from_numpy(np.ascontiguousarray(self.tod[j:k])),
            "y_dow":    torch.from_numpy(np.ascontiguousarray(self.dow[j:k])),
            "long_hist": torch.from_numpy(np.ascontiguousarray(long_hist)),
        }
        return sample


def split_t0_for_pretrained(T: int, T_long: int, T_in: int, T_out: int,
                            train_frac: float = 0.7, val_frac: float = 0.1):
    """
    Returns (train_range, val_range, test_range) for use with PretrainedSSSMDataset.

    Constraints:
      - The earliest prediction start t0 must be >= max(T_long, T_in) so the
        long-history window has data to look at.
      - The latest t0 is T - T_out.
      - Splits are by t0 (prediction start), matching the convention used
        elsewhere (70% train / 10% val / 20% test of prediction targets).
    """
    t0_min = max(T_long, T_in)
    t0_max = T - T_out

    # Compute split boundaries by prediction-start t0
    n_train = int(train_frac * T)
    n_val = int(val_frac * T)
    train_end = n_train
    val_end = n_train + n_val

    return (
        (max(t0_min, T_in), min(t0_max, train_end)),
        (max(t0_min, train_end), min(t0_max, val_end)),
        (max(t0_min, val_end), t0_max),
    )
