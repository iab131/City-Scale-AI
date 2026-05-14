"""
Long-history dataset for STMAE pretraining.

Each sample is a single contiguous slice of length T_long from the training
slice of the full timeseries. Slices are independent windows (no targets);
the masked autoencoder learns to reconstruct masked patches.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LongHistoryDataset(Dataset):
    """
    Yields windows of length T_long from the input array, in NODE-FIRST shape
    [N, T_long]. Used for STMAE pretraining where the model sees each window
    as a "long history" to reconstruct.

    Args:
        X_norm: [T, N] normalized speeds (already masked-normalized).
        t_start, t_end: range of valid starting indices into the full array.
            For training slice this is typically [0, n_train - T_long].
        T_long: length of each sample window (typically 2016 = 1 week).
        stride: step between successive windows. 12 is a good default for
            METR-LA (5-min cadence) — yields ~2000 windows from a 70%-train
            slice, which is plenty for pretraining.
        missing_mask: [T, N] mask of valid (non-zero) readings; passed through.
    """

    def __init__(self, X_norm, t_start: int, t_end: int, T_long: int = 2016,
                 stride: int = 12, missing_mask=None):
        self.X = X_norm
        self.t_start = t_start
        self.t_end = t_end
        self.T_long = T_long
        self.stride = stride
        self.mask = missing_mask
        # number of valid starting indices
        n_starts = max(0, (t_end - T_long - t_start) // stride + 1)
        self.starts = np.arange(t_start, t_start + n_starts * stride, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        j = i + self.T_long
        x_long = self.X[i:j].T                              # [N, T_long]
        out = {"x_long": torch.from_numpy(np.ascontiguousarray(x_long))}
        if self.mask is not None:
            m_long = self.mask[i:j].T                       # [N, T_long]
            out["mask_long"] = torch.from_numpy(np.ascontiguousarray(m_long))
        return out
