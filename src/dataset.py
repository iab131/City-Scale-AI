import numpy as np
import torch
from torch.utils.data import Dataset


class SpectralTrafficDataset(Dataset):
    def __init__(self, X_hat: np.ndarray, input_len: int = 12, pred_len: int = 12):
        """
        Args:
            X_hat: [T, k] spectral coefficients
        """
        self.X_hat = X_hat
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X_hat) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.X_hat[idx: idx + self.input_len]  # [input_len, k]
        y = self.X_hat[idx + self.input_len: idx + self.input_len + self.pred_len]  # [pred_len, k]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )