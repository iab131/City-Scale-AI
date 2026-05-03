import numpy as np
import torch
from torch.utils.data import Dataset


class SpectralTrafficDataset(Dataset):
    def __init__(self, X_hat: np.ndarray, X_node: np.ndarray = None, input_len: int = 12, pred_len: int = 12):
        """
        Args:
            X_hat: [T, k] spectral coefficients
            X_node: [T, N] original node features
        """
        self.X_hat = X_hat
        self.X_node = X_node
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X_hat) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.X_hat[idx: idx + self.input_len]  # [input_len, k]
        if self.X_node is not None:
            y = self.X_node[idx + self.input_len: idx + self.input_len + self.pred_len]  # [pred_len, N]
        else:
            y = self.X_hat[idx + self.input_len: idx + self.input_len + self.pred_len]  # [pred_len, k]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )