import torch
import torch.nn as nn


class SpectralGRU(nn.Module):
    def __init__(self, k: int, hidden_dim: int = 128, pred_len: int = 12):
        super().__init__()
        self.k = k
        self.pred_len = pred_len
        self.gru = nn.GRU(input_size=k, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, pred_len * k)

    def forward(self, x):
        """
        x: [B, Tin, k]
        returns: [B, Tout, k]
        """
        _, h = self.gru(x)
        h = h[-1]  # [B, hidden_dim]
        out = self.head(h)  # [B, pred_len * k]
        out = out.view(x.size(0), self.pred_len, self.k)
        return out