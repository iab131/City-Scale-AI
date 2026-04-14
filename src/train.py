import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_utils import load_metr_la_h5, load_adj_pkl
from graph_utils import normalized_laplacian
from gft import compute_gft_basis, gft
from dataset import SpectralTrafficDataset
from model import SpectralGRU


def main():
    # 1. Load data
    X = load_metr_la_h5("data/metr_la/metr-la.h5")  # [T, N]
    _, _, A = load_adj_pkl("data/metr_la/adj_METR-LA.pkl")

    # 2. Normalize traffic values
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    # 3. Build Laplacian and GFT basis
    L = normalized_laplacian(A)
    k = 64
    evals, U = compute_gft_basis(L, k=k)

    # 4. Transform to spectral domain
    X_hat = gft(X_norm, U)  # [T, k]

    # 5. Split
    n_total = len(X_hat)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)

    X_train = X_hat[:n_train]
    X_val = X_hat[n_train:n_train + n_val]
    X_test = X_hat[n_train + n_val:]

    train_ds = SpectralTrafficDataset(X_train, input_len=12, pred_len=12)
    val_ds = SpectralTrafficDataset(X_val, input_len=12, pred_len=12)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 6. Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpectralGRU(k=k, hidden_dim=128, pred_len=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()

    # 7. Train
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()

        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch+1:02d} | train={train_loss:.4f} | val={val_loss:.4f}")


if __name__ == "__main__":
    main()