import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import load_metr_la_h5, load_adj_pkl
from graph_utils import normalized_laplacian
from gft import compute_gft_basis, gft
from dataset import SpectralTrafficDataset
from model import SpectralGRU

# Configuration hyperparameters
class Config:
    input_len = 12
    pred_len = 12
    hidden_dim = 128
    k = 64
    lr = 1e-3
    batch_size = 64
    epochs = 20
    data_path = "data/METR-LA.h5"
    adj_path = "data/adj_METR-LA.pkl"
    checkpoint_dir = "checkpoints"


def compute_metrics(preds, labels):
    """
    Compute MAE, RMSE, and MAPE in the original node space.
    """
    mae = torch.abs(preds - labels).mean().item()
    rmse = torch.sqrt(((preds - labels) ** 2).mean()).item()
    mask = labels > 1e-4  # mask small values to avoid division by zero
    if mask.sum() > 0:
        mape = (torch.abs(preds[mask] - labels[mask]) / labels[mask]).mean().item()
    else:
        mape = 0.0
    return mae, rmse, mape


def main():
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    try:
        X = load_metr_la_h5(Config.data_path)  # [T, N]
        _, _, A = load_adj_pkl(Config.adj_path)
    except FileNotFoundError as e:
        print(f"Error loading dataset files. Ensure they exist at specified paths: {e}")
        return

    # 2. Normalize traffic values
    print("Normalizing data and building graph...")
    # Shape of mean and std will be [1, N]
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - mean) / std

    # 3. Build Laplacian and GFT basis
    # Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    L = normalized_laplacian(A)
    evals, U = compute_gft_basis(L, k=Config.k)

    # 4. Transform to spectral domain (GFT)
    print("Computing Graph Fourier Transform...")
    X_hat = gft(X_norm, U)  # [T, k]

    # 5. Make splits
    n_total = len(X_hat)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)

    X_train = X_hat[:n_train]
    X_val = X_hat[n_train:n_train + n_val]
    X_test = X_hat[n_train + n_val:]

    train_ds = SpectralTrafficDataset(X_train, input_len=Config.input_len, pred_len=Config.pred_len)
    val_ds = SpectralTrafficDataset(X_val, input_len=Config.input_len, pred_len=Config.pred_len)
    test_ds = SpectralTrafficDataset(X_test, input_len=Config.input_len, pred_len=Config.pred_len)

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False)

    # 6. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SpectralGRU(k=Config.k, hidden_dim=Config.hidden_dim, pred_len=Config.pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    loss_fn = torch.nn.L1Loss()  # Train in spectral space with L1 loss

    # Save inverse transform matrices onto the target device to enable fast metrics computation
    U_t = torch.tensor(U.T, dtype=torch.float32, device=device)  # [k, N]
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device) # [1, N]
    std_t = torch.tensor(std, dtype=torch.float32, device=device)   # [1, N]
    
    def evaluate(loader, split_name="Val"):
        model.eval()
        total_mae, total_rmse, total_mape = 0.0, 0.0, 0.0
        batches = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                
                # Forward pass (pred_hat is in spectral space)
                pred_hat = model(xb) # [B, pred_len, k]
                
                # Inverse GFT: X_rec_norm = X_hat @ U.T
                # Both pred_hat and yb are [B, pred_len, k]
                # U_t is [k, N] -> Output is [B, pred_len, N]
                pred_rec_norm = torch.matmul(pred_hat, U_t)
                y_rec_norm = torch.matmul(yb, U_t)
                
                # Reverse the Z-Score normalization to original node-space values
                pred_node = pred_rec_norm * std_t + mean_t
                y_node = y_rec_norm * std_t + mean_t
                
                mae, rmse, mape = compute_metrics(pred_node, y_node)
                total_mae += mae
                total_rmse += rmse
                total_mape += mape
                batches += 1
                
        if batches == 0:
            return 0.0, 0.0, 0.0
        return total_mae / batches, total_rmse / batches, total_mape / batches

    # 7. Training Loop
    print("Starting training...")
    best_val_mae = float('inf')
    for epoch in range(Config.epochs):
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

        val_mae, val_rmse, val_mape = evaluate(val_loader, "Val")
        print(f"Epoch {epoch+1:02d} | Train L1 (spectral): {train_loss:.4f} | Val MAE (node): {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val MAPE: {val_mape:.4f}")
        
        # Save checkpoint if improvement is found
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(Config.checkpoint_dir, "best_model.pth"))
            print("  -> Saved new best model!")

    # 8. Testing Evaluation
    print("Testing best model...")
    model.load_state_dict(torch.load(os.path.join(Config.checkpoint_dir, "best_model.pth")))
    test_mae, test_rmse, test_mape = evaluate(test_loader, "Test")
    print(f"Final Test - MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | MAPE: {test_mape:.4f}")

if __name__ == "__main__":
    main()