import os
import sys
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add root directory and src directory to python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.dataset import SpectralTrafficDataset
from src.preprocess import get_cached_gft_data
from models.mamba_model import SpectralMambaReal

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba for Traffic Forecasting")
    parser.add_argument('--config', type=str, default='configs/mamba.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--k', type=int, default=64, help='Number of GFT components')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

import numpy as np

def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def compute_metrics(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    mae = torch.abs(preds - labels) * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    mae = torch.mean(mae)
    
    # Add a small epsilon to avoid division by zero for valid labels that are very small
    mape = torch.abs((preds - labels) / torch.clamp(labels, min=1e-4)) * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    mape = torch.mean(mape)
    
    rmse = torch.square(preds - labels) * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    rmse = torch.sqrt(torch.mean(rmse))
    
    return mae.item(), rmse.item(), mape.item()

class Config:
    input_len = 12
    pred_len = 12
    k = 64
    d_model = 64
    num_layers = 2
    d_state = 16
    d_conv = 4
    expand = 2
    dropout = 0.1
    batch_size = 16
    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 50
    patience = 10
    gradient_clip = 5.0
    use_amp = True

    data_path = "data/METR-LA.h5"
    adj_path = "data/adj_METR-LA.pkl"
    checkpoint_dir = "results/mamba/checkpoints"
    results_dir = "results/mamba"

def run_mamba_training(config, args):
    import random
    import numpy as np
    import datetime

    # Set seed
    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    config.checkpoint_dir = "results/mamba/k_sweep/checkpoints"
    config.results_dir = "results/mamba/k_sweep"
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Device configuration
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Device selected: CUDA")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Allocated VRAM before training: {allocated_vram:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Device selected: CPU")

    try:
        mean, std, L, evals, U, X_hat, X_raw = get_cached_gft_data(
            config.data_path, config.adj_path, config.k, cache_dir="cache/gft"
        )
    except FileNotFoundError as e:
        print(f"Error loading dataset files: {e}")
        return None

    # Split: 70% Train, 10% Val, 20% Test (matches existing train.py logic)
    n_total = len(X_hat)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)

    X_train = X_hat[:n_train]
    X_val = X_hat[n_train:n_train + n_val]
    X_test = X_hat[n_train + n_val:]

    X_raw_train = X_raw[:n_train]
    X_raw_val = X_raw[n_train:n_train + n_val]
    X_raw_test = X_raw[n_train + n_val:]

    train_ds = SpectralTrafficDataset(X_train, X_node=X_raw_train, input_len=config.input_len, pred_len=config.pred_len)
    val_ds = SpectralTrafficDataset(X_val, X_node=X_raw_val, input_len=config.input_len, pred_len=config.pred_len)
    test_ds = SpectralTrafficDataset(X_test, X_node=X_raw_test, input_len=config.input_len, pred_len=config.pred_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    try:
        model = SpectralMambaReal(
            k=config.k,
            pred_len=config.pred_len,
            d_model=config.d_model,
            num_layers=config.num_layers,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=config.dropout
        ).to(device)
    except ImportError as e:
        print(f"FAILED TO LOAD MODEL: {e}")
        return None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=(config.use_amp and device.type == 'cuda'))

    U_t = torch.tensor(U.T, dtype=torch.float32, device=device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    def evaluate(loader):
        model.eval()
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for xb, yb_node in loader:
                xb, yb_node = xb.to(device), yb_node.to(device)

                with torch.cuda.amp.autocast(enabled=(config.use_amp and device.type == 'cuda')):
                    pred_hat = model(xb)

                pred_node = torch.matmul(pred_hat.float(), U_t) * std_t + mean_t
                preds_list.append(pred_node.cpu())
                labels_list.append(yb_node.cpu())

        preds = torch.cat(preds_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        metrics = {}
        mae, rmse, mape = compute_metrics(preds, labels)
        metrics['avg_mae'] = mae
        metrics['avg_rmse'] = rmse
        metrics['avg_mape'] = mape
        
        for t, m in [(2, 15), (5, 30), (11, 60)]:
            if preds.shape[1] > t:
                mae_t, rmse_t, mape_t = compute_metrics(preds[:, t:t+1, :], labels[:, t:t+1, :])
                metrics[f'mae_{m}'] = mae_t
                metrics[f'rmse_{m}'] = rmse_t
                metrics[f'mape_{m}'] = mape_t
            else:
                metrics[f'mae_{m}'] = 0.0
                metrics[f'rmse_{m}'] = 0.0
                metrics[f'mape_{m}'] = 0.0
                
        return metrics

    print(f"Starting Mamba training for {config.epochs} epochs...")
    best_val_mae = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, f"best_mamba_k_{config.k}.pth")

    start_time = time.time()
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for xb, yb_node in train_loader:
            xb, yb_node = xb.to(device), yb_node.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(config.use_amp and device.type == 'cuda')):
                pred_hat = model(xb)
                pred_node = torch.matmul(pred_hat.float(), U_t) * std_t + mean_t
                loss = masked_mae_loss(pred_node, yb_node, null_val=0.0)

            scaler.scale(loss).backward()

            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_metrics = evaluate(val_loader)
        val_mae = val_metrics['avg_mae']

        print(f"Epoch {epoch+1:02d}/{config.epochs} | Train L1 (node): {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_metrics['avg_rmse']:.4f} | Val MAPE: {val_metrics['avg_mape']:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
            print("  -> Saved new best model!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Optional: clear cache to free up fragmented memory (not recommended every batch)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    end_time = time.time()

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(test_loader)
    print(f"\nFinal Test - Avg MAE: {test_metrics['avg_mae']:.4f} | Avg RMSE: {test_metrics['avg_rmse']:.4f} | Avg MAPE: {test_metrics['avg_mape']:.4f}")

    res_dict = {
        "model_name": "SpectralMambaReal",
        "k": config.k,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "seed": args.seed if hasattr(args, 'seed') else None,
        
        "mae_15": test_metrics.get('mae_15', 0.0),
        "rmse_15": test_metrics.get('rmse_15', 0.0),
        "mape_15": test_metrics.get('mape_15', 0.0),
        
        "mae_30": test_metrics.get('mae_30', 0.0),
        "rmse_30": test_metrics.get('rmse_30', 0.0),
        "mape_30": test_metrics.get('mape_30', 0.0),
        
        "mae_60": test_metrics.get('mae_60', 0.0),
        "rmse_60": test_metrics.get('rmse_60', 0.0),
        "mape_60": test_metrics.get('mape_60', 0.0),
        
        "avg_mae": test_metrics.get('avg_mae', 0.0),
        "avg_rmse": test_metrics.get('avg_rmse', 0.0),
        "avg_mape": test_metrics.get('avg_mape', 0.0),
        
        "time_sec": round(end_time - start_time, 2),
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs_completed": epoch + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return res_dict

def main():
    args = parse_args()

    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.k = args.k
    config.d_model = args.d_model
    config.num_layers = args.num_layers
    config.learning_rate = args.learning_rate
    # In a real setup we'd parse configs/mamba.yaml here if args.config is set.
    # For simplicity, using the Config class above with overrides from args.

    res = run_mamba_training(config, args)

    if res is not None:
        out_dir = config.results_dir
        csv_path = os.path.join(out_dir, "mamba_k_sweep_results.csv")
        # Ensure we always have all columns in order
        columns = [
            "model_name", "k", "d_model", "num_layers", "seed",
            "mae_15", "rmse_15", "mape_15", 
            "mae_30", "rmse_30", "mape_30", 
            "mae_60", "rmse_60", "mape_60", 
            "avg_mae", "avg_rmse", "avg_mape", 
            "time_sec", "device", "gpu_name", 
            "batch_size", "learning_rate", "epochs_completed", "timestamp"
        ]
        df = pd.DataFrame([res], columns=columns)

        # Append if exists, else write header
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        print(f"\nResults successfully saved to {csv_path}")

if __name__ == "__main__":
    main()