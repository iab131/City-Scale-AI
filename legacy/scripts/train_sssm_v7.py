"""
Train SSSM v7 with multi-window input (recent + day-ago + week-ago).
Uses MultiWindowSSSMDataset which holds the full sequence and indexes by
prediction-start t0, looking up historical windows.

Example:
    python scripts/train_sssm_v7.py --k 207 --d_model 96 --num_layers 3 \
        --epochs 100 --batch_size 64 --tag v7_d96_L3 --seed 42
"""

import os
import sys
import time
import json
import math
import argparse
import datetime
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data                # noqa: E402
from dataset_v2 import MultiWindowSSSMDataset, split_t0_range   # noqa: E402
from models.spectral_ssm import build_model                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/sssm")
    p.add_argument("--tag", type=str, default="v7")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k", type=int, default=207)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--cheb_order", type=int, default=3)
    p.add_argument("--cheb_channels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_node_bias", action="store_true")

    # Multi-window
    p.add_argument("--use_daily", action="store_true", default=True)
    p.add_argument("--use_weekly", action="store_true", default=True)
    p.add_argument("--no_daily", action="store_true")
    p.add_argument("--no_weekly", action="store_true")

    p.add_argument("--input_len", type=int, default=12)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mae(pred, true, mask, eps=1e-6):
    m_mean = mask.mean().clamp(min=eps)
    return ((pred - true).abs() * mask).mean() / m_mean


def masked_rmse(pred, true, mask, eps=1e-6):
    m_mean = mask.mean().clamp(min=eps)
    mse = ((pred - true) ** 2 * mask).mean() / m_mean
    return torch.sqrt(mse)


def masked_mape(pred, true, mask, eps=1e-6):
    m = mask * (true.abs() > 1e-3).float()
    m_mean = m.mean().clamp(min=eps)
    return ((pred - true).abs() / true.abs().clamp(min=eps) * m).mean() / m_mean


def per_horizon_metrics(pred, true, mask):
    out = {
        "avg_mae":  masked_mae(pred, true, mask).item(),
        "avg_rmse": masked_rmse(pred, true, mask).item(),
        "avg_mape": masked_mape(pred, true, mask).item(),
    }
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p_t, y_t, m_t = pred[:, t:t+1, :], true[:, t:t+1, :], mask[:, t:t+1, :]
            out[f"mae_{tag}"]  = masked_mae(p_t, y_t, m_t).item()
            out[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            out[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return out


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    use_daily = args.use_daily and not args.no_daily
    use_weekly = args.use_weekly and not args.no_weekly
    num_windows = 1 + int(use_daily) + int(use_weekly)
    print(f"[setup] device={device}, k={args.k}, d_model={args.d_model}, "
          f"L={args.num_layers}, seed={args.seed}, "
          f"num_windows={num_windows} (daily={use_daily}, weekly={use_weekly})")

    # Load data (with calendar prior etc, even though v7 doesn't use prior — for future flexibility)
    data = get_cached_v2_data(args.data_path, args.adj_path, args.k, args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]
    U, evals = data["U"], data["evals"]
    print(f"[data] T={len(X)}, N={X.shape[1]}, mean={mean:.3f}, std={std:.3f}")

    # Split by t0 (prediction-start index)
    tr_range, va_range, te_range = split_t0_range(
        len(X), train_frac=0.7, val_frac=0.1, max_t0_lookback=args.pred_len,
    )
    print(f"[data] t0 ranges train={tr_range} val={va_range} test={te_range}")

    def mk_ds(t_range):
        t0_start, t0_end = t_range
        return MultiWindowSSSMDataset(
            X, X_norm, tod, dow, mask,
            t0_start=t0_start, t0_end=t0_end,
            input_len=args.input_len, pred_len=args.pred_len,
            use_daily=use_daily, use_weekly=use_weekly,
        )

    tr_ds, va_ds, te_ds = mk_ds(tr_range), mk_ds(va_range), mk_ds(te_range)
    print(f"[data] |train|={len(tr_ds)}  |val|={len(va_ds)}  |test|={len(te_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    model = build_model(
        K=args.k, U_np=U, evals_np=evals, version="v7",
        d_model=args.d_model, num_layers=args.num_layers,
        d_state=args.d_state, d_conv=args.d_conv, expand=args.expand,
        cheb_order=args.cheb_order, cheb_channels=args.cheb_channels,
        dropout=args.dropout,
        input_len=args.input_len, pred_len=args.pred_len,
        num_windows=num_windows, use_node_bias=not args.no_node_bias,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and device.type == "cuda"))

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_v7_k{args.k}_s{args.seed}.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        lr_mul = cosine_warmup(epoch - 1, args.epochs, args.warmup_epochs)
        for g in opt.param_groups:
            g["lr"] = args.learning_rate * lr_mul

        running = 0.0; nb = 0
        for batch in tr_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            win_id = batch["win_id"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            y_tod = batch["y_tod"].to(device, non_blocking=True)
            y_dow = batch["y_dow"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(args.use_amp and device.type == "cuda")):
                y_pred_norm = model(x_norm, tod_b, dow_b, win_id, y_tod, y_dow)
                y_pred = y_pred_norm * std_t + mean_t
                loss = masked_mae(y_pred, y_node, y_mask)
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(opt); scaler.update()
            running += float(loss.detach()); nb += 1
        train_mae = running / max(1, nb)

        # eval
        model.eval()
        all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                win_id = batch["win_id"].to(device, non_blocking=True)
                y_node = batch["y_node"].to(device, non_blocking=True)
                y_mask = batch["y_mask"].to(device, non_blocking=True)
                y_tod = batch["y_tod"].to(device, non_blocking=True)
                y_dow = batch["y_dow"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(args.use_amp and device.type == "cuda")):
                    y_pred_norm = model(x_norm, tod_b, dow_b, win_id, y_tod, y_dow)
                y_pred = y_pred_norm.float() * std_t + mean_t
                all_p.append(y_pred.cpu()); all_y.append(y_node.cpu()); all_m.append(y_mask.cpu())
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_metrics = per_horizon_metrics(P, Y, M)
        val_mae = val_metrics["avg_mae"]

        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} train_mae={train_mae:.3f} "
              f"val_mae={val_mae:.3f} "
              f"val_15/30/60={val_metrics.get('mae_15', 0):.3f}/"
              f"{val_metrics.get('mae_30', 0):.3f}/{val_metrics.get('mae_60', 0):.3f}",
              flush=True)

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_metrics": val_metrics, "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement in {args.patience} epochs")
                break

    # test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            win_id = batch["win_id"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            y_tod = batch["y_tod"].to(device, non_blocking=True)
            y_dow = batch["y_dow"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(args.use_amp and device.type == "cuda")):
                y_pred_norm = model(x_norm, tod_b, dow_b, win_id, y_tod, y_dow)
            y_pred = y_pred_norm.float() * std_t + mean_t
            all_p.append(y_pred.cpu()); all_y.append(y_node.cpu()); all_m.append(y_mask.cpu())
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    test_metrics = per_horizon_metrics(P, Y, M)

    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(test_metrics, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s  best_val_mae={best_val:.3f}")

    row = {
        "model": "SSSM_v7",
        "k": args.k, "d_model": args.d_model, "num_layers": args.num_layers,
        "seed": args.seed, "epochs": ckpt["epoch"],
        "num_windows": num_windows,
        "params_M": round(n_params / 1e6, 3),
        "elapsed_sec": round(elapsed, 1),
        **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "tag": args.tag,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = os.path.join(args.out_dir, "sssm_results.csv")
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"[done] appended to {csv_path}")


if __name__ == "__main__":
    main()
