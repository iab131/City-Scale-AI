"""
Train STAEformer (or hybrid STAEformer + Spectral Mamba) on METR-LA.
Uses our masked normalization + masked MAE pipeline so results are directly
comparable to v1–v8.

Example:
    python scripts/train_staeformer.py --tag stae_repro --seed 42

The hybrid model is enabled via --hybrid; uses our spectral Mamba as an
auxiliary feature branch fused with STAEformer's hidden state before the
output projection.
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
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data                # noqa: E402
from dataset_v2 import SSSMDataset, split_train_val_test    # noqa: E402
from models.staeformer import STAEformer                    # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/staeformer")
    p.add_argument("--tag", type=str, default="stae")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # STAEformer hyperparams (per the paper's METR-LA config)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--input_embedding_dim", type=int, default=24)
    p.add_argument("--tod_embedding_dim", type=int, default=24)
    p.add_argument("--dow_embedding_dim", type=int, default=24)
    p.add_argument("--adaptive_embedding_dim", type=int, default=80)
    p.add_argument("--feed_forward_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training hyperparams (per the paper)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[20, 30])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--gradient_clip", type=float, default=0.0,
                   help="0 disables (paper has no clipping for METR-LA)")
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)

    # Tier-S improvements
    p.add_argument("--horizon_weighted", action="store_true",
                   help="Use per-horizon weighted loss (heavier on 60-min)")
    p.add_argument("--horizon_weights", type=float, nargs="+",
                   default=[1.0, 1.0, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8],
                   help="Per-horizon weights for weighted loss (length must equal out_steps)")
    p.add_argument("--use_swa", action="store_true",
                   help="Apply Stochastic Weight Averaging after main training (5 extra epochs)")
    p.add_argument("--swa_epochs", type=int, default=5,
                   help="Number of SWA tail epochs at constant lr")
    p.add_argument("--swa_lr", type=float, default=1e-5,
                   help="Constant LR for SWA phase (kept low since main training already plateaued)")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mae(pred, true, mask, eps=1e-6):
    m_mean = mask.mean().clamp(min=eps)
    return ((pred - true).abs() * mask).mean() / m_mean


def horizon_weighted_masked_mae(pred, true, mask, weights, eps=1e-6):
    """
    Per-horizon weighted masked MAE.
    pred, true, mask: [B, T_out, N]
    weights: [T_out]  -- multiplicative weights per horizon

    Equivalent to masked_mae with each horizon-t loss multiplied by weights[t],
    then normalized by mean(mask * weights[t]). Keeps the gradient pointing
    where mass is.
    """
    # Expand weights to [1, T_out, 1] for broadcasting
    w = weights.view(1, -1, 1).to(pred.device)
    weighted_mask = mask * w
    wm_mean = weighted_mask.mean().clamp(min=eps)
    return ((pred - true).abs() * weighted_mask).mean() / wm_mean


def masked_rmse(pred, true, mask, eps=1e-6):
    m_mean = mask.mean().clamp(min=eps)
    return torch.sqrt(((pred - true) ** 2 * mask).mean() / m_mean)


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
            p_t, y_t, m_t = pred[:, t:t+1], true[:, t:t+1], mask[:, t:t+1]
            out[f"mae_{tag}"]  = masked_mae(p_t, y_t, m_t).item()
            out[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            out[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return out


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, seed={args.seed}")

    # Data
    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    mask = data["missing_mask"]
    mean, std = data["mean"], data["std"]
    print(f"[data] T={len(X)}, N={X.shape[1]}, mean={mean:.3f}, std={std:.3f}")

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
        (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
        (mk_tr, mk_va, mk_te) = arrs

    def mk_loader(X_p, Xn_p, tod_p, dow_p, mk_p, shuffle):
        ds = SSSMDataset(X_p, Xn_p, tod_p, dow_p, mk_p,
                         input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)

    tr_loader = mk_loader(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, True)
    va_loader = mk_loader(X_va, Xn_va, tod_va, dow_va, mk_va, False)
    te_loader = mk_loader(X_te, Xn_te, tod_te, dow_te, mk_te, False)
    print(f"[data] |train|={len(tr_loader.dataset)}  |val|={len(va_loader.dataset)}  |test|={len(te_loader.dataset)}")

    # Model
    model = STAEformer(
        N=X.shape[1],
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        input_embedding_dim=args.input_embedding_dim,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] STAEformer params={n_params/1e6:.2f}M, model_dim={model.model_dim}")

    # Optimizer & schedule
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones,
                                                     gamma=args.lr_gamma)

    # AMP — bf16 on H200, no scaler
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = args.use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f"[amp] dtype={amp_dtype} scaler_enabled={use_scaler}")

    # Loss configuration
    if args.horizon_weighted:
        assert len(args.horizon_weights) == args.out_steps, \
            f"horizon_weights length {len(args.horizon_weights)} != out_steps {args.out_steps}"
        h_weights = torch.tensor(args.horizon_weights, dtype=torch.float32)
        print(f"[loss] horizon_weighted MAE, weights={args.horizon_weights}")
    else:
        h_weights = None
        print(f"[loss] standard masked MAE")

    # SWA setup
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(opt, swa_lr=args.swa_lr, anneal_epochs=1, anneal_strategy="cos")
        print(f"[swa] enabled, {args.swa_epochs} tail epochs at lr={args.swa_lr}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_stae_s{args.seed}.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0; nb = 0
        for batch in tr_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype,
                                    enabled=(args.use_amp and device.type == "cuda")):
                y_pred_norm = model(x_norm, tod_b, dow_b)        # [B, T_out, N]
                y_pred = y_pred_norm * std_t + mean_t            # raw mph
                if h_weights is not None:
                    loss = horizon_weighted_masked_mae(y_pred, y_node, y_mask, h_weights)
                else:
                    loss = masked_mae(y_pred, y_node, y_mask)

            if use_scaler:
                scaler.scale(loss).backward()
                if args.gradient_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                opt.step()

            running += float(loss.detach()); nb += 1
        train_mae = running / max(1, nb)

        # Eval
        model.eval()
        all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                y_node = batch["y_node"]; y_mask = batch["y_mask"]
                with torch.amp.autocast('cuda', dtype=amp_dtype,
                                        enabled=(args.use_amp and device.type == "cuda")):
                    yn = model(x_norm, tod_b, dow_b)
                y_pred = yn.float() * std_t + mean_t
                all_p.append(y_pred.cpu()); all_y.append(y_node); all_m.append(y_mask)
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_metrics = per_horizon_metrics(P, Y, M)
        val_mae = val_metrics["avg_mae"]

        scheduler.step()
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

    # ---- SWA tail (if enabled) ----
    swa_ckpt_path = None
    if args.use_swa and swa_model is not None:
        from torch.optim.swa_utils import update_bn
        print(f"[swa] starting SWA tail: {args.swa_epochs} epochs at lr={args.swa_lr}")
        # Load best checkpoint as starting point
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # Rebuild SWA model from the best checkpoint
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        # Set constant low lr for SWA
        for g in opt.param_groups:
            g["lr"] = args.swa_lr
        for swa_epoch in range(args.swa_epochs):
            model.train()
            running = 0.0; nb = 0
            for batch in tr_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                y_node = batch["y_node"].to(device, non_blocking=True)
                y_mask = batch["y_mask"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype,
                                        enabled=(args.use_amp and device.type == "cuda")):
                    y_pred_norm = model(x_norm, tod_b, dow_b)
                    y_pred = y_pred_norm * std_t + mean_t
                    if h_weights is not None:
                        loss = horizon_weighted_masked_mae(y_pred, y_node, y_mask, h_weights)
                    else:
                        loss = masked_mae(y_pred, y_node, y_mask)
                loss.backward()
                opt.step()
                running += float(loss.detach()); nb += 1
            swa_model.update_parameters(model)
            print(f"[swa ep {swa_epoch+1}/{args.swa_epochs}] train_loss={running/max(1, nb):.4f}", flush=True)
        # STAEformer has no BatchNorm, only LayerNorm — update_bn is a no-op but safe to call
        update_bn(tr_loader, swa_model, device=device)

        # Evaluate SWA model on val
        swa_model.eval()
        ap, ay, am = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                y_node = batch["y_node"]; y_mask = batch["y_mask"]
                with torch.amp.autocast('cuda', dtype=amp_dtype,
                                        enabled=(args.use_amp and device.type == "cuda")):
                    yn = swa_model(x_norm, tod_b, dow_b)
                y_pred = yn.float() * std_t + mean_t
                ap.append(y_pred.cpu()); ay.append(y_node); am.append(y_mask)
        P = torch.cat(ap); Y = torch.cat(ay); M = torch.cat(am)
        swa_val_metrics = per_horizon_metrics(P, Y, M)
        print(f"[swa val] {json.dumps(swa_val_metrics)}")

        # Save SWA model and use it if val is better than best ckpt
        swa_ckpt_path = os.path.join(out_dir, f"swa_stae_s{args.seed}.pth")
        torch.save({"model": swa_model.module.state_dict(), "args": vars(args),
                    "val_metrics": swa_val_metrics, "epoch": -1}, swa_ckpt_path)
        if swa_val_metrics["avg_mae"] < best_val:
            print(f"[swa] SWA val ({swa_val_metrics['avg_mae']:.4f}) BEATS pre-SWA best "
                  f"({best_val:.4f}) — using SWA model for test")
            ckpt_path = swa_ckpt_path  # use SWA ckpt for final test
            best_val = swa_val_metrics["avg_mae"]
        else:
            print(f"[swa] SWA val ({swa_val_metrics['avg_mae']:.4f}) does NOT beat pre-SWA best "
                  f"({best_val:.4f}) — keeping pre-SWA ckpt for test, SWA saved separately")

    # Test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"]; y_mask = batch["y_mask"]
            with torch.amp.autocast('cuda', dtype=amp_dtype,
                                    enabled=(args.use_amp and device.type == "cuda")):
                yn = model(x_norm, tod_b, dow_b)
            y_pred = yn.float() * std_t + mean_t
            all_p.append(y_pred.cpu()); all_y.append(y_node); all_m.append(y_mask)
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    test_metrics = per_horizon_metrics(P, Y, M)

    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(test_metrics, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s  best_val_mae={best_val:.3f}")

    row = {
        "model": "STAEformer_repro",
        "seed": args.seed, "epochs": ckpt["epoch"],
        "params_M": round(n_params / 1e6, 3),
        "elapsed_sec": round(elapsed, 1),
        **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "tag": args.tag,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = os.path.join(args.out_dir, "staeformer_results.csv")
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"[done] appended to {csv_path}")


if __name__ == "__main__":
    main()
