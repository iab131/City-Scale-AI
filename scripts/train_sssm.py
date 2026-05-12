"""
Train the Spectral State Space Model (SSSM) on METR-LA.

Key differences from scripts/train_mamba.py:
  - Uses masked normalization (zeros excluded from train mean/std).
  - Includes time-of-day and day-of-week features.
  - Predicts directly in NODE space (model handles inverse-GFT internally).
  - Bi-axis (modes + time) Mamba architecture (models.spectral_ssm.SpectralStateSpaceModel).
  - Masked MAE loss + masked metrics; reports per-horizon 15/30/60-min results.
  - Cosine LR schedule with linear warmup.
  - Multi-seed evaluation supported via --seed.

Example:
    python scripts/train_sssm.py --k 207 --d_model 96 --num_layers 3 \
        --epochs 100 --batch_size 64 --learning_rate 1e-3 --seed 42
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
from models.spectral_ssm import build_model                 # noqa: E402


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/sssm")
    p.add_argument("--tag", type=str, default="run")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--version", type=str, default="v1", choices=["v1", "v2", "v3", "v4", "v5", "v8"])
    p.add_argument("--no_residualize", action="store_true", help="v5: disable residualized input")
    p.add_argument("--adp_emb_dim", type=int, default=80, help="v8: adaptive embedding dimension")
    p.add_argument("--tod_dim", type=int, default=32, help="v8: TOD embedding dimension")
    p.add_argument("--dow_dim", type=int, default=32, help="v8: DOW embedding dimension")
    p.add_argument("--unidirectional", action="store_true", help="v8: use unidirectional Mamba (faster)")
    p.add_argument("--n_heads", type=int, default=4, help="v3 only")
    p.add_argument("--k", type=int, default=207)
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_dec_layers", type=int, default=2, help="v2 only")
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--cheb_order", type=int, default=3)
    p.add_argument("--cheb_channels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_node_bias", action="store_true", help="v2 only: disable per-sensor bias")

    # Training
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

    p.add_argument("--quick", action="store_true",
                   help="Quick sanity run: fewer epochs, smaller subset.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Losses & metrics (all in NODE space, masked on zeros = missing)
# ---------------------------------------------------------------------------


def masked_mae(pred, true, mask, eps=1e-6):
    # mask: 1 where valid, 0 where missing
    m = mask
    m_mean = m.mean().clamp(min=eps)
    return ((pred - true).abs() * m).mean() / m_mean


def masked_rmse(pred, true, mask, eps=1e-6):
    m = mask
    m_mean = m.mean().clamp(min=eps)
    mse = ((pred - true) ** 2 * m).mean() / m_mean
    return torch.sqrt(mse)


def masked_mape(pred, true, mask, eps=1e-6):
    m = mask * (true.abs() > 1e-3).float()
    m_mean = m.mean().clamp(min=eps)
    return ((pred - true).abs() / true.abs().clamp(min=eps) * m).mean() / m_mean


def per_horizon_metrics(pred, true, mask):
    """
    pred, true, mask: [B, T_out, N]
    Returns dict with avg_* and per-horizon (15/30/60-min) MAE/RMSE/MAPE.
    """
    metrics = {
        "avg_mae":  masked_mae(pred, true, mask).item(),
        "avg_rmse": masked_rmse(pred, true, mask).item(),
        "avg_mape": masked_mape(pred, true, mask).item(),
    }
    # METR-LA is 5-min steps; horizons 15/30/60 -> t = 2/5/11
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p_t = pred[:, t:t + 1, :]
            y_t = true[:, t:t + 1, :]
            m_t = mask[:, t:t + 1, :]
            metrics[f"mae_{tag}"]  = masked_mae(p_t, y_t, m_t).item()
            metrics[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            metrics[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def cosine_warmup(step: int, total: int, warmup: int):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, k={args.k}, d_model={args.d_model}, "
          f"L={args.num_layers}, seed={args.seed}")

    # ------------------------------------------------------------------ data
    data = get_cached_v2_data(args.data_path, args.adj_path, args.k, args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    mask = data["missing_mask"]
    mean, std = data["mean"], data["std"]
    U = data["U"]
    evals = data["evals"]
    prior = data.get("prior")
    prior_norm = data.get("prior_norm")
    print(f"[data] T={len(X)}, N={X.shape[1]}, mean={mean:.3f}, std={std:.3f}")

    use_prior = args.version == "v5"
    split_arrays = [X, X_norm, tod, dow, mask]
    if use_prior:
        split_arrays.extend([prior, prior_norm])
    arrs = split_train_val_test(split_arrays, train_frac=0.7, val_frac=0.1)
    if use_prior:
        (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
            (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
            (mk_tr, mk_va, mk_te), \
            (pr_tr, pr_va, pr_te), (prn_tr, prn_va, prn_te) = arrs
    else:
        (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
            (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
            (mk_tr, mk_va, mk_te) = arrs
        pr_tr = pr_va = pr_te = None
        prn_tr = prn_va = prn_te = None

    def mk_loader(X_p, Xn_p, tod_p, dow_p, mk_p, pr_p, prn_p, shuffle):
        ds = SSSMDataset(X_p, Xn_p, tod_p, dow_p, mk_p,
                         input_len=args.input_len, pred_len=args.pred_len,
                         prior=pr_p, prior_norm=prn_p)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)

    tr_loader = mk_loader(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, pr_tr, prn_tr, True)
    va_loader = mk_loader(X_va, Xn_va, tod_va, dow_va, mk_va, pr_va, prn_va, False)
    te_loader = mk_loader(X_te, Xn_te, tod_te, dow_te, mk_te, pr_te, prn_te, False)
    print(f"[data] |train|={len(tr_loader.dataset)}  |val|={len(va_loader.dataset)}  "
          f"|test|={len(te_loader.dataset)}")

    # ---------------------------------------------------------------- model
    model_kwargs = dict(
        d_model=args.d_model, num_layers=args.num_layers,
        d_state=args.d_state, d_conv=args.d_conv, expand=args.expand,
        cheb_order=args.cheb_order, cheb_channels=args.cheb_channels,
        dropout=args.dropout,
        input_len=args.input_len, pred_len=args.pred_len,
    )
    if args.version == "v2":
        model_kwargs["num_dec_layers"] = args.num_dec_layers
        model_kwargs["use_node_bias"] = not args.no_node_bias
    elif args.version == "v3":
        model_kwargs["n_heads"] = args.n_heads
        model_kwargs["use_node_bias"] = not args.no_node_bias
    elif args.version == "v4":
        model_kwargs["use_node_bias"] = not args.no_node_bias
    elif args.version == "v5":
        model_kwargs["use_node_bias"] = not args.no_node_bias
        model_kwargs["residualize_input"] = not args.no_residualize
    elif args.version == "v8":
        model_kwargs["use_node_bias"] = not args.no_node_bias
        model_kwargs["adp_emb_dim"] = args.adp_emb_dim
        model_kwargs["tod_dim"] = args.tod_dim
        model_kwargs["dow_dim"] = args.dow_dim
        model_kwargs["bidirectional"] = not args.unidirectional
    model = build_model(
        K=args.k, U_np=U, evals_np=evals,
        version=args.version, **model_kwargs,
    ).to(device)
    print(f"[model] version={args.version}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    # bf16 has the same exponent range as fp32 -> no overflow, no need for GradScaler.
    # We keep GradScaler disabled when AMP is bf16, enabled only when AMP is fp16.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = args.use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f"[amp] use_amp={args.use_amp} dtype={amp_dtype} scaler_enabled={use_scaler}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    # --------------------------------------------------------------- train
    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_sssm_k{args.k}_s{args.seed}.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    t_start = time.time()

    if args.quick:
        args.epochs = min(args.epochs, 3)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        lr_mul = cosine_warmup(epoch - 1, args.epochs, args.warmup_epochs)
        for g in opt.param_groups:
            g["lr"] = args.learning_rate * lr_mul

        running = 0.0
        n_batches = 0
        for batch in tr_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            y_tod = batch["y_tod"].to(device, non_blocking=True)
            y_dow = batch["y_dow"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(args.use_amp and device.type == "cuda")):
                if args.version == "v5":
                    pn_in = batch["prior_norm_in"].to(device, non_blocking=True)
                    pn_out = batch["prior_norm_out"].to(device, non_blocking=True)
                    y_pred_norm = model(x_norm, tod_b, dow_b, y_tod, y_dow, pn_in, pn_out)
                elif args.version in ("v2", "v3", "v4", "v8"):
                    y_pred_norm = model(x_norm, tod_b, dow_b, y_tod, y_dow)
                else:
                    y_pred_norm = model(x_norm, tod_b, dow_b)
                y_pred = y_pred_norm * std_t + mean_t                # node-space speeds
                loss = masked_mae(y_pred, y_node, y_mask)

            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            n_batches += 1

        train_mae = running / max(1, n_batches)

        # ---- eval
        model.eval()
        all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                y_node = batch["y_node"].to(device, non_blocking=True)
                y_mask = batch["y_mask"].to(device, non_blocking=True)
                y_tod = batch["y_tod"].to(device, non_blocking=True)
                y_dow = batch["y_dow"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(args.use_amp and device.type == "cuda")):
                    if args.version == "v5":
                        pn_in_v = batch["prior_norm_in"].to(device, non_blocking=True)
                        pn_out_v = batch["prior_norm_out"].to(device, non_blocking=True)
                        y_pred_norm = model(x_norm, tod_b, dow_b, y_tod, y_dow, pn_in_v, pn_out_v)
                    elif args.version in ("v2", "v3", "v4", "v8"):
                        y_pred_norm = model(x_norm, tod_b, dow_b, y_tod, y_dow)
                    else:
                        y_pred_norm = model(x_norm, tod_b, dow_b)
                y_pred = y_pred_norm.float() * std_t + mean_t
                all_p.append(y_pred.cpu())
                all_y.append(y_node.cpu())
                all_m.append(y_mask.cpu())
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_metrics = per_horizon_metrics(P, Y, M)
        val_mae = val_metrics["avg_mae"]

        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} train_mae={train_mae:.3f} "
              f"val_mae={val_mae:.3f} "
              f"val_15/30/60={val_metrics.get('mae_15', 0):.3f}/"
              f"{val_metrics.get('mae_30', 0):.3f}/{val_metrics.get('mae_60', 0):.3f}")

        history.append({"epoch": epoch, "train_mae": train_mae, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "val_metrics": val_metrics,
                        "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement in {args.patience} epochs")
                break

    # ---------------------------------------------------------------- test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            y_tod = batch["y_tod"].to(device, non_blocking=True)
            y_dow = batch["y_dow"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(args.use_amp and device.type == "cuda")):
                if args.version == "v5":
                    pn_in = batch["prior_norm_in"].to(device, non_blocking=True)
                    pn_out = batch["prior_norm_out"].to(device, non_blocking=True)
                    y_pred_norm = model(x_norm, tod_b, dow_b, y_tod, y_dow, pn_in, pn_out)
                elif args.version in ("v2", "v3", "v4", "v8"):
                    y_pred_norm = model(x_norm, tod_b, dow_b, y_tod, y_dow)
                else:
                    y_pred_norm = model(x_norm, tod_b, dow_b)
            y_pred = y_pred_norm.float() * std_t + mean_t
            all_p.append(y_pred.cpu())
            all_y.append(y_node.cpu())
            all_m.append(y_mask.cpu())
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    test_metrics = per_horizon_metrics(P, Y, M)

    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(test_metrics, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s  best_val_mae={best_val:.3f}")

    # Append to CSV
    row = {
        "model": "SSSM",
        "k": args.k, "d_model": args.d_model, "num_layers": args.num_layers,
        "seed": args.seed, "epochs": ckpt["epoch"],
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
