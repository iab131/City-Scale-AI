"""
Train TESTAM on METR-LA. Mirrors scripts/train_staeformer.py (same data
plumbing, masked-MAE loss on raw mph, bf16 AMP, MultiStepLR, early stop) so
the trained checkpoints drop straight into the existing per-horizon top-K
ensemble eval.

Example:
    python scripts/train_testam.py --tag testam_s0 --seed 0
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.testam import TESTAM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/testam")
    p.add_argument("--tag", type=str, default="testam")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # TESTAM hyperparameters (paper defaults for METR-LA)
    p.add_argument("--N", type=int, default=207)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--ffn_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--memory_size", type=int, default=20)
    p.add_argument("--d_emb", type=int, default=16)
    p.add_argument("--d_gate", type=int, default=32)

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[30, 50])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mae(pred, true, mask, eps=1e-6):
    m_mean = mask.mean().clamp(min=eps)
    return ((pred - true).abs() * mask).mean() / m_mean


def masked_rmse(pred, true, mask, eps=1e-6):
    m_mean = mask.mean().clamp(min=eps)
    return torch.sqrt(((pred - true) ** 2 * mask).mean() / m_mean)


def masked_mape(pred, true, mask, eps=1e-6):
    m = mask * (true.abs() > 1e-3).float()
    m_mean = m.mean().clamp(min=eps)
    return ((pred - true).abs() / true.abs().clamp(min=eps) * m).mean() / m_mean


def per_horizon_metrics(pred, true, mask):
    out = {
        "avg_mae": masked_mae(pred, true, mask).item(),
        "avg_rmse": masked_rmse(pred, true, mask).item(),
        "avg_mape": masked_mape(pred, true, mask).item(),
    }
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p_t, y_t, m_t = pred[:, t:t+1], true[:, t:t+1], mask[:, t:t+1]
            out[f"mae_{tag}"] = masked_mae(p_t, y_t, m_t).item()
            out[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            out[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return out


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} seed={args.seed} tag={args.tag}")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207,
                               cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    mask = data["missing_mask"]
    mean, std = data["mean"], data["std"]
    print(f"[data] T={len(X)} N={X.shape[1]} mean={mean:.3f} std={std:.3f}")

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
        (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
        (mk_tr, mk_va, mk_te) = arrs

    def mk_loader(X_p, Xn_p, tod_p, dow_p, mk_p, shuffle):
        ds = SSSMDataset(X_p, Xn_p, tod_p, dow_p, mk_p,
                         input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=False)

    tr = mk_loader(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, True)
    va = mk_loader(X_va, Xn_va, tod_va, dow_va, mk_va, False)
    te = mk_loader(X_te, Xn_te, tod_te, dow_te, mk_te, False)
    print(f"[data] |tr|={len(tr.dataset)} |va|={len(va.dataset)} "
          f"|te|={len(te.dataset)}")

    model = TESTAM(
        N=args.N,
        in_steps=args.in_steps, out_steps=args.out_steps,
        d_model=args.d_model, num_layers=args.num_layers,
        ffn_dim=args.ffn_dim, num_heads=args.num_heads,
        dropout=args.dropout, memory_size=args.memory_size,
        d_emb=args.d_emb, d_gate=args.d_gate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TESTAM params={n_params/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=args.lr_milestones, gamma=args.lr_gamma,
    )
    amp_dtype = (torch.bfloat16 if torch.cuda.is_available()
                 and torch.cuda.is_bf16_supported() else torch.float16)
    use_amp = args.use_amp and device.type == "cuda"

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_testam_s{args.seed}.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    max_gpu_mb = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0; nb = 0
        for batch in tr:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                y_norm = model(x_norm, tod_b, dow_b)
                y_pred = y_norm * std_t + mean_t
                loss = masked_mae(y_pred, y_node, y_mask)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.gradient_clip)
            opt.step()
            run += float(loss.detach()); nb += 1
            if torch.cuda.is_available():
                max_gpu_mb = max(max_gpu_mb,
                                  int(torch.cuda.max_memory_allocated() / 1024 / 1024))
        scheduler.step()
        tr_mae = run / max(1, nb)

        model.eval()
        all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    yn = model(x_norm, tod_b, dow_b)
                yp = yn.float() * std_t + mean_t
                all_p.append(yp.cpu())
                all_y.append(batch["y_node"])
                all_m.append(batch["y_mask"])
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_m = per_horizon_metrics(P, Y, M)
        v_mae = val_m["avg_mae"]
        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} tr_mae={tr_mae:.3f} "
              f"val avg/15/30/60 = {v_mae:.3f} / {val_m['mae_15']:.3f} / "
              f"{val_m['mae_30']:.3f} / {val_m['mae_60']:.3f}  "
              f"gpu={max_gpu_mb}MB", flush=True)

        if v_mae < best_val - 1e-4:
            best_val = v_mae
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_metrics": val_m, "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs")
                break

    # test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                yn = model(x_norm, tod_b, dow_b)
            yp = yn.float() * std_t + mean_t
            all_p.append(yp.cpu())
            all_y.append(batch["y_node"])
            all_m.append(batch["y_mask"])
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    te_m = per_horizon_metrics(P, Y, M)
    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(te_m, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s best_val={best_val:.3f} max_gpu={max_gpu_mb}MB")

    row = {
        "model": "TESTAM",
        "seed": args.seed,
        "epochs": ckpt["epoch"],
        "params_M": round(n_params / 1e6, 3),
        "elapsed_sec": round(elapsed, 1),
        **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
        **{f"test_{k}": v for k, v in te_m.items()},
        "tag": args.tag,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = os.path.join(args.out_dir, "testam_results.csv")
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path),
              index=False)
    print(f"[done] appended to {csv_path}")


if __name__ == "__main__":
    main()
