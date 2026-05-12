"""
Train the Hybrid STAEformer + Spectral Mamba model on METR-LA.

Uses the same pipeline as train_staeformer.py — only the model differs.
"""

import os
import sys
import time
import json
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

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.hybrid import HybridSTAEMamba


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/hybrid")
    p.add_argument("--tag", type=str, default="hybrid")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # STAEformer args (paper defaults)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--adaptive_embedding_dim", type=int, default=80)
    p.add_argument("--feed_forward_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # Spectral branch args
    p.add_argument("--spec_d", type=int, default=32)
    p.add_argument("--spec_layers", type=int, default=2)

    # Training (paper defaults)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[20, 30])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--gradient_clip", type=float, default=0.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mae(p, y, m, eps=1e-6):
    mm = m.mean().clamp(min=eps)
    return ((p - y).abs() * m).mean() / mm


def masked_rmse(p, y, m, eps=1e-6):
    mm = m.mean().clamp(min=eps)
    return torch.sqrt(((p - y) ** 2 * m).mean() / mm)


def masked_mape(p, y, m, eps=1e-6):
    mm = m * (y.abs() > 1e-3).float()
    mmean = mm.mean().clamp(min=eps)
    return ((p - y).abs() / y.abs().clamp(min=eps) * mm).mean() / mmean


def per_horizon(pred, true, mask):
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

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    mask = data["missing_mask"]
    mean, std = data["mean"], data["std"]
    U_np, evals_np = data["U"], data["evals"]
    U = torch.from_numpy(U_np).float()
    evals = torch.from_numpy(evals_np).float()
    print(f"[data] T={len(X)}, N={X.shape[1]}")

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
        (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
        (mk_tr, mk_va, mk_te) = arrs

    def mk_loader(Xp, Xnp, tp, dp, mp, sh):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                          num_workers=args.num_workers, pin_memory=True)
    tr = mk_loader(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, True)
    va = mk_loader(X_va, Xn_va, tod_va, dow_va, mk_va, False)
    te = mk_loader(X_te, Xn_te, tod_te, dow_te, mk_te, False)
    print(f"[data] |tr|={len(tr.dataset)} |va|={len(va.dataset)} |te|={len(te.dataset)}")

    model = HybridSTAEMamba(
        N=X.shape[1], U=U, evals=evals,
        in_steps=args.in_steps, out_steps=args.out_steps,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout,
        spec_d=args.spec_d, spec_layers=args.spec_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] HybridSTAEMamba params={n_params/1e6:.2f}M (stae model_dim={model.stae_model_dim}, spec_d={model.spec_d})")

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones, gamma=args.lr_gamma)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = args.use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f"[amp] dtype={amp_dtype} scaler_enabled={use_scaler}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_hybrid_s{args.seed}.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0; nb = 0
        for batch in tr:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype,
                                    enabled=(args.use_amp and device.type == "cuda")):
                yn = model(x_norm, tod_b, dow_b)
                y_pred = yn * std_t + mean_t
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
        ap, ay, am = [], [], []
        with torch.no_grad():
            for batch in va:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                y_node = batch["y_node"]; y_mask = batch["y_mask"]
                with torch.amp.autocast('cuda', dtype=amp_dtype,
                                        enabled=(args.use_amp and device.type == "cuda")):
                    yn = model(x_norm, tod_b, dow_b)
                y_pred = yn.float() * std_t + mean_t
                ap.append(y_pred.cpu()); ay.append(y_node); am.append(y_mask)
        P = torch.cat(ap); Y = torch.cat(ay); M = torch.cat(am)
        vm = per_horizon(P, Y, M); val_mae = vm["avg_mae"]

        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} train_mae={train_mae:.3f} "
              f"val_mae={val_mae:.3f} "
              f"val_15/30/60={vm.get('mae_15', 0):.3f}/"
              f"{vm.get('mae_30', 0):.3f}/{vm.get('mae_60', 0):.3f}",
              flush=True)

        if val_mae < best_val - 1e-4:
            best_val = val_mae; epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_metrics": vm, "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement in {args.patience} epochs")
                break

    # Test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ap, ay, am = [], [], []
    with torch.no_grad():
        for batch in te:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"]; y_mask = batch["y_mask"]
            with torch.amp.autocast('cuda', dtype=amp_dtype,
                                    enabled=(args.use_amp and device.type == "cuda")):
                yn = model(x_norm, tod_b, dow_b)
            y_pred = yn.float() * std_t + mean_t
            ap.append(y_pred.cpu()); ay.append(y_node); am.append(y_mask)
    P = torch.cat(ap); Y = torch.cat(ay); M = torch.cat(am)
    tm = per_horizon(P, Y, M)

    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(tm, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s  best_val_mae={best_val:.3f}")

    row = {
        "model": "HybridSTAEMamba",
        "seed": args.seed, "epochs": ckpt["epoch"],
        "params_M": round(n_params / 1e6, 3), "elapsed_sec": round(elapsed, 1),
        **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
        **{f"test_{k}": v for k, v in tm.items()},
        "tag": args.tag, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = os.path.join(args.out_dir, "hybrid_results.csv")
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"[done] appended to {csv_path}")


if __name__ == "__main__":
    main()
