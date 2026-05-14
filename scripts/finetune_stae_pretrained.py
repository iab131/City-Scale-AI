"""
Finetune STAEformer with frozen pretrained STMAE encoder(s).

Usage:
  python scripts/finetune_stae_pretrained.py \
      --tmae_ckpt results/stmae/pretrain/tmae_best.pth \
      --smae_ckpt results/stmae/pretrain/smae_best.pth \
      --tag R03_stae_pretrained --seed 42
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
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_pretrained import PretrainedSSSMDataset, split_t0_for_pretrained
from models.staeformer_pretrained import STAEformerPretrained


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/stae_pretrained")
    p.add_argument("--tag", type=str, default="stae_pretrained")

    p.add_argument("--tmae_ckpt", type=str, default=None)
    p.add_argument("--smae_ckpt", type=str, default=None)
    p.add_argument("--no_freeze", action="store_true",
                   help="Don't freeze pretrained encoders (full finetune)")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--T_long", type=int, default=2016)
    p.add_argument("--input_embedding_dim", type=int, default=24)
    p.add_argument("--tod_embedding_dim", type=int, default=24)
    p.add_argument("--dow_embedding_dim", type=int, default=24)
    p.add_argument("--adaptive_embedding_dim", type=int, default=80)
    p.add_argument("--feed_forward_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_pre", type=int, default=32)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[20, 30])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--gradient_clip", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


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
    print(f"[setup] device={device} seed={args.seed} freeze={not args.no_freeze}")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    missing = data["missing_mask"]
    mean, std = data["mean"], data["std"]
    T, N = X.shape

    tr_r, va_r, te_r = split_t0_for_pretrained(T, args.T_long, args.in_steps, args.out_steps)
    print(f"[data] train t0 {tr_r}  val t0 {va_r}  test t0 {te_r}")

    def mk_ds(rng):
        return PretrainedSSSMDataset(X, X_norm, tod, dow, missing,
                                     t0_start=rng[0], t0_end=rng[1],
                                     T_in=args.in_steps, T_out=args.out_steps,
                                     T_long=args.T_long)

    tr_ds = mk_ds(tr_r)
    va_ds = mk_ds(va_r)
    te_ds = mk_ds(te_r)
    print(f"[data] |train|={len(tr_ds)}  |val|={len(va_ds)}  |test|={len(te_ds)}")

    model = STAEformerPretrained(
        N=N,
        tmae_ckpt_path=args.tmae_ckpt,
        smae_ckpt_path=args.smae_ckpt,
        in_steps=args.in_steps, out_steps=args.out_steps,
        input_embedding_dim=args.input_embedding_dim,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout, d_pre=args.d_pre,
        freeze_pretrained=not args.no_freeze,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={trainable/1e6:.2f}M total={total/1e6:.2f}M model_dim={model.model_dim}")

    # ---- Precompute encoder cache if frozen (saves 80% of training time) ----
    use_precompute = not args.no_freeze
    pre_cache = {"tr": None, "va": None, "te": None}
    if use_precompute:
        amp_dtype_pre = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        def build_cache(ds, tag):
            from torch.utils.data import DataLoader as DL
            loader = DL(ds, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            out = []
            model.eval()
            t0_build = time.time()
            with torch.no_grad():
                for batch in loader:
                    lh = batch["long_hist"].to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=amp_dtype_pre):
                        pre = model.encode_pre(lh)
                    out.append(pre.float().cpu())
            cache = torch.cat(out, dim=0)
            print(f"[precompute {tag}] {cache.shape} {time.time()-t0_build:.1f}s")
            return cache
        pre_cache["tr"] = build_cache(tr_ds, "tr")
        pre_cache["va"] = build_cache(va_ds, "va")
        pre_cache["te"] = build_cache(te_ds, "te")
        # Leave encoders allocated so saved checkpoints include their weights
        # (the cache is float32 on CPU, ~few hundred MB; encoders' weight memory
        #  is negligible on H200)
        torch.cuda.empty_cache()

    class CachedWrap(torch.utils.data.Dataset):
        """Wraps a PretrainedSSSMDataset to substitute the long_hist with the
        precomputed encoder output."""
        def __init__(self, base, cache):
            self.base = base; self.cache = cache
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            s = self.base[idx]
            if self.cache is not None:
                s["pre_combined"] = self.cache[idx]
                del s["long_hist"]
            return s

    tr_loader = DataLoader(CachedWrap(tr_ds, pre_cache["tr"]),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(CachedWrap(va_ds, pre_cache["va"]),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(CachedWrap(te_ds, pre_cache["te"]),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones,
                                                     gamma=args.lr_gamma)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[amp] dtype={amp_dtype}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_stae_pre_s{args.seed}.pth")

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
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                if use_precompute:
                    pre_combined = batch["pre_combined"].to(device, non_blocking=True)
                    y_pred_norm = model.forward_with_pre(x_norm, tod_b, dow_b, pre_combined)
                else:
                    long_hist = batch["long_hist"].to(device, non_blocking=True)
                    y_pred_norm = model(x_norm, tod_b, dow_b, long_hist)
                y_pred = y_pred_norm * std_t + mean_t
                loss = masked_mae(y_pred, y_node, y_mask)

            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            running += float(loss.detach()); nb += 1

        train_mae = running / max(1, nb)

        model.eval()
        all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    if use_precompute:
                        pre_combined = batch["pre_combined"].to(device, non_blocking=True)
                        yn = model.forward_with_pre(x_norm, tod_b, dow_b, pre_combined)
                    else:
                        long_hist = batch["long_hist"].to(device, non_blocking=True)
                        yn = model(x_norm, tod_b, dow_b, long_hist)
                y_pred = yn.float() * std_t + mean_t
                all_p.append(y_pred.cpu())
                all_y.append(batch["y_node"]); all_m.append(batch["y_mask"])
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_metrics = per_horizon_metrics(P, Y, M)
        val_mae = val_metrics["avg_mae"]
        scheduler.step()
        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} train_mae={train_mae:.3f} "
              f"val_mae={val_mae:.3f} val_15/30/60={val_metrics.get('mae_15',0):.3f}/"
              f"{val_metrics.get('mae_30',0):.3f}/{val_metrics.get('mae_60',0):.3f}", flush=True)

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

    # ---- test ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                if use_precompute:
                    pre_combined = batch["pre_combined"].to(device, non_blocking=True)
                    yn = model.forward_with_pre(x_norm, tod_b, dow_b, pre_combined)
                else:
                    long_hist = batch["long_hist"].to(device, non_blocking=True)
                    yn = model(x_norm, tod_b, dow_b, long_hist)
            y_pred = yn.float() * std_t + mean_t
            all_p.append(y_pred.cpu())
            all_y.append(batch["y_node"]); all_m.append(batch["y_mask"])
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    test_metrics = per_horizon_metrics(P, Y, M)

    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(test_metrics, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s best_val_mae={best_val:.3f}")

    # Append to CSV
    row = {
        "model": "STAEformer_pretrained",
        "seed": args.seed, "epochs": ckpt["epoch"],
        "use_tmae": args.tmae_ckpt is not None,
        "use_smae": args.smae_ckpt is not None,
        "freeze": not args.no_freeze,
        "elapsed_sec": round(elapsed, 1),
        **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "tag": args.tag,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = os.path.join(args.out_dir, "stae_pretrained_results.csv")
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"[done] appended to {csv_path}")


if __name__ == "__main__":
    main()
