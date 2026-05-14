"""
Pretrain TMAE or SMAE on long-history METR-LA windows.

Usage:
    python scripts/pretrain_stmae.py --kind tmae --epochs 50 --batch_size 8
    python scripts/pretrain_stmae.py --kind smae --epochs 50 --batch_size 8

Checkpoints saved to results/stmae/<tag>/<kind>_best.pth
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data       # noqa: E402
from dataset_long import LongHistoryDataset         # noqa: E402
from models.stmae import TMAE, SMAE                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/stmae")
    p.add_argument("--tag", type=str, default="pretrain")
    p.add_argument("--kind", type=str, choices=["tmae", "smae"], required=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--patch_size", type=int, default=12)
    p.add_argument("--max_patches", type=int, default=168)
    p.add_argument("--embed_dim", type=int, default=96)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--encoder_depth", type=int, default=4)
    p.add_argument("--decoder_depth", type=int, default=1)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--T_long", type=int, default=2016)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def masked_l1(pred, gt, mask=None, eps=1e-6):
    """Mean absolute error. If `mask` (broadcastable to pred), only count
    entries where mask>0. Used in normalized space — already small magnitude."""
    if mask is None:
        return (pred - gt).abs().mean()
    mm = mask.mean().clamp(min=eps)
    return ((pred - gt).abs() * mask).mean() / mm


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[pretrain {args.kind}] seed={args.seed} T_long={args.T_long} "
          f"patch={args.patch_size} d={args.embed_dim} depth={args.encoder_depth}+{args.decoder_depth}")

    # ---- data ----
    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X_norm = data["X_norm"]
    missing_mask = data["missing_mask"]
    T, N = X_norm.shape
    n_train = int(0.7 * T)
    n_val = int(0.1 * T)

    tr_ds = LongHistoryDataset(X_norm, t_start=0, t_end=n_train,
                               T_long=args.T_long, stride=args.stride,
                               missing_mask=missing_mask)
    va_ds = LongHistoryDataset(X_norm, t_start=n_train, t_end=n_train + n_val,
                               T_long=args.T_long, stride=args.stride * 4,
                               missing_mask=missing_mask)
    print(f"[data] |train|={len(tr_ds)} |val|={len(va_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- model ----
    common_kw = dict(
        patch_size=args.patch_size, max_patches=args.max_patches,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio, dropout=args.dropout,
    )
    if args.kind == "tmae":
        model = TMAE(**common_kw).to(device)
    else:
        model = SMAE(N=N, **common_kw).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.kind} params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[amp] dtype={amp_dtype}")

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{args.kind}_best.pth")

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
            x = batch["x_long"].to(device, non_blocking=True)        # [B, N, T_long]
            m = batch.get("mask_long")
            if m is not None: m = m.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                recon_m, gt_m, masked_idx = model(x)
                # Build the corresponding mask (same shape as recon_m)
                if m is not None:
                    P = x.shape[2] // args.patch_size
                    mp = m.view(m.shape[0], m.shape[1], P, args.patch_size)
                    if args.kind == "tmae":
                        m_m = mp[:, :, masked_idx, :]
                    else:
                        m_m = mp[:, masked_idx, :, :]
                else:
                    m_m = None
                loss = masked_l1(recon_m, gt_m, m_m)

            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()
            running += float(loss.detach()); nb += 1

        train_loss = running / max(1, nb)

        # ---- val ----
        model.eval()
        vr = 0.0; vn = 0
        with torch.no_grad():
            for batch in va_loader:
                x = batch["x_long"].to(device, non_blocking=True)
                m = batch.get("mask_long")
                if m is not None: m = m.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    recon_m, gt_m, masked_idx = model(x)
                    if m is not None:
                        P = x.shape[2] // args.patch_size
                        mp = m.view(m.shape[0], m.shape[1], P, args.patch_size)
                        if args.kind == "tmae":
                            m_m = mp[:, :, masked_idx, :]
                        else:
                            m_m = mp[:, masked_idx, :, :]
                    else:
                        m_m = None
                    loss = masked_l1(recon_m, gt_m, m_m)
                vr += float(loss); vn += 1
        val_loss = vr / max(1, vn)

        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} train={train_loss:.4f} val={val_loss:.4f}",
              flush=True)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_loss": val_loss, "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement in {args.patience} epochs")
                break

    elapsed = time.time() - t_start
    print(f"[done] elapsed={elapsed:.1f}s best_val_loss={best_val:.4f}")
    print(f"[done] checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
