"""
Precompute STAEformer's normalized predictions Y_base for train/val/test.
The residual branch is trained on cached deltas, so this script runs once
per STAEformer checkpoint and saves npz files in ``results/disr/stae_base/``.

Usage:
    python scripts/disr/precompute_stae_base.py \\
        --ckpt results/staeformer/stae_s42/best_stae_s42.pth \\
        --out_dir results/disr/stae_base/s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make project importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data  # noqa: E402
from dataset_v2 import SSSMDataset, split_train_val_test  # noqa: E402
from models.staeformer import STAEformer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def run(args):
    os.chdir(ROOT)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207,
                               cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mk = data["tod"], data["dow"], data["missing_mask"]
    arrs = split_train_val_test([X, X_norm, tod, dow, mk], 0.7, 0.1)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = STAEformer(
        N=a.get("N", 207),
        in_steps=a["in_steps"], out_steps=a["out_steps"],
        input_embedding_dim=a["input_embedding_dim"],
        tod_embedding_dim=a["tod_embedding_dim"],
        dow_embedding_dim=a["dow_embedding_dim"],
        adaptive_embedding_dim=a["adaptive_embedding_dim"],
        feed_forward_dim=a["feed_forward_dim"],
        num_heads=a["num_heads"], num_layers=a["num_layers"],
        dropout=a["dropout"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stae] loaded {args.ckpt} ({n_params/1e6:.2f} M params)")

    splits = ["train", "val", "test"]
    for i, name in enumerate(splits):
        (X_p,), (Xn_p,), (tod_p,), (dow_p,), (mk_p,) = (
            (arrs[j][i],) for j in range(5)
        )
        ds = SSSMDataset(X_p, Xn_p, tod_p, dow_p, mk_p,
                         input_len=a["in_steps"], pred_len=a["out_steps"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        y_pred_norm = []
        y_true_raw = []
        y_mask = []
        t0 = time.time()
        for batch in loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                yn = model(x_norm, tod_b, dow_b)
            y_pred_norm.append(yn.float().cpu().numpy())
            y_true_raw.append(batch["y_node"].numpy())
            y_mask.append(batch["y_mask"].numpy())
        Yn = np.concatenate(y_pred_norm, axis=0)
        Yt = np.concatenate(y_true_raw, axis=0)
        Mk = np.concatenate(y_mask, axis=0)
        dt = time.time() - t0
        out_path = os.path.join(args.out_dir, f"{name}.npz")
        np.savez(out_path, y_pred_norm=Yn, y_true_raw=Yt, y_mask=Mk)
        print(f"[stae] {name}: {Yn.shape}  -> {out_path}  ({dt:.1f}s)")

    # Persist scaling stats too so the residual trainer can de-normalize.
    np.savez(os.path.join(args.out_dir, "norm_stats.npz"),
             mean=np.array(data["mean"], dtype=np.float32),
             std=np.array(data["std"], dtype=np.float32))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "in_steps": a["in_steps"],
            "out_steps": a["out_steps"],
            "stae_params_M": round(n_params / 1e6, 3),
            "elapsed_sec": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"[stae] cache written to {args.out_dir}")


if __name__ == "__main__":
    run(parse_args())
