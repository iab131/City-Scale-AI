"""
Evaluate a STAEformer checkpoint on the test set with no residual branch.
Used to establish the trunk-only baseline (Stage A) when the trunk training
script didn't complete its own test eval.

Usage:
    python scripts/disr/eval_trunk_only.py \\
        --ckpt results/staeformer/stae_trunk/best_stae_s42.pth
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.staeformer import STAEformer
from models.disr.losses import per_horizon_metrics, per_speed_regime_mae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out", type=str, default="results/disr/trunk_only_metrics.json")
    args = p.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207,
                               cache_dir=args.cache_dir)
    arrs = split_train_val_test(
        [data["X"], data["X_norm"], data["tod"], data["dow"], data["missing_mask"]],
        0.7, 0.1)
    (_, _, X_te), (_, _, Xn_te), (_, _, tod_te), \
        (_, _, dow_te), (_, _, mk_te) = arrs
    te_ds = SSSMDataset(X_te, Xn_te, tod_te, dow_te, mk_te, 12, 12)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, pin_memory=True)
    mean, std = float(data["mean"]), float(data["std"])

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

    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=device.type == "cuda"):
                yn = model(x_norm, tod_b, dow_b)
            yp = yn.float() * std + mean
            all_p.append(yp.cpu()); all_y.append(batch["y_node"])
            all_m.append(batch["y_mask"])
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    m = per_horizon_metrics(P, Y, M)
    psr = per_speed_regime_mae(P, Y, M)
    res = {**m, "per_speed_regime": psr,
           "ckpt": args.ckpt,
           "epoch": ckpt.get("epoch"),
           "val_metrics": ckpt.get("val_metrics")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(m, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
