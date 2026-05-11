"""
Evaluate an ensemble of v4 checkpoints on the test set.

Loads each checkpoint, runs it on test data, averages the (normalized) predictions
across seeds, then computes masked MAE / RMSE / MAPE per horizon.
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.spectral_ssm import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_pattern", type=str,
                   default="results/sssm/v4_d96_L3*/best_sssm_k207_s*.pth",
                   help="Glob for checkpoints to ensemble")
    p.add_argument("--k", type=int, default=207)
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--input_len", type=int, default=12)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def masked_mae(p, y, m):
    mm = m.mean().clamp(min=1e-6)
    return ((p - y).abs() * m).mean() / mm


def masked_rmse(p, y, m):
    mm = m.mean().clamp(min=1e-6)
    return torch.sqrt(((p - y) ** 2 * m).mean() / mm)


def masked_mape(p, y, m):
    mm = (m * (y.abs() > 1e-3).float())
    mmean = mm.mean().clamp(min=1e-6)
    return ((p - y).abs() / y.abs().clamp(min=1e-6) * mm).mean() / mmean


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(glob.glob(args.ckpt_pattern))
    if not ckpts:
        print(f"NO checkpoints matched {args.ckpt_pattern}")
        sys.exit(1)
    print(f"Found {len(ckpts)} checkpoints:")
    for c in ckpts:
        print(f"  {c}")

    data = get_cached_v2_data(args.data_path, args.adj_path, args.k, args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]
    U, evals = data["U"], data["evals"]

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (_, _, X_te), (_, _, Xn_te), (_, _, tod_te), \
        (_, _, dow_te), (_, _, mk_te) = arrs
    te_ds = SSSMDataset(X_te, Xn_te, tod_te, dow_te, mk_te,
                        input_len=args.input_len, pred_len=args.pred_len)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    print(f"|test|={len(te_ds)}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    # Run each checkpoint, collect predictions in normalized space
    all_pred_norm = []
    all_y = []
    all_m = []

    for i, ckpt_path in enumerate(ckpts):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckargs = ckpt["args"]
        model = build_model(
            K=ckargs["k"], U_np=U, evals_np=evals,
            version=ckargs.get("version", "v4"),
            d_model=ckargs["d_model"], num_layers=ckargs["num_layers"],
            d_state=ckargs.get("d_state", 16),
            d_conv=ckargs.get("d_conv", 4),
            expand=ckargs.get("expand", 2),
            cheb_order=ckargs.get("cheb_order", 3),
            cheb_channels=ckargs.get("cheb_channels", 4),
            dropout=ckargs.get("dropout", 0.1),
            input_len=ckargs.get("input_len", 12),
            pred_len=ckargs.get("pred_len", 12),
            use_node_bias=not ckargs.get("no_node_bias", False),
        ).to(device).eval()
        model.load_state_dict(ckpt["model"])

        preds = []
        y_batches = []
        m_batches = []
        with torch.no_grad():
            for batch in te_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                y_node = batch["y_node"]
                y_mask = batch["y_mask"]
                y_tod = batch["y_tod"].to(device, non_blocking=True)
                y_dow = batch["y_dow"].to(device, non_blocking=True)
                version = ckargs.get("version", "v4")
                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    if version in ("v2", "v3", "v4"):
                        yn = model(x_norm, tod_b, dow_b, y_tod, y_dow)
                    else:
                        yn = model(x_norm, tod_b, dow_b)
                preds.append(yn.float().cpu())
                if i == 0:
                    y_batches.append(y_node)
                    m_batches.append(y_mask)
        P_norm = torch.cat(preds, dim=0)
        all_pred_norm.append(P_norm)
        if i == 0:
            Y = torch.cat(y_batches, dim=0)
            M = torch.cat(m_batches, dim=0)
        # individual model report
        P_node = P_norm * float(std) + float(mean)
        m_i = per_horizon(P_node, Y, M)
        print(f"\n[ckpt {os.path.basename(ckpt_path)}] {json.dumps(m_i)}")

    # Ensemble: average normalized predictions across all models
    P_ens_norm = torch.stack(all_pred_norm).mean(dim=0)
    P_ens_node = P_ens_norm * float(std) + float(mean)
    m_ens = per_horizon(P_ens_node, Y, M)

    print("\n" + "=" * 60)
    print(f"ENSEMBLE ({len(ckpts)} models)")
    print("=" * 60)
    print(json.dumps(m_ens, indent=2))
    print()
    print("Per-horizon comparison vs published SOTA on METR-LA:")
    print(f"  {'horizon':<8} {'ours':>8} {'GraphWaveNet':>14} {'STAEformer':>12} {'STD-MAE':>9} {'MLCAFormer':>12}")
    sota = {
        "15": (m_ens["mae_15"], 2.69, 2.65, 2.62, None),
        "30": (m_ens["mae_30"], 3.08, 2.97, 2.99, None),
        "60": (m_ens["mae_60"], 3.51, 3.34, 3.40, 3.30),
    }
    for h, vals in sota.items():
        ours, gwn, sta, std_mae, mlca = vals
        mlca_s = f"{mlca:>12.2f}" if mlca else f"{'-':>12}"
        print(f"  {h+'-min':<8} {ours:>8.3f} {gwn:>14.2f} {sta:>12.2f} {std_mae:>9.2f} {mlca_s}")


if __name__ == "__main__":
    main()
