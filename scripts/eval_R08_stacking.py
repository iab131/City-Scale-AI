"""
R08 — Stacking with a learned per-(sensor, horizon) residual predictor.

The setup: for each test sample, our super-ensemble gives us a prediction P_e.
We train a small MLP on val data that takes (P_e, gating features) and
predicts the residual (y_true - P_e). Then on test, our final prediction is
P_e + predicted_residual.

This is a form of stacking where the base layer is the ensemble and the
meta-layer is the residual predictor.

Risk: val overfitting (only ~3400 val samples). Mitigation: small MLP
(< 5000 params total), early stopping on a held-out 20% of val.
"""

import os
import sys
import glob
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from dataset_pretrained import PretrainedSSSMDataset, split_t0_for_pretrained
from data_utils import load_adj_pkl
from graph_utils import symmetrize_adjacency
from models.staeformer import STAEformer
from models.graph_wavenet import GraphWaveNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stae_glob", type=str,
                   default="results/staeformer/stae_R0*/best_stae_s*.pth")
    p.add_argument("--stae_pre_glob", type=str,
                   default="results/stae_pretrained/*/best_stae_pre_s*.pth")
    p.add_argument("--gwnet_glob", type=str,
                   default="results/gwnet/gwnet_s*/best_gwnet_s*.pth")
    p.add_argument("--use_gwnet", action="store_true")

    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)

    p.add_argument("--stack_hidden", type=int, default=32)
    p.add_argument("--stack_lr", type=float, default=1e-3)
    p.add_argument("--stack_epochs", type=int, default=50)
    p.add_argument("--stack_wd", type=float, default=1e-3)
    p.add_argument("--stack_holdout", type=float, default=0.2,
                   help="Fraction of val held out as inner-val for early stop")
    p.add_argument("--out_json", type=str, default="results/R08_stacking.json")
    return p.parse_args()


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


class ResidualStacker(nn.Module):
    """Small MLP that takes ensemble prediction + gating features and
    outputs a per-(t, n) additive correction.

    Input dim: T_out + F_gate (= 12 + ~10)
    Output dim: T_out * N (= 12 * 207)  → reshape to [B, T, N]

    To keep param count low, use shared per-horizon stack and a per-sensor
    bias term:
      logits = mlp(ens_per_horizon + gate_feat) → [B, T, 1]
      correction = logits.expand(B, T, N) + sensor_bias[T, N]
    """

    def __init__(self, N: int, T_out: int = 12, F_gate: int = 10, hidden: int = 32):
        super().__init__()
        self.N = N
        self.T_out = T_out
        self.shared = nn.Sequential(
            nn.Linear(T_out + F_gate, hidden),
            nn.GELU(),
            nn.Linear(hidden, T_out),
        )
        # Zero init: identity (no correction)
        nn.init.zeros_(self.shared[-1].weight); nn.init.zeros_(self.shared[-1].bias)
        # Per-sensor zero-init bias (additive)
        self.sensor_bias = nn.Parameter(torch.zeros(T_out, N))

    def forward(self, ens_pred, gate_feat):
        # ens_pred: [B, T, N]  in raw mph
        # gate_feat: [B, F]
        # Use the mean-over-N ensemble prediction at each horizon as a feature
        ens_mean = ens_pred.mean(dim=2)               # [B, T]
        x = torch.cat([ens_mean, gate_feat], dim=-1)  # [B, T+F]
        delta_h = self.shared(x)                       # [B, T]  per-horizon shared correction
        delta = delta_h.unsqueeze(-1).expand_as(ens_pred) + self.sensor_bias.unsqueeze(0)
        return ens_pred + delta


def build_features(x_norm, tod, dow):
    """Per-sample features for the stacker."""
    B = x_norm.shape[0]
    mean_speed = x_norm.mean(dim=(1, 2)).unsqueeze(-1)
    std_speed = x_norm.std(dim=(1, 2)).unsqueeze(-1)
    tod_now = tod[:, -1]
    ang = 2.0 * 3.14159 * tod_now
    tod_sc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
    dow_now = dow[:, -1].long()
    dow_oh = torch.nn.functional.one_hot(dow_now, num_classes=7).float()
    return torch.cat([mean_speed, std_speed, tod_sc, dow_oh], dim=-1)


def load_stae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False); a = ckpt["args"]
    m = STAEformer(N=207, in_steps=a["in_steps"], out_steps=a["out_steps"],
                   input_embedding_dim=a["input_embedding_dim"],
                   tod_embedding_dim=a["tod_embedding_dim"],
                   dow_embedding_dim=a["dow_embedding_dim"],
                   adaptive_embedding_dim=a["adaptive_embedding_dim"],
                   feed_forward_dim=a["feed_forward_dim"],
                   num_heads=a["num_heads"], num_layers=a["num_layers"],
                   dropout=a["dropout"]).to(device).eval()
    m.load_state_dict(ckpt["model"])
    return m


def collect_preds(model, loader, device, amp_dtype):
    preds = []; ys = []; ms = []; feats = []
    with torch.no_grad():
        for batch in loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                yn = model(x_norm, tod_b, dow_b)
            preds.append(yn.float().cpu())
            ys.append(batch["y_node"]); ms.append(batch["y_mask"])
            feats.append(build_features(batch["x_norm"], batch["tod"], batch["dow"]))
    return torch.cat(preds), torch.cat(ys), torch.cat(ms), torch.cat(feats)


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask_arr = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]

    arrs = split_train_val_test([X, X_norm, tod, dow, mask_arr], 0.7, 0.1)
    (_, X_va, X_te), (_, Xn_va, Xn_te), (_, tod_va, tod_te), \
        (_, dow_va, dow_te), (_, mk_va, mk_te) = arrs

    def mk(Xp, Xnp, tp, dp, mp):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    short_va = mk(X_va, Xn_va, tod_va, dow_va, mk_va)
    short_te = mk(X_te, Xn_te, tod_te, dow_te, mk_te)
    print(f"|val|={len(short_va.dataset)} |test|={len(short_te.dataset)}")

    stae_paths = sorted(glob.glob(args.stae_glob))
    if not stae_paths:
        print("no STAE checkpoints"); sys.exit(1)
    print(f"found {len(stae_paths)} stae checkpoints")

    val_preds = []; test_preds = []
    Y_val = M_val = F_val = None
    Y_test = M_test = F_test = None
    for path in stae_paths:
        print(f"  {os.path.basename(path)}")
        m = load_stae(path, device)
        Pv_norm, Yv, Mv, Fv = collect_preds(m, short_va, device, amp_dtype)
        Pt_norm, Yt, Mt, Ft = collect_preds(m, short_te, device, amp_dtype)
        Pv = Pv_norm * float(std) + float(mean)
        Pt = Pt_norm * float(std) + float(mean)
        val_preds.append(Pv); test_preds.append(Pt)
        if Y_val is None:
            Y_val = Yv; M_val = Mv; F_val = Fv
            Y_test = Yt; M_test = Mt; F_test = Ft
        del m; torch.cuda.empty_cache()

    # uniform ensemble
    Pv_e = torch.stack(val_preds).mean(dim=0)
    Pt_e = torch.stack(test_preds).mean(dim=0)

    base_metrics = per_horizon(Pt_e, Y_test, M_test)
    print(f"\nbase ensemble 60-min: {base_metrics['mae_60']:.4f}")

    # Holdout
    n_val = Pv_e.shape[0]
    n_held = int(n_val * args.stack_holdout)
    perm = torch.randperm(n_val)
    inner_idx = perm[:n_val - n_held]
    held_idx = perm[n_val - n_held:]

    # Train stacker
    N = 207; T_out = args.out_steps
    F_gate = F_val.shape[-1]
    stacker = ResidualStacker(N=N, T_out=T_out, F_gate=F_gate, hidden=args.stack_hidden).to(device)
    n_params = sum(p.numel() for p in stacker.parameters())
    print(f"stacker params: {n_params}")
    opt = torch.optim.AdamW(stacker.parameters(), lr=args.stack_lr, weight_decay=args.stack_wd)

    Pv_e_dev = Pv_e.to(device); Y_val_dev = Y_val.float().to(device); M_val_dev = M_val.float().to(device)
    F_val_dev = F_val.to(device)

    best_inner_mae = float("inf"); best_state = None; epochs_no_improve = 0
    for epoch in range(1, args.stack_epochs + 1):
        stacker.train()
        # Train on inner_idx
        idx = inner_idx[torch.randperm(len(inner_idx))]
        bs = 256
        for i in range(0, len(idx), bs):
            ix = idx[i:i+bs]
            P_b = Pv_e_dev[ix]; Y_b = Y_val_dev[ix]; M_b = M_val_dev[ix]; F_b = F_val_dev[ix]
            P_corr = stacker(P_b, F_b)
            loss = masked_mae(P_corr, Y_b, M_b)
            opt.zero_grad(); loss.backward(); opt.step()
        # Eval on held
        stacker.eval()
        with torch.no_grad():
            P_held = stacker(Pv_e_dev[held_idx], F_val_dev[held_idx])
            inner_mae = masked_mae(P_held, Y_val_dev[held_idx], M_val_dev[held_idx]).item()
        if inner_mae < best_inner_mae - 1e-5:
            best_inner_mae = inner_mae; best_state = {k: v.detach().clone() for k, v in stacker.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break
        if epoch % 5 == 0:
            print(f"[ep {epoch:03d}] inner_mae={inner_mae:.4f}  best={best_inner_mae:.4f}")

    stacker.load_state_dict(best_state); stacker.eval()
    # Apply on test
    with torch.no_grad():
        Pt_e_dev = Pt_e.to(device); F_test_dev = F_test.to(device)
        Pt_corr = stacker(Pt_e_dev, F_test_dev).cpu()
    stacked_metrics = per_horizon(Pt_corr, Y_test, M_test)
    print(f"\nstacked  60-min: {stacked_metrics['mae_60']:.4f}  (delta {stacked_metrics['mae_60'] - base_metrics['mae_60']:+.4f})")

    results = {
        "base": base_metrics, "stacked": stacked_metrics,
        "stack_hidden": args.stack_hidden, "stack_holdout": args.stack_holdout,
        "n_stae_checkpoints": len(stae_paths),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
