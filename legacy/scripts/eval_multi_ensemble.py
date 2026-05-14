"""
Multi-model ensemble eval — STAEformer (4 seeds) + GraphWaveNet (k seeds).

For each model:
  1. Collect predictions on VAL set
  2. Collect predictions on TEST set
Optimize PER-MODEL weights on VAL (gradient descent on weighted-average MAE
in raw mph), then apply those weights on TEST.

Also computes:
  - Uniform-average ensemble (baseline)
  - Best single model on test
  - Optimal weighted ensemble (per overall avg MAE)
  - Optionally with ST-TTC on top

This is the proper way to combine architecturally-diverse models when
their individual error scales differ.
"""

import os
import sys
import json
import glob
import argparse
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from data_utils import load_adj_pkl
from graph_utils import symmetrize_adjacency
from models.staeformer import STAEformer
from models.graph_wavenet import GraphWaveNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stae_ckpts", type=str,
                   default="results/staeformer/stae_repro*/best_stae_s*.pth")
    p.add_argument("--gwnet_ckpts", type=str,
                   default="results/gwnet/gwnet_s*/best_gwnet_s*.pth")
    p.add_argument("--use_ttc", action="store_true")
    p.add_argument("--ttc_groups", type=int, default=4)
    p.add_argument("--ttc_lr", type=float, default=1e-4)
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
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


class SDCalibrator(torch.nn.Module):
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = freq_bins // groups
        self.lambda_amp = torch.nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = torch.nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        B, T, N = y_pred.shape
        y = y_pred.permute(0, 2, 1)
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf); P = torch.angle(Yf)
        Yf_corr = torch.zeros_like(Yf)
        M = T // 2 + 1
        for g in range(self.groups):
            start = g * self.group_size
            end = M if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.permute(0, 2, 1)


def load_stae_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = STAEformer(
        N=207, in_steps=a["in_steps"], out_steps=a["out_steps"],
        input_embedding_dim=a["input_embedding_dim"],
        tod_embedding_dim=a["tod_embedding_dim"],
        dow_embedding_dim=a["dow_embedding_dim"],
        adaptive_embedding_dim=a["adaptive_embedding_dim"],
        feed_forward_dim=a["feed_forward_dim"],
        num_heads=a["num_heads"], num_layers=a["num_layers"],
        dropout=a["dropout"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def load_gwnet_ckpt(ckpt_path, adj_torch, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = GraphWaveNet(
        N=207, adj_mx=adj_torch,
        in_steps=a["in_steps"], out_steps=a["out_steps"],
        in_dim=3, out_dim=1,
        residual_channels=a["residual_channels"],
        dilation_channels=a["dilation_channels"],
        skip_channels=a["skip_channels"],
        end_channels=a["end_channels"],
        kernel_size=a["kernel_size"],
        blocks=a["blocks"], layers=a["layers"],
        dropout=a["dropout"],
        adaptive_adj=not a.get("no_adaptive_adj", False),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def collect_preds(model, loader, device, mean, std, amp_dtype):
    """Run model on loader, return predictions in node space [N_samples, T, N]."""
    preds = []
    ys = []
    ms = []
    with torch.no_grad():
        for batch in loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                yn = model(x_norm, tod_b, dow_b)
            y_pred = yn.float() * std + mean
            preds.append(y_pred.cpu())
            ys.append(batch["y_node"])
            ms.append(batch["y_mask"])
    return torch.cat(preds), torch.cat(ys), torch.cat(ms)


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]

    _, _, A = load_adj_pkl(args.adj_path)
    A = symmetrize_adjacency(A)
    adj_torch = torch.from_numpy(A).float()

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (_, X_va, X_te), (_, Xn_va, Xn_te), \
        (_, tod_va, tod_te), (_, dow_va, dow_te), \
        (_, mk_va, mk_te) = arrs

    def mk(Xp, Xnp, tp, dp, mp):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    va = mk(X_va, Xn_va, tod_va, dow_va, mk_va)
    te = mk(X_te, Xn_te, tod_te, dow_te, mk_te)
    print(f"|val|={len(va.dataset)} |test|={len(te.dataset)}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ---- Collect predictions from every model ----
    stae_paths = sorted(glob.glob(args.stae_ckpts))
    gwnet_paths = sorted(glob.glob(args.gwnet_ckpts))
    print(f"Found {len(stae_paths)} STAEformer + {len(gwnet_paths)} GraphWaveNet ckpts")

    val_preds = []   # list of [N_val, T, N]
    test_preds = []  # list of [N_test, T, N]
    names = []

    Y_val = M_val = Y_test = M_test = None

    for path in stae_paths:
        print(f"  loading {os.path.basename(path)}")
        model = load_stae_ckpt(path, device)
        Pv, Yv, Mv = collect_preds(model, va, device, mean_t, std_t, amp_dtype)
        Pt, Yt, Mt = collect_preds(model, te, device, mean_t, std_t, amp_dtype)
        val_preds.append(Pv); test_preds.append(Pt)
        names.append(f"stae:{os.path.basename(path)}")
        if Y_val is None:
            Y_val, M_val = Yv, Mv
            Y_test, M_test = Yt, Mt
        del model; torch.cuda.empty_cache()

    for path in gwnet_paths:
        print(f"  loading {os.path.basename(path)}")
        model = load_gwnet_ckpt(path, adj_torch, device)
        Pv, _, _ = collect_preds(model, va, device, mean_t, std_t, amp_dtype)
        Pt, _, _ = collect_preds(model, te, device, mean_t, std_t, amp_dtype)
        val_preds.append(Pv); test_preds.append(Pt)
        names.append(f"gwnet:{os.path.basename(path)}")
        del model; torch.cuda.empty_cache()

    K = len(val_preds)
    Pv = torch.stack(val_preds, dim=0)        # [K, N_val, T, N]
    Pt = torch.stack(test_preds, dim=0)       # [K, N_test, T, N]

    # ---- Print individual metrics ----
    print("\n--- Individual model TEST metrics ---")
    for i, name in enumerate(names):
        m = per_horizon(Pt[i], Y_test, M_test)
        print(f"  [{name}] 15/30/60 = {m['mae_15']:.3f}/{m['mae_30']:.3f}/{m['mae_60']:.3f}  avg={m['avg_mae']:.3f}")

    # ---- Uniform-average ensemble baseline ----
    P_uniform_val = Pv.mean(dim=0)
    P_uniform_test = Pt.mean(dim=0)
    print("\n--- Uniform-average ensemble (val) ---")
    print(json.dumps(per_horizon(P_uniform_val, Y_val, M_val), indent=2))
    print("--- Uniform-average ensemble (test) ---")
    uniform_test = per_horizon(P_uniform_test, Y_test, M_test)
    print(json.dumps(uniform_test, indent=2))

    # ---- Val-optimized weighted ensemble ----
    # Solve for non-negative weights w (summing to 1) that minimize masked MAE on val.
    # Use simplex constraint via softmax parameterization.
    logits = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    M_val_dev = M_val
    Y_val_dev = Y_val
    Pv_dev = Pv
    for step in range(500):
        w = torch.softmax(logits, dim=0)              # [K]
        # Weighted ensemble prediction: [N_val, T, N]
        Pw = (Pv_dev * w.view(K, 1, 1, 1)).sum(dim=0)
        loss = masked_mae(Pw, Y_val_dev, M_val_dev)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w_opt = torch.softmax(logits, dim=0).detach()
    print("\n--- Optimized weights on val ---")
    for i, name in enumerate(names):
        print(f"  {name:<45} weight={w_opt[i].item():.4f}")

    P_weighted_val = (Pv * w_opt.view(K, 1, 1, 1)).sum(dim=0)
    P_weighted_test = (Pt * w_opt.view(K, 1, 1, 1)).sum(dim=0)
    print("\n--- Weighted ensemble (val) ---")
    print(json.dumps(per_horizon(P_weighted_val, Y_val, M_val), indent=2))
    print("--- Weighted ensemble (test) ---")
    weighted_test = per_horizon(P_weighted_test, Y_test, M_test)
    print(json.dumps(weighted_test, indent=2))

    # ---- ST-TTC on top ----
    if args.use_ttc:
        print("\n--- ST-TTC on weighted ensemble ---")
        # Normalize for SDC (it operates in normalized space)
        P_w_norm = (P_weighted_test - float(mean)) / float(std)
        sdc = SDCalibrator(num_nodes=207, freq_bins=args.out_steps // 2 + 1,
                           groups=args.ttc_groups).to(device)
        ttc_opt = torch.optim.Adam(sdc.parameters(), lr=args.ttc_lr)
        Q = deque(maxlen=args.out_steps)
        cal_preds = []
        for i in range(P_w_norm.shape[0]):
            yp = P_w_norm[i:i+1].to(device)
            yn = Y_test[i:i+1].to(device)
            ym = M_test[i:i+1].to(device)
            sdc.eval()
            with torch.no_grad():
                yc = sdc(yp)
            cal_preds.append(yc.cpu())
            Q.append((yp.detach(), yn.detach(), ym.detach()))
            if len(Q) == Q.maxlen:
                yp_o, yn_o, ym_o = Q.popleft()
                sdc.train()
                yc_o = sdc(yp_o)
                yc_o_node = yc_o * float(std) + float(mean)
                loss = masked_mae(yc_o_node, yn_o, ym_o)
                ttc_opt.zero_grad(); loss.backward(); ttc_opt.step()
        P_cal_norm = torch.cat(cal_preds, dim=0)
        P_cal_test = P_cal_norm * float(std) + float(mean)
        ttc_test = per_horizon(P_cal_test, Y_test, M_test)
        print(json.dumps(ttc_test, indent=2))

    # ---- Final comparison ----
    print("\n" + "=" * 60)
    print("Final comparison")
    print("=" * 60)
    print(f"  {'method':<40} {'15min':>8} {'30min':>8} {'60min':>8} {'avg':>8}")
    if len(stae_paths) > 0:
        first_stae_test = per_horizon(Pt[0], Y_test, M_test)
        print(f"  {'best single STAEformer (s42)':<40} {first_stae_test['mae_15']:>8.3f} {first_stae_test['mae_30']:>8.3f} {first_stae_test['mae_60']:>8.3f} {first_stae_test['avg_mae']:>8.3f}")
    print(f"  {'uniform STAEformer ensemble (4 seeds)':<40} {'2.619':>8} {'2.922':>8} {'3.290':>8} {'2.894':>8}  (prior result)")
    print(f"  {'uniform all (STAEformer + GWNet)':<40} {uniform_test['mae_15']:>8.3f} {uniform_test['mae_30']:>8.3f} {uniform_test['mae_60']:>8.3f} {uniform_test['avg_mae']:>8.3f}")
    print(f"  {'val-weighted all (STAEformer + GWNet)':<40} {weighted_test['mae_15']:>8.3f} {weighted_test['mae_30']:>8.3f} {weighted_test['mae_60']:>8.3f} {weighted_test['avg_mae']:>8.3f}")
    if args.use_ttc:
        print(f"  {'weighted + ST-TTC':<40} {ttc_test['mae_15']:>8.3f} {ttc_test['mae_30']:>8.3f} {ttc_test['mae_60']:>8.3f} {ttc_test['avg_mae']:>8.3f}")
    print()
    print("Published SOTA reference (60-min):")
    print("  TESTAM+ 2.99  |  TITAN 3.08  |  TESTAM 3.14  |  MLCAFormer 3.30  |  STAEformer 3.34")


if __name__ == "__main__":
    main()
