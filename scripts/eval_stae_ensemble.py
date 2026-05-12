"""
Evaluate a multi-seed STAEformer (and optional hybrid) ensemble on the test set.
Optionally apply ST-TTC test-time calibration (FFT-based) on top.

ST-TTC reference: arXiv:2506.00635 (NeurIPS 2025 Spotlight)
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
from collections import deque
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.staeformer import STAEformer
from models.hybrid import HybridSTAEMamba


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stae_ckpts", type=str,
                   default="results/staeformer/stae_*/best_stae_s*.pth",
                   help="Glob for STAEformer checkpoints")
    p.add_argument("--hybrid_ckpts", type=str,
                   default="results/hybrid/hybrid_*/best_hybrid_s*.pth",
                   help="Glob for hybrid checkpoints (set empty to skip)")
    p.add_argument("--include_hybrid", action="store_true",
                   help="Include hybrid models in ensemble")
    p.add_argument("--use_ttc", action="store_true",
                   help="Apply ST-TTC test-time calibration")
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
    """ST-TTC Spectral-Domain Calibrator (FFT-based amplitude+phase modulation).
    Operates on predictions of shape [B, T_out, N] in normalized space.
    """
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = freq_bins // groups
        self.lambda_amp = torch.nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = torch.nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        # y_pred: [B, T, N] (normalized space). Reshape to [B, N, T].
        B, T, N = y_pred.shape
        y = y_pred.permute(0, 2, 1)                          # [B, N, T]
        Yf = torch.fft.rfft(y, dim=-1)                       # [B, N, M=T//2+1]
        A = torch.abs(Yf); P = torch.angle(Yf)
        Yf_corr = torch.zeros_like(Yf)
        M = T // 2 + 1
        for g in range(self.groups):
            start = g * self.group_size
            end = M if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)          # [1, N, 1]
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)       # [B, N, T]
        return y_time.permute(0, 2, 1)                       # [B, T, N]


def load_stae_ckpt(ckpt_path, U, evals, device):
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


def load_hybrid_ckpt(ckpt_path, U, evals, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = HybridSTAEMamba(
        N=207, U=U, evals=evals,
        in_steps=a["in_steps"], out_steps=a["out_steps"],
        adaptive_embedding_dim=a["adaptive_embedding_dim"],
        feed_forward_dim=a["feed_forward_dim"],
        num_heads=a["num_heads"], num_layers=a["num_layers"],
        dropout=a["dropout"],
        spec_d=a["spec_d"], spec_layers=a["spec_layers"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]
    U_np, evals_np = data["U"], data["evals"]
    U = torch.from_numpy(U_np).float().to(device)
    evals = torch.from_numpy(evals_np).float().to(device)

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (_, _, X_te), (_, _, Xn_te), (_, _, tod_te), \
        (_, _, dow_te), (_, _, mk_te) = arrs
    te_ds = SSSMDataset(X_te, Xn_te, tod_te, dow_te, mk_te,
                        input_len=args.in_steps, pred_len=args.out_steps)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    print(f"|test|={len(te_ds)}")

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    # Collect checkpoints
    stae_paths = sorted(glob.glob(args.stae_ckpts))
    hybrid_paths = sorted(glob.glob(args.hybrid_ckpts)) if args.include_hybrid else []
    paths = [(p, "stae") for p in stae_paths] + [(p, "hybrid") for p in hybrid_paths]
    if not paths:
        print("NO checkpoints found"); sys.exit(1)
    print(f"Found {len(paths)} checkpoints:")
    for p, k in paths:
        print(f"  [{k}] {p}")

    # Run each, collect normalized predictions
    all_pred_norm = []
    Y_cache = None; M_cache = None
    for i, (path, kind) in enumerate(paths):
        if kind == "stae":
            model = load_stae_ckpt(path, U, evals, device)
        else:
            model = load_hybrid_ckpt(path, U, evals, device)
        preds = []; ys = []; ms = []
        with torch.no_grad():
            for batch in te_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    yn = model(x_norm, tod_b, dow_b)
                preds.append(yn.float().cpu())
                if i == 0:
                    ys.append(batch["y_node"]); ms.append(batch["y_mask"])
        P_norm = torch.cat(preds)
        all_pred_norm.append(P_norm)
        if i == 0:
            Y_cache = torch.cat(ys); M_cache = torch.cat(ms)
        P_node = P_norm * float(std) + float(mean)
        m_i = per_horizon(P_node, Y_cache, M_cache)
        print(f"\n[{kind} {os.path.basename(path)}] {json.dumps(m_i)}")

    # Ensemble = mean of normalized predictions
    P_ens_norm = torch.stack(all_pred_norm).mean(dim=0)        # [B, T, N]
    P_ens_node = P_ens_norm * float(std) + float(mean)
    ens_metrics = per_horizon(P_ens_node, Y_cache, M_cache)

    print("\n" + "=" * 60)
    print(f"ENSEMBLE ({len(paths)} models)")
    print("=" * 60)
    print(json.dumps(ens_metrics, indent=2))

    # Optionally apply ST-TTC
    if args.use_ttc:
        print("\n" + "=" * 60)
        print("ST-TTC test-time calibration (single-pass)")
        print("=" * 60)
        # Apply ST-TTC on the ensemble's normalized predictions.
        # ST-TTC normally streams, but here we do a one-shot fit on test
        # predictions using the calibrator trained over many test windows.
        sdc = SDCalibrator(num_nodes=207, freq_bins=args.out_steps // 2 + 1,
                           groups=args.ttc_groups).to(device)
        ttc_opt = torch.optim.Adam(sdc.parameters(), lr=args.ttc_lr)

        # Streaming flash update: process samples in order, calibrate, then update
        # on the dequeued (now-observed) sample.
        Q = deque(maxlen=args.out_steps)
        calibrated_preds = []
        for i in range(P_ens_norm.shape[0]):
            yp = P_ens_norm[i:i+1].to(device)                    # [1, T, N]
            yn = Y_cache[i:i+1].to(device)                       # [1, T, N]
            ym = M_cache[i:i+1].to(device)                       # [1, T, N]

            # Calibrate (eval mode)
            sdc.eval()
            with torch.no_grad():
                yc = sdc(yp)
            calibrated_preds.append(yc.cpu())

            # FIFO update
            Q.append((yp.detach(), yn.detach(), ym.detach()))
            if len(Q) == Q.maxlen:
                yp_o, yn_o, ym_o = Q.popleft()
                sdc.train()
                yc_o = sdc(yp_o)
                yc_o_node = yc_o * float(std) + float(mean)
                loss = masked_mae(yc_o_node, yn_o, ym_o)
                ttc_opt.zero_grad()
                loss.backward()
                ttc_opt.step()

        P_cal_norm = torch.cat(calibrated_preds, dim=0)
        P_cal_node = P_cal_norm * float(std) + float(mean)
        ttc_metrics = per_horizon(P_cal_node, Y_cache, M_cache)
        print(json.dumps(ttc_metrics, indent=2))

        # Print summary table comparing pre-/post-TTC
        print()
        print("Ensemble pre vs post ST-TTC:")
        for k in ["mae_15", "mae_30", "mae_60", "avg_mae"]:
            print(f"  {k:<10} {ens_metrics[k]:.4f} -> {ttc_metrics[k]:.4f}  "
                  f"(delta {ttc_metrics[k]-ens_metrics[k]:+.4f})")

    # Comparison to SOTA
    print("\n" + "=" * 60)
    print("vs Published SOTA on METR-LA test")
    print("=" * 60)
    final = ens_metrics if not args.use_ttc else ttc_metrics
    table = [
        ("ours (ensemble)", final["mae_15"], final["mae_30"], final["mae_60"]),
        ("STAEformer (CIKM 2023)", 2.65, 2.97, 3.34),
        ("MLCAFormer (PLOS One 2025)", None, None, 3.30),
        ("TITAN (arXiv 2024)", 2.41, 2.72, 3.08),
        ("TESTAM+ (arXiv 2025)", None, None, 2.99),
    ]
    print(f"  {'model':<30} {'15min':>8} {'30min':>8} {'60min':>8}")
    for name, h15, h30, h60 in table:
        s15 = f"{h15:.3f}" if h15 is not None else "  --  "
        s30 = f"{h30:.3f}" if h30 is not None else "  --  "
        s60 = f"{h60:.3f}" if h60 is not None else "  --  "
        print(f"  {name:<30} {s15:>8} {s30:>8} {s60:>8}")


if __name__ == "__main__":
    main()
