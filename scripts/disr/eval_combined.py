"""
Combined ensemble evaluation for STAEformer trunk checkpoints AND DiSR-Mamba
checkpoints, optionally followed by ST-TTC v2 spectral calibration.

Usage:
    python scripts/disr/eval_combined.py \\
        --stae_ckpts 'results/staeformer/stae_trunk*/best_stae_s*.pth' \\
        --disr_ckpts 'results/disr/*_s*/best_disr.pth' \\
        --use_ttc --ttc_groups 4
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from data_utils import load_adj_pkl
from models.staeformer import STAEformer
from models.disr.staeformer_wrapper import STAEFrozenWrapper
from models.disr.disr_mamba import build_disr_from_config
from models.disr.losses import per_horizon_metrics, per_speed_regime_mae, masked_mae
from models.disr.spectral_basis import load_or_build_symmetric_basis
from models.disr.magnetic_laplacian import magnetic_basis_from_adjacency


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stae_ckpts", type=str, default="")
    p.add_argument("--disr_ckpts", type=str, default="")
    p.add_argument("--use_ttc", action="store_true")
    p.add_argument("--ttc_groups", type=int, default=4)
    p.add_argument("--ttc_lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", type=str,
                   default="results/disr/combined_ensemble_metrics.json")
    return p.parse_args()


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
            s = g * self.group_size
            e = M if g == self.groups - 1 else (g + 1) * self.group_size
            la = self.lambda_amp[g].unsqueeze(0)
            lp = self.lambda_phi[g].unsqueeze(0)
            Yf_corr[:, :, s:e] = (A[:, :, s:e] * (1 + la)) * \
                                 torch.exp(1j * (P[:, :, s:e] + lp))
        return torch.fft.irfft(Yf_corr, n=T, dim=-1).permute(0, 2, 1)


@torch.no_grad()
def stae_predict(ckpt_path, te_loader, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = STAEformer(
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
    m.load_state_dict(ckpt["model"])
    preds = []
    for batch in te_loader:
        x_norm = batch["x_norm"].to(device, non_blocking=True)
        tod_b = batch["tod"].to(device, non_blocking=True)
        dow_b = batch["dow"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            yn = m(x_norm, tod_b, dow_b)
        preds.append(yn.float().cpu())
    del m
    torch.cuda.empty_cache()
    return torch.cat(preds)


@torch.no_grad()
def disr_predict(ckpt_path, te_loader, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    trunk = STAEFrozenWrapper.from_checkpoint(blob["trunk_ckpt"], device=device, freeze=True)
    if blob.get("trunk_state_dict") is not None:
        trunk.staeformer.load_state_dict(blob["trunk_state_dict"])

    _, _, A = load_adj_pkl(cfg["data"]["adj_path"])
    A = np.asarray(A, dtype=np.float32)
    k = int(cfg["model"]["k_modes"])
    side = cfg["model"].get("spectral_side", "low")
    q = float(cfg["model"].get("q_charge", 0.10))
    cache_root = os.path.join(cfg["data"]["cache_dir"], "disr")
    U_sym = None
    if cfg["model"].get("use_symmetric_spectral"):
        _, U_sym = load_or_build_symmetric_basis(
            A, k=k, side=side,
            cache_path=os.path.join(cache_root, f"sym_k{k}_{side}.npz"))
        U_sym = U_sym.to(device)
    U_mag = None
    if cfg["model"].get("use_magnetic_spectral"):
        _, U_mag = magnetic_basis_from_adjacency(
            A_sym=A, X_train=None, k=k, q=q, side=side,
            cache_path=os.path.join(cache_root, f"mag_k{k}_q{q:.2f}_{side}.npz"))
        U_mag = U_mag.to(device)
    cluster_id = None
    if cfg["model"].get("use_horizon_cluster_router"):
        cluster_id = torch.from_numpy(np.load(
            os.path.join(cache_root, f"clusters_n{cfg['model']['n_clusters']}_v1.npy")
        )).long().to(device)
    disr = build_disr_from_config(
        {**cfg, "n_nodes": cfg["data"]["n_nodes"],
         "in_steps": cfg["data"]["in_steps"],
         "out_steps": cfg["data"]["out_steps"]},
        U_sym=U_sym, U_mag=U_mag, cluster_id=cluster_id,
    ).to(device)
    disr.load_state_dict(blob["disr_state_dict"])
    disr.eval()
    use_router = bool(cfg["model"].get("use_horizon_cluster_router", False))

    preds = []
    for batch in te_loader:
        x_norm = batch["x_norm"].to(device, non_blocking=True)
        tod_b = batch["tod"].to(device, non_blocking=True)
        dow_b = batch["dow"].to(device, non_blocking=True)
        x_recent_raw = batch["x_node"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            y_base_norm = trunk.forward_base(x_norm, tod_b, dow_b)
            out = disr(x_norm, tod_b, dow_b, x_recent_raw=x_recent_raw if use_router else None)
            y_pred_norm = y_base_norm + out["delta_y_norm"]
        preds.append(y_pred_norm.float().cpu())
    del disr, trunk
    torch.cuda.empty_cache()
    return torch.cat(preds)


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_cached_v2_data("data/METR-LA.h5", "data/adj_METR-LA.pkl", k=207, cache_dir="cache/gft")
    mean, std = float(data["mean"]), float(data["std"])
    arrs = split_train_val_test([data["X"], data["X_norm"], data["tod"],
                                  data["dow"], data["missing_mask"]], 0.7, 0.1)
    (_, _, X_te), (_, _, Xn_te), (_, _, tod_te), \
        (_, _, dow_te), (_, _, mk_te) = arrs
    te_ds = SSSMDataset(X_te, Xn_te, tod_te, dow_te, mk_te, 12, 12)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    # Capture y_true / y_mask once
    Y, M = [], []
    for b in te_loader:
        Y.append(b["y_node"]); M.append(b["y_mask"])
    Y = torch.cat(Y); M = torch.cat(M)

    stae_paths = sorted(glob.glob(args.stae_ckpts)) if args.stae_ckpts else []
    disr_paths = sorted(glob.glob(args.disr_ckpts)) if args.disr_ckpts else []
    print(f"[combine] {len(stae_paths)} STAE ckpts, {len(disr_paths)} DiSR ckpts")

    all_norm = []
    for p in stae_paths:
        Pn = stae_predict(p, te_loader, device)
        Pr = Pn * std + mean
        m = per_horizon_metrics(Pr, Y, M)
        print(f"[STAE]  {os.path.basename(p):<30}  mae_15/30/60 = "
              f"{m['mae_15']:.4f} / {m['mae_30']:.4f} / {m['mae_60']:.4f}")
        all_norm.append(Pn)
    for p in disr_paths:
        Pn = disr_predict(p, te_loader, device)
        Pr = Pn * std + mean
        m = per_horizon_metrics(Pr, Y, M)
        print(f"[DiSR]  {os.path.basename(os.path.dirname(p)):<30}  mae_15/30/60 = "
              f"{m['mae_15']:.4f} / {m['mae_30']:.4f} / {m['mae_60']:.4f}")
        all_norm.append(Pn)

    if not all_norm:
        print("no predictions")
        sys.exit(1)

    P_ens_norm = torch.stack(all_norm).mean(0)
    P_ens = P_ens_norm * std + mean
    ens_m = per_horizon_metrics(P_ens, Y, M)
    print(f"\n[ENSEMBLE {len(all_norm)}] {json.dumps(ens_m)}")
    psr = per_speed_regime_mae(P_ens, Y, M)
    print(f"[ENSEMBLE per_speed_regime] {json.dumps(psr)}")

    metrics_to_save = {"ensemble": ens_m, "per_speed_regime": psr,
                       "n_models": len(all_norm),
                       "stae_paths": stae_paths, "disr_paths": disr_paths}

    if args.use_ttc:
        sdc = SDCalibrator(num_nodes=207, freq_bins=12 // 2 + 1, groups=args.ttc_groups).to(device)
        opt = torch.optim.Adam(sdc.parameters(), lr=args.ttc_lr)
        Q = deque(maxlen=12)
        cal = []
        for i in range(P_ens_norm.shape[0]):
            yp = P_ens_norm[i:i+1].to(device)
            yn = Y[i:i+1].to(device); ym = M[i:i+1].to(device)
            sdc.eval()
            with torch.no_grad():
                yc = sdc(yp)
            cal.append(yc.cpu())
            Q.append((yp.detach(), yn.detach(), ym.detach()))
            if len(Q) == Q.maxlen:
                yp_o, yn_o, ym_o = Q.popleft()
                sdc.train()
                yc_o = sdc(yp_o)
                loss = masked_mae(yc_o * std + mean, yn_o, ym_o)
                opt.zero_grad(); loss.backward(); opt.step()
        P_cal_norm = torch.cat(cal, dim=0)
        P_cal = P_cal_norm * std + mean
        ttc_m = per_horizon_metrics(P_cal, Y, M)
        print(f"\n[+ST-TTC v2 g={args.ttc_groups}] {json.dumps(ttc_m)}")
        for k in ["mae_15", "mae_30", "mae_60", "avg_mae"]:
            print(f"  {k:<10} {ens_m[k]:.4f} -> {ttc_m[k]:.4f}  "
                  f"(delta {ttc_m[k]-ens_m[k]:+.4f})")
        metrics_to_save["with_ttc"] = ttc_m

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics_to_save, f, indent=2, default=float)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
