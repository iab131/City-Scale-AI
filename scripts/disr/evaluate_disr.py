"""
Evaluate one or more DiSR-Mamba checkpoints with optional ST-TTC calibration
on top. Supports single-model eval and uniform/multi-seed ensembles.

Usage:
    python scripts/disr/evaluate_disr.py \\
        --ckpts results/disr/stageD_magspec_s*/best_disr.pth \\
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data  # noqa: E402
from dataset_v2 import SSSMDataset, split_train_val_test  # noqa: E402
from models.disr.staeformer_wrapper import STAEFrozenWrapper  # noqa: E402
from models.disr.disr_mamba import build_disr_from_config  # noqa: E402
from models.disr.losses import (  # noqa: E402
    per_horizon_metrics, per_speed_regime_mae, masked_mae,
)
from models.disr.spectral_basis import load_or_build_symmetric_basis  # noqa: E402
from models.disr.magnetic_laplacian import magnetic_basis_from_adjacency  # noqa: E402
from models.disr.residual_router import build_sensor_clusters  # noqa: E402
from data_utils import load_adj_pkl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=str, required=True,
                   help="Glob for one or more best_disr.pth checkpoints")
    p.add_argument("--use_ttc", action="store_true")
    p.add_argument("--ttc_groups", type=int, default=4)
    p.add_argument("--ttc_lr", type=float, default=1.0e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--save", type=str, default="")
    return p.parse_args()


class SDCalibrator(torch.nn.Module):
    """ST-TTC FFT amplitude+phase calibrator (zero-init)."""
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
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.permute(0, 2, 1)


@torch.no_grad()
def predict_one_model(blob_path, batch_iter_fn, device):
    """
    Returns (P_norm, Y_true_raw, M, B_norm) where:
      P_norm = final normalised prediction = B_norm + delta_norm
      B_norm = STAEformer's normalised base
    """
    blob = torch.load(blob_path, map_location=device, weights_only=False)
    cfg = blob["cfg"]

    # rebuild trunk
    trunk_ckpt = blob["trunk_ckpt"]
    trunk = STAEFrozenWrapper.from_checkpoint(trunk_ckpt, device=device,
                                              freeze=True)
    if blob.get("trunk_state_dict") is not None:
        trunk.staeformer.load_state_dict(blob["trunk_state_dict"])

    # rebuild bases + clusters
    adj_path = cfg["data"]["adj_path"]
    _, _, A = load_adj_pkl(adj_path)
    A = np.asarray(A, dtype=np.float32)
    k = int(cfg["model"]["k_modes"])
    side = cfg["model"].get("spectral_side", "low")
    q = float(cfg["model"].get("q_charge", 0.10))
    cache_root = os.path.join(cfg["data"]["cache_dir"], "disr")
    U_sym = None
    if cfg["model"].get("use_symmetric_spectral"):
        _, U_sym = load_or_build_symmetric_basis(
            A, k=k, side=side,
            cache_path=os.path.join(cache_root, f"sym_k{k}_{side}.npz"),
        )
        U_sym = U_sym.to(device)
    U_mag = None
    if cfg["model"].get("use_magnetic_spectral"):
        _, U_mag = magnetic_basis_from_adjacency(
            A_sym=A, X_train=None, k=k, q=q, side=side,
            cache_path=os.path.join(cache_root, f"mag_k{k}_q{q:.2f}_{side}.npz"),
        )
        U_mag = U_mag.to(device)
    cluster_id = None
    if cfg["model"].get("use_horizon_cluster_router"):
        path = os.path.join(cache_root,
                            f"clusters_n{cfg['model']['n_clusters']}_v1.npy")
        cluster_id = torch.from_numpy(np.load(path)).long().to(device)

    disr = build_disr_from_config(
        {**cfg, "n_nodes": cfg["data"]["n_nodes"],
         "in_steps": cfg["data"]["in_steps"],
         "out_steps": cfg["data"]["out_steps"]},
        U_sym=U_sym, U_mag=U_mag, cluster_id=cluster_id,
    ).to(device)
    disr.load_state_dict(blob["disr_state_dict"])
    disr.eval()

    use_router = bool(cfg["model"].get("use_horizon_cluster_router", False))

    all_p, all_y, all_m, all_b = [], [], [], []
    for batch in batch_iter_fn():
        x_norm = batch["x_norm"].to(device, non_blocking=True)
        tod_b = batch["tod"].to(device, non_blocking=True)
        dow_b = batch["dow"].to(device, non_blocking=True)
        x_recent_raw = batch["x_node"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
            y_base_norm = trunk.forward_base(x_norm, tod_b, dow_b)
            out = disr(x_norm, tod_b, dow_b,
                       x_recent_raw=x_recent_raw if use_router else None)
            y_pred_norm = y_base_norm + out["delta_y_norm"]
        all_p.append(y_pred_norm.float().cpu())
        all_y.append(batch["y_node"])
        all_m.append(batch["y_mask"])
        all_b.append(y_base_norm.float().cpu())
    return (torch.cat(all_p), torch.cat(all_y),
            torch.cat(all_m), torch.cat(all_b))


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207,
                               cache_dir=args.cache_dir)
    mean = float(data["mean"]); std = float(data["std"])
    arrs = split_train_val_test(
        [data["X"], data["X_norm"], data["tod"], data["dow"],
         data["missing_mask"]], 0.7, 0.1,
    )
    (_, _, X_te), (_, _, Xn_te), (_, _, tod_te), \
        (_, _, dow_te), (_, _, mk_te) = arrs
    te_ds = SSSMDataset(X_te, Xn_te, tod_te, dow_te, mk_te,
                        input_len=12, pred_len=12)
    te_loader = torch.utils.data.DataLoader(
        te_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    def iter_test():
        for b in te_loader:
            yield b

    ckpts = sorted(glob.glob(args.ckpts))
    if not ckpts:
        print(f"No checkpoints found for glob {args.ckpts}")
        sys.exit(1)
    print(f"Found {len(ckpts)} ckpts")

    all_P_norm = []
    Y_cache = None; M_cache = None
    for p in ckpts:
        P_norm, Y_true, M, B_norm = predict_one_model(p, iter_test, device)
        all_P_norm.append(P_norm)
        if Y_cache is None:
            Y_cache = Y_true; M_cache = M
        # report each model alone
        P_node = P_norm * std + mean
        m = per_horizon_metrics(P_node, Y_cache, M_cache)
        print(f"\n[ckpt] {os.path.basename(p)} {json.dumps(m)}")

    # Ensemble = mean of normalised predictions
    if len(all_P_norm) > 1:
        P_ens_norm = torch.stack(all_P_norm).mean(0)
        P_ens = P_ens_norm * std + mean
        ens_m = per_horizon_metrics(P_ens, Y_cache, M_cache)
        print(f"\n[ensemble {len(all_P_norm)}] {json.dumps(ens_m)}")
    else:
        P_ens_norm = all_P_norm[0]
        P_ens = P_ens_norm * std + mean
        ens_m = per_horizon_metrics(P_ens, Y_cache, M_cache)

    # ---- ST-TTC ----
    if args.use_ttc:
        sdc = SDCalibrator(num_nodes=207, freq_bins=12 // 2 + 1,
                            groups=args.ttc_groups).to(device)
        ttc_opt = torch.optim.Adam(sdc.parameters(), lr=args.ttc_lr)
        Q = deque(maxlen=12)
        cal_preds = []
        for i in range(P_ens_norm.shape[0]):
            yp = P_ens_norm[i:i+1].to(device)
            yn = Y_cache[i:i+1].to(device)
            ym = M_cache[i:i+1].to(device)
            sdc.eval()
            with torch.no_grad():
                yc = sdc(yp)
            cal_preds.append(yc.cpu())
            Q.append((yp.detach(), yn.detach(), ym.detach()))
            if len(Q) == Q.maxlen:
                yp_o, yn_o, ym_o = Q.popleft()
                sdc.train()
                yc_o = sdc(yp_o)
                yc_o_raw = yc_o * std + mean
                loss = masked_mae(yc_o_raw, yn_o, ym_o)
                ttc_opt.zero_grad()
                loss.backward()
                ttc_opt.step()
        P_cal_norm = torch.cat(cal_preds, dim=0)
        P_cal = P_cal_norm * std + mean
        ttc_m = per_horizon_metrics(P_cal, Y_cache, M_cache)
        print(f"\n[+ST-TTC g={args.ttc_groups}] {json.dumps(ttc_m)}")
        if args.save:
            np.savez(args.save, y_pred=P_cal.numpy(), y_true=Y_cache.numpy(),
                     y_mask=M_cache.numpy())
    else:
        if args.save:
            np.savez(args.save, y_pred=P_ens.numpy(), y_true=Y_cache.numpy(),
                     y_mask=M_cache.numpy())


if __name__ == "__main__":
    main()
