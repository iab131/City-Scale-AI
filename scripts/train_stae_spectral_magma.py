"""
Train STAE-Spectral-Magma on METR-LA. STAEformer encoder + three-view bi-axis
spectral Mamba sidechain (residual). Trained end-to-end with masked MAE.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from data_utils import load_adj_pkl
from models.disr.spectral_basis import load_or_build_symmetric_basis
from models.disr.magnetic_laplacian import magnetic_basis_from_adjacency
from models.disr.residual_router import build_sensor_clusters
from models.stae_spectral_magma import build_stae_spectral_magma


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/stae_spectral_magma")
    p.add_argument("--tag", type=str, default="stae_spectral_magma")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--N", type=int, default=207)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)

    # STAEformer encoder
    p.add_argument("--input_embedding_dim", type=int, default=24)
    p.add_argument("--tod_embedding_dim", type=int, default=24)
    p.add_argument("--dow_embedding_dim", type=int, default=24)
    p.add_argument("--adaptive_embedding_dim", type=int, default=80)
    p.add_argument("--feed_forward_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--spatial_embedding_dim", type=int, default=0)

    # Spectral sidechain
    p.add_argument("--d_branch", type=int, default=64)
    p.add_argument("--spec_n_layers", type=int, default=2)
    p.add_argument("--spec_dropout", type=float, default=0.10)
    p.add_argument("--k_modes_sym", type=int, default=64)
    p.add_argument("--k_modes_mag", type=int, default=64)
    p.add_argument("--k_modes_sem", type=int, default=64)
    p.add_argument("--q_charge", type=float, default=0.10)
    p.add_argument("--d_sem", type=int, default=24)
    p.add_argument("--k_neighbors", type=int, default=12)
    p.add_argument("--n_clusters", type=int, default=12)
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=1.5)

    p.add_argument("--use_sym", action="store_true", default=True)
    p.add_argument("--use_mag", action="store_true", default=True)
    p.add_argument("--use_sem", action="store_true", default=True)
    p.add_argument("--use_router", action="store_true", default=True)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[30, 50])
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        "avg_mae": masked_mae(pred, true, mask).item(),
        "avg_rmse": masked_rmse(pred, true, mask).item(),
        "avg_mape": masked_mape(pred, true, mask).item(),
    }
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p_t, y_t, m_t = pred[:, t:t+1], true[:, t:t+1], mask[:, t:t+1]
            out[f"mae_{tag}"] = masked_mae(p_t, y_t, m_t).item()
            out[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            out[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return out


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} seed={args.seed} tag={args.tag}", flush=True)

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207,
                               cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    mask = data["missing_mask"]
    mean, std = data["mean"], data["std"]
    print(f"[data] T={len(X)} N={X.shape[1]} mean={mean:.3f} std={std:.3f}",
          flush=True)

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
        (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
        (mk_tr, mk_va, mk_te) = arrs

    def mk(X_p, Xn_p, tod_p, dow_p, mk_p, shuffle):
        ds = SSSMDataset(X_p, Xn_p, tod_p, dow_p, mk_p,
                         input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=False)

    tr = mk(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, True)
    va = mk(X_va, Xn_va, tod_va, dow_va, mk_va, False)
    te = mk(X_te, Xn_te, tod_te, dow_te, mk_te, False)
    print(f"[data] |tr|={len(tr.dataset)} |va|={len(va.dataset)} "
          f"|te|={len(te.dataset)}", flush=True)

    _, _, A = load_adj_pkl(args.adj_path)
    A = np.asarray(A, dtype=np.float32)
    cache_root = os.path.join(args.cache_dir, "disr")
    os.makedirs(cache_root, exist_ok=True)

    U_sym = None
    if args.use_sym:
        _, U_sym = load_or_build_symmetric_basis(
            A, k=args.k_modes_sym, side="low",
            cache_path=os.path.join(cache_root,
                                     f"sym_k{args.k_modes_sym}_low.npz"),
        )
        U_sym = U_sym.to(device)
    U_mag = None
    if args.use_mag:
        _, U_mag = magnetic_basis_from_adjacency(
            A_sym=A, X_train=X_tr, k=args.k_modes_mag, q=args.q_charge,
            side="low",
            cache_path=os.path.join(
                cache_root,
                f"mag_k{args.k_modes_mag}_q{args.q_charge:.2f}_low.npz",
            ),
        )
        U_mag = U_mag.to(device)
    cluster_id = None
    if args.use_router:
        cpath = os.path.join(cache_root, f"clusters_n{args.n_clusters}_v1.npy")
        if os.path.exists(cpath):
            cid = np.load(cpath)
        else:
            cid = build_sensor_clusters(A, n_clusters=args.n_clusters,
                                          X_train=X_tr)
            np.save(cpath, cid)
        cluster_id = torch.from_numpy(cid).long().to(device)

    cfg = {
        "n_nodes": args.N,
        "in_steps": args.in_steps,
        "out_steps": args.out_steps,
        "model": {
            "input_embedding_dim": args.input_embedding_dim,
            "tod_embedding_dim": args.tod_embedding_dim,
            "dow_embedding_dim": args.dow_embedding_dim,
            "adaptive_embedding_dim": args.adaptive_embedding_dim,
            "feed_forward_dim": args.feed_forward_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "spatial_embedding_dim": args.spatial_embedding_dim,
            "d_branch": args.d_branch,
            "spec_n_layers": args.spec_n_layers,
            "spec_dropout": args.spec_dropout,
            "k_modes_sem": args.k_modes_sem,
            "d_sem": args.d_sem, "k_neighbors": args.k_neighbors,
            "n_clusters": args.n_clusters,
            "alpha_init": args.alpha_init, "alpha_max": args.alpha_max,
            "use_sym": args.use_sym, "use_mag": args.use_mag,
            "use_sem": args.use_sem, "use_router": args.use_router,
        },
    }
    model = build_stae_spectral_magma(cfg, U_sym=U_sym, U_mag=U_mag,
                                       cluster_id=cluster_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] STAE-Spectral-Magma params={n_params/1e6:.3f}M "
          f"views={model.spectral_aug.view_names} "
          f"router={'on' if model.spectral_aug.router is not None else 'off'}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=args.lr_milestones, gamma=args.lr_gamma,
    )
    amp_dtype = (torch.bfloat16 if torch.cuda.is_available()
                  and torch.cuda.is_bf16_supported() else torch.float16)
    use_amp = args.use_amp and device.type == "cuda"

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_stae_spec_s{args.seed}.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    max_gpu_mb = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0; nb = 0
        for batch in tr:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            x_recent_raw = batch["x_node"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                y_norm = model(x_norm, tod_b, dow_b,
                                x_recent_raw=x_recent_raw)
                y_pred = y_norm * std_t + mean_t
                loss = masked_mae(y_pred, y_node, y_mask)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.gradient_clip)
            opt.step()
            run += float(loss.detach()); nb += 1
            if torch.cuda.is_available():
                max_gpu_mb = max(max_gpu_mb,
                                  int(torch.cuda.max_memory_allocated() / 1024 / 1024))
        scheduler.step()
        tr_mae = run / max(1, nb)

        model.eval()
        all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                x_recent_raw = batch["x_node"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    yn = model(x_norm, tod_b, dow_b,
                                x_recent_raw=x_recent_raw)
                yp = yn.float() * std_t + mean_t
                all_p.append(yp.cpu())
                all_y.append(batch["y_node"])
                all_m.append(batch["y_mask"])
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_m = per_horizon_metrics(P, Y, M)
        v_mae = val_m["avg_mae"]
        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {epoch:03d}/{args.epochs}] lr={lr_now:.2e} tr_mae={tr_mae:.3f} "
              f"val avg/15/30/60 = {v_mae:.3f} / {val_m['mae_15']:.3f} / "
              f"{val_m['mae_30']:.3f} / {val_m['mae_60']:.3f}  "
              f"gpu={max_gpu_mb}MB", flush=True)

        if v_mae < best_val - 1e-4:
            best_val = v_mae
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_metrics": val_m, "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            x_recent_raw = batch["x_node"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                yn = model(x_norm, tod_b, dow_b, x_recent_raw=x_recent_raw)
            yp = yn.float() * std_t + mean_t
            all_p.append(yp.cpu())
            all_y.append(batch["y_node"])
            all_m.append(batch["y_mask"])
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    te_m = per_horizon_metrics(P, Y, M)
    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(te_m, indent=2)}", flush=True)
    print(f"[done] elapsed={elapsed:.1f}s best_val={best_val:.3f} "
          f"max_gpu={max_gpu_mb}MB", flush=True)

    row = {
        "model": "STAE_Spectral_Magma",
        "seed": args.seed,
        "epochs": ckpt["epoch"],
        "params_M": round(n_params / 1e6, 3),
        "elapsed_sec": round(elapsed, 1),
        **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
        **{f"test_{k}": v for k, v in te_m.items()},
        "tag": args.tag,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = os.path.join(args.out_dir, "stae_spectral_magma_results.csv")
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path),
              index=False)
    print(f"[done] appended to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
