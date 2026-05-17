"""
Train DiSR-Mamba on top of a frozen STAEformer trunk.

Composes:
  * STAEformer trunk: loaded from a checkpoint, frozen unless cfg.trunk.partial_unfreeze=true.
  * DiSR-Mamba residual: experts switchable from cfg.model.use_*.
  * Composite loss: L_main + λ_res * L_res + λ_cong * L_cong + λ_ent * (-H(router)).

Saves per run (out_dir = results/disr/<tag>_s<seed>/):
  - config.yaml, meta.json, best_disr.pth
  - log.csv (per-epoch train/val metrics)
  - test_metrics.json, per_horizon.json, per_speed_regime.json
  - test_predictions.npz (predictions + base + delta) for downstream stacking
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_utils import load_adj_pkl  # noqa: E402
from models.staeformer import STAEformer  # noqa: E402
from models.disr.staeformer_wrapper import STAEFrozenWrapper  # noqa: E402
from models.disr.disr_mamba import build_disr_from_config  # noqa: E402
from models.disr.losses import (  # noqa: E402
    masked_mae, masked_rmse, masked_mape,
    per_horizon_metrics, per_speed_regime_mae,
    disr_composite_loss,
)
from models.disr.spectral_basis import load_or_build_symmetric_basis  # noqa: E402
from models.disr.magnetic_laplacian import magnetic_basis_from_adjacency  # noqa: E402
from models.disr.residual_router import build_sensor_clusters  # noqa: E402

from scripts.disr.common import (  # noqa: E402
    load_config, set_seed, get_amp_dtype, build_dataloaders,
    save_run_metadata, now_iso, git_commit_hash,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, action="append", required=True,
                   help="YAML override config(s). May be passed multiple times; "
                        "later entries override earlier ones. The first entry "
                        "is merged on top of --base_config.")
    p.add_argument("--base_config", type=str,
                   default="configs/disr/base.yaml")
    p.add_argument("--seed", type=int, default=None,
                   help="Overrides cfg.experiment.seed if given")
    p.add_argument("--trunk_ckpt", type=str, default=None,
                   help="Overrides cfg.trunk.ckpt_path if given")
    p.add_argument("--tag_suffix", type=str, default="",
                   help="Appended to cfg.experiment.tag")
    p.add_argument("--out_root", type=str, default=None,
                   help="Overrides cfg.experiment.out_dir if given")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile regardless of config")
    return p.parse_args()


# ----------------------------------------------------------------------------
def resolve_trunk_ckpt(cfg) -> str:
    """Resolve the STAEformer checkpoint path from config."""
    direct = cfg["trunk"].get("ckpt_path", "")
    if direct:
        if not os.path.exists(direct):
            raise FileNotFoundError(f"trunk.ckpt_path not found: {direct}")
        return direct
    glob_p = cfg["trunk"].get("ckpt_glob", "")
    candidates = sorted(glob.glob(glob_p))
    if not candidates:
        raise FileNotFoundError(f"No STAEformer ckpts matched glob {glob_p}")
    pick = cfg["trunk"].get("pick", "best_val")
    if pick == "first":
        return candidates[0]
    # pick best_val: look inside each ckpt's val_metrics, choose lowest avg_mae
    best = None
    for p in candidates:
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            v = c.get("val_metrics", {}).get("avg_mae", float("inf"))
        except Exception:
            v = float("inf")
        if best is None or v < best[0]:
            best = (v, p)
    return best[1]


# ----------------------------------------------------------------------------
def build_bases_and_clusters(cfg, X_train, device):
    """Build symmetric & magnetic spectral bases and sensor clusters."""
    adj_path = cfg["data"]["adj_path"]
    sensor_ids, sensor_idx, A = load_adj_pkl(adj_path)
    A = np.asarray(A, dtype=np.float32)
    N = cfg["data"]["n_nodes"]
    k = int(cfg["model"]["k_modes"])
    side = cfg["model"].get("spectral_side", "low")
    q = float(cfg["model"].get("q_charge", 0.10))
    n_clusters = int(cfg["model"].get("n_clusters", 12))
    cache_root = os.path.join(cfg["data"]["cache_dir"], "disr")
    os.makedirs(cache_root, exist_ok=True)

    U_sym = None
    if cfg["model"].get("use_symmetric_spectral"):
        ev, U = load_or_build_symmetric_basis(
            A, k=k, side=side,
            cache_path=os.path.join(cache_root, f"sym_k{k}_{side}.npz"),
        )
        U_sym = U.to(device)
        print(f"[bases] sym basis K={k} side={side} (cached)")

    U_mag = None
    if cfg["model"].get("use_magnetic_spectral"):
        ev, U = magnetic_basis_from_adjacency(
            A_sym=A, X_train=X_train, k=k, q=q, side=side,
            cache_path=os.path.join(cache_root, f"mag_k{k}_q{q:.2f}_{side}.npz"),
        )
        U_mag = U.to(device)
        print(f"[bases] magnetic basis K={k} q={q} side={side} (cached)")

    cluster_id = None
    if cfg["model"].get("use_horizon_cluster_router"):
        cluster_path = os.path.join(
            cache_root, f"clusters_n{n_clusters}_v1.npy"
        )
        if os.path.exists(cluster_path):
            cid_np = np.load(cluster_path)
        else:
            cid_np = build_sensor_clusters(
                A, n_clusters=n_clusters, X_train=X_train,
                use_history=True, random_state=0,
            )
            np.save(cluster_path, cid_np)
            print(f"[clusters] saved {cluster_path}")
        cluster_id = torch.from_numpy(cid_np).long().to(device)
        # report cluster sizes
        sizes = np.bincount(cid_np, minlength=n_clusters)
        print(f"[clusters] n_clusters={n_clusters} sizes={sizes.tolist()}")
    return U_sym, U_mag, cluster_id


# ----------------------------------------------------------------------------
def evaluate_on_loader(disr, trunk, loader, mean_t, std_t,
                      device, amp_dtype, use_router):
    disr.eval()
    all_p, all_y, all_m, all_b, all_d = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"]
            y_mask = batch["y_mask"]
            x_recent_raw = batch["x_node"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                     enabled=device.type == "cuda"):
                y_base_norm = trunk.forward_base(x_norm, tod_b, dow_b)
                out = disr(x_norm, tod_b, dow_b,
                           x_recent_raw=x_recent_raw if use_router else None)
                delta_norm = out["delta_y_norm"]
                y_pred_norm = y_base_norm + delta_norm
            y_pred = y_pred_norm.float() * std_t + mean_t
            y_base = y_base_norm.float() * std_t + mean_t
            delta_raw = delta_norm.float() * std_t  # (delta is a delta in normed space)
            all_p.append(y_pred.cpu())
            all_y.append(y_node)
            all_m.append(y_mask)
            all_b.append(y_base.cpu())
            all_d.append(delta_raw.cpu())
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    B = torch.cat(all_b); D = torch.cat(all_d)
    metrics = per_horizon_metrics(P, Y, M)
    return P, Y, M, B, D, metrics


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.base_config, args.config)
    # `args.config` is a list now; keep behaviour stable for the rest.
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    if args.trunk_ckpt is not None:
        cfg.setdefault("trunk", {})["ckpt_path"] = args.trunk_ckpt
    if args.no_compile:
        cfg["train"]["use_torch_compile"] = False
    tag = cfg["experiment"]["tag"] + (args.tag_suffix or "")
    seed = int(cfg["experiment"]["seed"])
    out_root = args.out_root or cfg["experiment"]["out_dir"]
    out_dir = os.path.join(ROOT, out_root, f"{tag}_s{seed}")
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(ROOT)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype()
    print(f"[setup] tag={tag} seed={seed} device={device} amp={amp_dtype}")

    # ---- Data --------------------------------------------------------
    tr, va, te, data, splits = build_dataloaders(cfg)
    mean = float(data["mean"]); std = float(data["std"])
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    print(f"[data] |tr|={splits['n_train']} |va|={splits['n_val']} "
          f"|te|={splits['n_test']}  mean={mean:.3f} std={std:.3f}")

    # ---- Trunk -------------------------------------------------------
    trunk_ckpt = resolve_trunk_ckpt(cfg)
    trunk = STAEFrozenWrapper.from_checkpoint(
        trunk_ckpt, device=device,
        freeze=not cfg["trunk"].get("partial_unfreeze", False),
    )
    if cfg["trunk"].get("partial_unfreeze", False):
        trunk.partial_unfreeze(
            last_n_spatial=int(cfg["trunk"].get("partial_unfreeze_last_n", 1)),
            include_output=bool(cfg["trunk"].get("partial_unfreeze_output", True)),
        )
    n_trunk_params = sum(p.numel() for p in trunk.staeformer.parameters())
    n_trunk_train = sum(p.numel() for p in trunk.staeformer.parameters()
                        if p.requires_grad)
    print(f"[trunk] {trunk_ckpt}  total={n_trunk_params/1e6:.2f}M  "
          f"trainable={n_trunk_train/1e6:.3f}M")

    # ---- Bases & Clusters -------------------------------------------
    U_sym, U_mag, cluster_id = build_bases_and_clusters(
        cfg, X_train=splits["train_X"], device=device
    )

    # ---- DiSR-Mamba residual ----------------------------------------
    disr = build_disr_from_config(
        {**cfg, "n_nodes": cfg["data"]["n_nodes"],
         "in_steps": cfg["data"]["in_steps"],
         "out_steps": cfg["data"]["out_steps"]},
        U_sym=U_sym, U_mag=U_mag, cluster_id=cluster_id,
    ).to(device)
    n_disr_params = sum(p.numel() for p in disr.parameters())
    print(f"[disr] params={n_disr_params/1e6:.3f}M  "
          f"experts={disr.expert_names}")

    # optional torch.compile
    if bool(cfg["train"].get("use_torch_compile", False)) and not args.no_compile:
        try:
            disr = torch.compile(disr)  # type: ignore[assignment]
            print("[disr] torch.compile enabled")
        except Exception as e:
            print(f"[disr] torch.compile failed, falling back to eager: {e}")

    # ---- Optimizer / scheduler --------------------------------------
    params = [p for p in disr.parameters() if p.requires_grad]
    if cfg["trunk"].get("partial_unfreeze", False):
        params += [p for p in trunk.staeformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(cfg["train"]["lr_milestones"]),
        gamma=float(cfg["train"]["lr_gamma"]),
    )
    use_amp = bool(cfg["train"].get("use_amp", True))

    # ---- Save metadata ----------------------------------------------
    save_run_metadata(out_dir, cfg, extras={
        "trunk_ckpt": trunk_ckpt,
        "n_disr_params_M": round(n_disr_params / 1e6, 4),
        "n_trunk_params_M": round(n_trunk_params / 1e6, 4),
        "n_trunk_train_M": round(n_trunk_train / 1e6, 4),
        "expert_names": disr.expert_names,
        "seed": seed,
    })

    # ---- Training loop ----------------------------------------------
    use_router = bool(cfg["model"].get("use_horizon_cluster_router", False))
    lambda_res = float(cfg["loss"].get("lambda_res", 0.2))
    lambda_cong = float(cfg["loss"].get("lambda_cong", 0.1))
    lambda_ent = float(cfg["loss"].get("lambda_entropy", 0.0))
    speed_thr = float(cfg["loss"].get("speed_thr", 20.0))
    delta_thr = float(cfg["loss"].get("delta_thr", 5.0))
    grad_clip = float(cfg["train"].get("gradient_clip", 0.0))

    log_csv_path = os.path.join(out_dir, "log.csv")
    log_csv = open(log_csv_path, "w", newline="")
    log_writer = csv.writer(log_csv)
    log_writer.writerow([
        "epoch", "train_loss", "train_L_main", "train_L_res", "train_L_cong",
        "val_mae", "val_mae_15", "val_mae_30", "val_mae_60", "val_rmse_60",
        "lr", "epoch_sec",
    ])

    best_val = float("inf")
    best_ckpt = os.path.join(out_dir, "best_disr.pth")
    epochs_no_improve = 0
    max_gpu_mb = 0
    t_total = time.time()

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        disr.train()
        if cfg["trunk"].get("partial_unfreeze", False):
            trunk.staeformer.train()
        else:
            trunk.staeformer.eval()

        ep_loss = 0.0; ep_main = 0.0; ep_res = 0.0; ep_cong = 0.0; nb = 0
        t_ep = time.time()
        for batch in tr:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)
            x_recent_raw = batch["x_node"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                     enabled=use_amp and device.type == "cuda"):
                y_base_norm = trunk.forward_base(x_norm, tod_b, dow_b)
                out = disr(x_norm, tod_b, dow_b,
                           x_recent_raw=x_recent_raw if use_router else None)
                delta_norm = out["delta_y_norm"]
                y_pred_norm = y_base_norm + delta_norm
                # de-normalise for raw-mph losses
                y_pred = y_pred_norm * std_t + mean_t
                y_base = y_base_norm * std_t + mean_t
                delta_raw = delta_norm * std_t

                router_entropy = None
                if use_router and lambda_ent > 0:
                    router_entropy = out["aux"]["entropy"]
                loss, parts = disr_composite_loss(
                    y_pred=y_pred, y_true=y_node, y_base=y_base,
                    delta_y=delta_raw, y_mask=y_mask,
                    x_recent=x_recent_raw[:, -1:, :],
                    lambda_res=lambda_res, lambda_cong=lambda_cong,
                    speed_thr=speed_thr, delta_thr=delta_thr,
                    router_entropy=router_entropy,
                    lambda_entropy=lambda_ent,
                )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

            ep_loss += float(loss.detach())
            ep_main += parts["L_main"]
            ep_res  += parts["L_res"]
            ep_cong += parts["L_cong"]
            nb += 1
            if torch.cuda.is_available():
                max_gpu_mb = max(max_gpu_mb,
                                  int(torch.cuda.max_memory_allocated() / 1024 / 1024))
        scheduler.step()

        # ---- validation ----
        _, _, _, _, _, val_m = evaluate_on_loader(
            disr, trunk, va, mean_t, std_t, device, amp_dtype, use_router,
        )
        lr_now = opt.param_groups[0]["lr"]
        ep_sec = time.time() - t_ep
        log_writer.writerow([
            epoch, ep_loss / max(1, nb), ep_main / max(1, nb),
            ep_res / max(1, nb), ep_cong / max(1, nb),
            val_m["avg_mae"], val_m["mae_15"], val_m["mae_30"], val_m["mae_60"],
            val_m["rmse_60"], lr_now, ep_sec,
        ])
        log_csv.flush()
        print(f"[ep {epoch:03d}/{cfg['train']['epochs']}] "
              f"lr={lr_now:.2e} loss={ep_loss/max(1,nb):.4f} "
              f"(main={ep_main/max(1,nb):.3f} res={ep_res/max(1,nb):.3f} "
              f"cong={ep_cong/max(1,nb):.3f}) "
              f"val 15/30/60={val_m['mae_15']:.3f}/"
              f"{val_m['mae_30']:.3f}/{val_m['mae_60']:.3f} "
              f"avg={val_m['avg_mae']:.3f}  gpu={max_gpu_mb}MB  "
              f"{ep_sec:.1f}s",
              flush=True)

        if val_m["avg_mae"] < best_val - 1e-4:
            best_val = val_m["avg_mae"]
            epochs_no_improve = 0
            # save the residual model only; trunk is unchanged unless partial_unfreeze
            torch.save({
                "disr_state_dict": (disr._orig_mod.state_dict()
                                    if hasattr(disr, "_orig_mod")
                                    else disr.state_dict()),
                "trunk_state_dict": (trunk.staeformer.state_dict()
                                     if cfg["trunk"].get("partial_unfreeze", False)
                                     else None),
                "trunk_ckpt": trunk_ckpt,
                "cfg": cfg,
                "epoch": epoch,
                "val_metrics": val_m,
            }, best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(cfg["train"]["patience"]):
                print(f"[early stop] no improvement for "
                      f"{cfg['train']['patience']} epochs")
                break

    log_csv.close()

    # ---- Final test on best checkpoint ------------------------------
    blob = torch.load(best_ckpt, map_location=device, weights_only=False)
    target = disr._orig_mod if hasattr(disr, "_orig_mod") else disr
    target.load_state_dict(blob["disr_state_dict"])
    if blob.get("trunk_state_dict") is not None:
        trunk.staeformer.load_state_dict(blob["trunk_state_dict"])

    P, Y, M, B, D, te_m = evaluate_on_loader(
        disr, trunk, te, mean_t, std_t, device, amp_dtype, use_router,
    )

    # per-speed-regime + per-horizon details
    speed_regime = per_speed_regime_mae(P, Y, M)
    per_h_table = {}
    for h in range(P.shape[1]):
        m = per_horizon_metrics(P[:, h:h+1], Y[:, h:h+1], M[:, h:h+1])
        per_h_table[f"h{h+1}"] = m

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(te_m, f, indent=2)
    with open(os.path.join(out_dir, "per_horizon.json"), "w") as f:
        json.dump(per_h_table, f, indent=2)
    with open(os.path.join(out_dir, "per_speed_regime.json"), "w") as f:
        json.dump(speed_regime, f, indent=2)
    # save test predictions for downstream stacking / ST-TTC
    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             y_pred=P.numpy(), y_true=Y.numpy(), y_mask=M.numpy(),
             y_base=B.numpy(), delta=D.numpy())

    elapsed = time.time() - t_total
    summary = {
        "tag": tag,
        "seed": seed,
        "trunk_ckpt": trunk_ckpt,
        "best_val_mae": best_val,
        "test_metrics": te_m,
        "elapsed_sec": round(elapsed, 1),
        "max_gpu_mb": max_gpu_mb,
        "git_commit": git_commit_hash(),
        "timestamp": now_iso(),
        "expert_names": disr.expert_names if hasattr(disr, "expert_names")
                         else disr._orig_mod.expert_names,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {json.dumps(te_m)}  elapsed={elapsed:.1f}s "
          f"max_gpu={max_gpu_mb}MB")

    # Append to a CSV leaderboard
    csv_path = os.path.join(ROOT, "results", "disr", "disr_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "tag", "seed", "trunk_ckpt", "experts",
                "best_val_mae", "test_mae_15", "test_mae_30", "test_mae_60",
                "test_avg_mae", "test_rmse_60", "elapsed_sec", "max_gpu_mb",
                "timestamp",
            ])
        w.writerow([
            tag, seed, trunk_ckpt,
            "|".join(summary["expert_names"]),
            round(best_val, 4),
            round(te_m["mae_15"], 4), round(te_m["mae_30"], 4),
            round(te_m["mae_60"], 4), round(te_m["avg_mae"], 4),
            round(te_m.get("rmse_60", float("nan")), 4),
            round(elapsed, 1), max_gpu_mb,
            now_iso(),
        ])


if __name__ == "__main__":
    main()
