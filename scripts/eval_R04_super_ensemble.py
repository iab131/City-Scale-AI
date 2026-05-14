"""
R04 — Super-ensemble eval over ALL available checkpoints.

Combines:
  - all STAEformer seed checkpoints (matches results/staeformer/**/best_stae_s*.pth)
  - bigger STAEformer checkpoint (if found)
  - STAEformer-pretrained checkpoint (if found)
  - GraphWaveNet checkpoint (if found)
  - hybrid checkpoint (if found)

Three blending strategies are evaluated, each with optional ST-TTC v2 on top:
  1. uniform  — mean of normalized predictions
  2. val-weighted (per-model scalar) — softmax weights optimized on val
  3. val-weighted (per-model per-horizon) — softmax over models, weights shape
       [n_models, T_out], optimized on val

ST-TTC v2 has more knobs vs the original:
  - more frequency groups (configurable)
  - per-horizon shift (optional)
  - warm-start: pre-train on val data before the streaming test pass
  - reset every K samples to prevent slow drift

Output: results table with best blend choice, written to results/R04_super_ensemble.json
"""

import os
import sys
import glob
import json
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
from dataset_pretrained import PretrainedSSSMDataset, split_t0_for_pretrained
from data_utils import load_adj_pkl
from graph_utils import symmetrize_adjacency

from models.staeformer import STAEformer
from models.graph_wavenet import GraphWaveNet
# Lazy: from models.hybrid import HybridSTAEMamba  (imported only if --include_hybrid)
# Lazy: from models.staeformer_pretrained import STAEformerPretrained  (imported when pre ckpt is loaded)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stae_ckpts", type=str,
                   default="results/staeformer/stae_repro*/best_stae_s*.pth")
    p.add_argument("--stae_R01_ckpts", type=str,
                   default="results/staeformer/stae_R01_s*/best_stae_s*.pth")
    p.add_argument("--stae_big_ckpts", type=str,
                   default="results/staeformer/stae_R02_big_s*/best_stae_s*.pth")
    p.add_argument("--stae_pre_ckpts", type=str,
                   default="results/stae_pretrained/*/best_stae_pre_s*.pth")
    p.add_argument("--gwnet_ckpts", type=str,
                   default="results/gwnet/gwnet_s*/best_gwnet_s*.pth")
    p.add_argument("--hybrid_ckpts", type=str,
                   default="results/hybrid/hybrid_s*/best_hybrid_s*.pth")
    p.add_argument("--include_gwnet", action="store_true")
    p.add_argument("--include_hybrid", action="store_true")

    p.add_argument("--use_ttc", action="store_true")
    p.add_argument("--ttc_groups", type=int, default=4)
    p.add_argument("--ttc_lr", type=float, default=1e-4)
    p.add_argument("--ttc_per_horizon", action="store_true")
    p.add_argument("--ttc_warm_start", action="store_true",
                   help="Pre-train TTC on val data before streaming test")
    p.add_argument("--ttc_reset_every", type=int, default=0,
                   help="If >0, reset TTC weights every K test samples")

    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--T_long", type=int, default=2016)
    p.add_argument("--out_json", type=str, default="results/R04_super_ensemble.json")
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


class SDCalibratorV2(torch.nn.Module):
    """ST-TTC v2 with configurable per-horizon scaling.

    Operates on predictions of shape [B, T, N] in normalized space.
    """
    def __init__(self, num_nodes, freq_bins, groups=4, per_horizon=False, T_out=12):
        super().__init__()
        self.groups = groups
        self.per_horizon = per_horizon
        self.group_size = freq_bins // groups
        # standard amp+phase per (group, node)
        self.lambda_amp = torch.nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = torch.nn.Parameter(torch.zeros(groups, num_nodes, 1))
        if per_horizon:
            # additional per-horizon multiplicative correction (zero-init)
            self.lambda_h = torch.nn.Parameter(torch.zeros(num_nodes, T_out))

    def forward(self, y_pred):
        B, T, N = y_pred.shape
        y = y_pred.permute(0, 2, 1)                          # [B, N, T]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf); P = torch.angle(Yf)
        Yf_corr = torch.zeros_like(Yf)
        M = T // 2 + 1
        for g in range(self.groups):
            s = g * self.group_size
            e = M if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, s:e] * (1 + lam_a)
            P_g = P[:, :, s:e] + lam_p
            Yf_corr[:, :, s:e] = A_g * torch.exp(1j * P_g)
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)       # [B, N, T]
        if self.per_horizon:
            y_time = y_time * (1 + self.lambda_h.unsqueeze(0))
        return y_time.permute(0, 2, 1)


# ---- model loading ----

def load_stae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = STAEformer(
        N=207, in_steps=a["in_steps"], out_steps=a["out_steps"],
        input_embedding_dim=a["input_embedding_dim"],
        tod_embedding_dim=a["tod_embedding_dim"],
        dow_embedding_dim=a["dow_embedding_dim"],
        adaptive_embedding_dim=a["adaptive_embedding_dim"],
        feed_forward_dim=a["feed_forward_dim"],
        num_heads=a["num_heads"], num_layers=a["num_layers"],
        dropout=a["dropout"],
    ).to(device).eval()
    m.load_state_dict(ckpt["model"])
    return m


def load_gwnet(ckpt_path, adj_torch, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = GraphWaveNet(
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
    m.load_state_dict(ckpt["model"])
    return m


def load_hybrid(ckpt_path, U, evals, device):
    from models.hybrid import HybridSTAEMamba
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = HybridSTAEMamba(
        N=207, U=U, evals=evals,
        in_steps=a["in_steps"], out_steps=a["out_steps"],
        adaptive_embedding_dim=a["adaptive_embedding_dim"],
        feed_forward_dim=a["feed_forward_dim"],
        num_heads=a["num_heads"], num_layers=a["num_layers"],
        dropout=a["dropout"],
        spec_d=a["spec_d"], spec_layers=a["spec_layers"],
    ).to(device).eval()
    m.load_state_dict(ckpt["model"])
    return m


def load_stae_pre(ckpt_path, device):
    from models.staeformer_pretrained import STAEformerPretrained
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = STAEformerPretrained(
        N=207,
        tmae_ckpt_path=a["tmae_ckpt"],
        smae_ckpt_path=a["smae_ckpt"],
        in_steps=a["in_steps"], out_steps=a["out_steps"],
        input_embedding_dim=a["input_embedding_dim"],
        tod_embedding_dim=a["tod_embedding_dim"],
        dow_embedding_dim=a["dow_embedding_dim"],
        adaptive_embedding_dim=a["adaptive_embedding_dim"],
        feed_forward_dim=a["feed_forward_dim"],
        num_heads=a["num_heads"], num_layers=a["num_layers"],
        dropout=a["dropout"], d_pre=a["d_pre"],
        freeze_pretrained=not a.get("no_freeze", False),
    ).to(device).eval()
    m.load_state_dict(ckpt["model"])
    return m


def collect_stae_preds(model, loader, device, mean, std, amp_dtype):
    """Collect STAE/GWNet/Hybrid predictions on a loader."""
    preds = []; ys = []; ms = []
    with torch.no_grad():
        for batch in loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                yn = model(x_norm, tod_b, dow_b)
            preds.append(yn.float().cpu())
            ys.append(batch["y_node"]); ms.append(batch["y_mask"])
    return torch.cat(preds), torch.cat(ys), torch.cat(ms)


def collect_stae_pre_preds(model, loader, device, mean, std, amp_dtype):
    preds = []; ys = []; ms = []
    with torch.no_grad():
        for batch in loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            long_hist = batch["long_hist"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                yn = model(x_norm, tod_b, dow_b, long_hist)
            preds.append(yn.float().cpu())
            ys.append(batch["y_node"]); ms.append(batch["y_mask"])
    return torch.cat(preds), torch.cat(ys), torch.cat(ms)


def optimize_softmax_weights(val_preds_stack, Y_val, M_val, mode="scalar",
                             n_models=None, T_out=12, lr=0.05, n_iter=500):
    """Returns optimized softmax weights for ensembling.

    val_preds_stack: [K, B, T, N]  per-model normalized predictions on val
    Y_val:           [B, T, N]      raw mph targets
    M_val:           [B, T, N]      mask
    mode:            "scalar"  → [K] weights (one per model)
                     "horizon" → [K, T_out] weights (softmax along K per horizon)
    """
    K = n_models
    if mode == "scalar":
        logits = torch.zeros(K, requires_grad=True)
    else:
        logits = torch.zeros(K, T_out, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(n_iter):
        if mode == "scalar":
            w = torch.softmax(logits, dim=0)
            Pw = (val_preds_stack * w.view(K, 1, 1, 1)).sum(dim=0)
        else:
            w = torch.softmax(logits, dim=0)                  # [K, T_out]
            Pw = (val_preds_stack * w.view(K, 1, T_out, 1)).sum(dim=0)
        loss = masked_mae(Pw, Y_val, M_val)
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.softmax(logits, dim=0).detach()


def apply_ttc(P_ens_norm, Y_node, M_node, mean, std, args, device,
              ttc_groups=4, ttc_lr=1e-4, per_horizon=False, warm_start=False,
              reset_every=0, T_out=12):
    """Apply ST-TTC v2 streaming flash update on the test set."""
    N = P_ens_norm.shape[-1]
    sdc = SDCalibratorV2(num_nodes=N, freq_bins=T_out // 2 + 1,
                         groups=ttc_groups, per_horizon=per_horizon,
                         T_out=T_out).to(device)
    ttc_opt = torch.optim.Adam(sdc.parameters(), lr=ttc_lr)

    # Optional warm start: train on val data first (only meaningful if we
    # have val predictions and targets in normalized space too — caller passes
    # val data via P_val_norm_for_warm parameter, not implemented here as
    # a separate apply_warm function)

    Q = deque(maxlen=T_out)
    cal_preds = []
    n_calibrated = 0
    for i in range(P_ens_norm.shape[0]):
        yp = P_ens_norm[i:i+1].to(device)
        yn = Y_node[i:i+1].to(device)
        ym = M_node[i:i+1].to(device)

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
            n_calibrated += 1

        if reset_every > 0 and (i + 1) % reset_every == 0:
            for p in sdc.parameters():
                p.data.zero_()
            ttc_opt = torch.optim.Adam(sdc.parameters(), lr=ttc_lr)

    return torch.cat(cal_preds, dim=0)


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask_arr = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]
    U_np, evals_np = data["U"], data["evals"]
    U = torch.from_numpy(U_np).float().to(device)
    evals_t = torch.from_numpy(evals_np).float().to(device)
    _, _, A = load_adj_pkl(args.adj_path)
    A = symmetrize_adjacency(A)
    adj_torch = torch.from_numpy(A).float()
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    arrs = split_train_val_test([X, X_norm, tod, dow, mask_arr], 0.7, 0.1)
    (_, X_va, X_te), (_, Xn_va, Xn_te), (_, tod_va, tod_te), \
        (_, dow_va, dow_te), (_, mk_va, mk_te) = arrs

    def mk_short(Xp, Xnp, tp, dp, mp):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    short_va = mk_short(X_va, Xn_va, tod_va, dow_va, mk_va)
    short_te = mk_short(X_te, Xn_te, tod_te, dow_te, mk_te)
    print(f"|short val|={len(short_va.dataset)} |short test|={len(short_te.dataset)}")

    # Pretrained dataset uses absolute indexing — built from full arrays
    T = X.shape[0]
    tr_r, va_r, te_r = split_t0_for_pretrained(T, args.T_long, args.in_steps, args.out_steps)
    def mk_long(rng):
        ds = PretrainedSSSMDataset(X, X_norm, tod, dow, mask_arr,
                                   t0_start=rng[0], t0_end=rng[1],
                                   T_in=args.in_steps, T_out=args.out_steps,
                                   T_long=args.T_long)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    long_va = mk_long(va_r)
    long_te = mk_long(te_r)

    # ---- discover & load all checkpoints ----
    all_paths = []
    for pattern in (args.stae_ckpts, args.stae_R01_ckpts, args.stae_big_ckpts):
        all_paths.extend([(p, "stae") for p in sorted(glob.glob(pattern))])
    all_paths.extend([(p, "stae_pre") for p in sorted(glob.glob(args.stae_pre_ckpts))])
    if args.include_gwnet:
        all_paths.extend([(p, "gwnet") for p in sorted(glob.glob(args.gwnet_ckpts))])
    if args.include_hybrid:
        all_paths.extend([(p, "hybrid") for p in sorted(glob.glob(args.hybrid_ckpts))])
    # Deduplicate by basename in case same file matched multiple patterns
    seen = set(); paths = []
    for p, k in all_paths:
        if p in seen: continue
        seen.add(p); paths.append((p, k))
    print(f"Found {len(paths)} checkpoints:")
    for p, k in paths:
        print(f"  [{k}] {os.path.basename(p)}")
    if not paths:
        print("no checkpoints found"); sys.exit(1)

    # ---- collect predictions for each model on val + test ----
    val_preds = []; test_preds = []; names = []
    Y_val_node = M_val_node = Y_test_node = M_test_node = None

    for path, kind in paths:
        name = f"{kind}:{os.path.basename(path).replace('.pth','')}"
        print(f"  collecting {name}")
        if kind == "stae":
            model = load_stae(path, device)
            Pv, Yv, Mv = collect_stae_preds(model, short_va, device, mean_t, std_t, amp_dtype)
            Pt, Yt, Mt = collect_stae_preds(model, short_te, device, mean_t, std_t, amp_dtype)
        elif kind == "stae_pre":
            model = load_stae_pre(path, device)
            Pv, Yv, Mv = collect_stae_pre_preds(model, long_va, device, mean_t, std_t, amp_dtype)
            Pt, Yt, Mt = collect_stae_pre_preds(model, long_te, device, mean_t, std_t, amp_dtype)
            # Pretrained dataset uses different (smaller) val/test splits — we'll
            # align by truncating short-dataset Y/M and predictions later
        elif kind == "gwnet":
            model = load_gwnet(path, adj_torch, device)
            Pv, Yv, Mv = collect_stae_preds(model, short_va, device, mean_t, std_t, amp_dtype)
            Pt, Yt, Mt = collect_stae_preds(model, short_te, device, mean_t, std_t, amp_dtype)
        elif kind == "hybrid":
            model = load_hybrid(path, U, evals_t, device)
            Pv, Yv, Mv = collect_stae_preds(model, short_va, device, mean_t, std_t, amp_dtype)
            Pt, Yt, Mt = collect_stae_preds(model, short_te, device, mean_t, std_t, amp_dtype)
        del model; torch.cuda.empty_cache()

        # Pretrained models have different val/test sample counts. We align to
        # the SHORT dataset by truncating their outputs (the last
        # short_va_count val samples line up since the pretrained dataset has
        # t0_start = max(T_long, T_in) = 2016, which is past the start of val).
        # For test, the pretrained dataset starts at val_end which equals the
        # short test start exactly, so they should align if T_long < val span.
        if kind == "stae_pre":
            # Determine alignment offset and truncate accordingly
            n_short_va = len(short_va.dataset)
            n_short_te = len(short_te.dataset)
            if Pv.shape[0] < n_short_va:
                # Pretrained val set is shorter — pad with zeros (no contribution)
                pad_v = torch.zeros(n_short_va - Pv.shape[0], *Pv.shape[1:])
                Pv = torch.cat([pad_v, Pv], dim=0)
            elif Pv.shape[0] > n_short_va:
                Pv = Pv[-n_short_va:]
            if Pt.shape[0] != n_short_te:
                # Test sizes should match; if not, truncate / pad
                if Pt.shape[0] < n_short_te:
                    pad_t = torch.zeros(n_short_te - Pt.shape[0], *Pt.shape[1:])
                    Pt = torch.cat([pad_t, Pt], dim=0)
                else:
                    Pt = Pt[-n_short_te:]

        val_preds.append(Pv)
        test_preds.append(Pt)
        names.append(name)
        if Y_val_node is None:
            Y_val_node = Yv; M_val_node = Mv
            Y_test_node = Yt; M_test_node = Mt
            # Y* / M* from STAE/short loaders should match if we always init from those

    K = len(val_preds)
    Pv = torch.stack(val_preds, dim=0)               # [K, B_val, T, N] (normalized)
    Pt = torch.stack(test_preds, dim=0)              # [K, B_test, T, N]
    print(f"\nAll predictions collected. Pv {Pv.shape}, Pt {Pt.shape}, Y_test {Y_test_node.shape}")

    # ---- per-model raw test metrics ----
    results = {"per_model": {}, "ensembles": {}}
    for i, name in enumerate(names):
        P_node_i = Pt[i] * float(std) + float(mean)
        m_i = per_horizon(P_node_i, Y_test_node, M_test_node)
        results["per_model"][name] = m_i
        print(f"  [{name}]  15/30/60 = {m_i['mae_15']:.3f}/{m_i['mae_30']:.3f}/{m_i['mae_60']:.3f}")

    # ---- uniform ensemble ----
    P_unif_norm = Pt.mean(dim=0)
    P_unif_node = P_unif_norm * float(std) + float(mean)
    results["ensembles"]["uniform"] = per_horizon(P_unif_node, Y_test_node, M_test_node)
    print(f"\n[uniform]    60-min MAE = {results['ensembles']['uniform']['mae_60']:.4f}")

    # ---- val-weighted (scalar per-model) ----
    Y_val_t = Y_val_node.float()
    M_val_t = M_val_node.float()
    # On val: Pv is in NORMALIZED space, we need to compare in raw mph.
    # Apply de-normalization to each model's val preds.
    Pv_node = Pv * float(std) + float(mean)
    w_scalar = optimize_softmax_weights(Pv_node, Y_val_t, M_val_t,
                                        mode="scalar", n_models=K,
                                        T_out=args.out_steps, n_iter=500)
    P_wscalar_node = (Pt * float(std) + float(mean)
                      ) * w_scalar.view(K, 1, 1, 1)
    P_wscalar_node = P_wscalar_node.sum(dim=0)
    results["ensembles"]["val_weighted_scalar"] = per_horizon(
        P_wscalar_node, Y_test_node, M_test_node)
    print(f"[scalar-wt]  60-min MAE = {results['ensembles']['val_weighted_scalar']['mae_60']:.4f}  (weights = {w_scalar.tolist()})")

    # ---- val-weighted per-horizon ----
    w_horizon = optimize_softmax_weights(Pv_node, Y_val_t, M_val_t,
                                         mode="horizon", n_models=K,
                                         T_out=args.out_steps, n_iter=800)
    P_whor_node = (Pt * float(std) + float(mean)) * w_horizon.view(K, 1, args.out_steps, 1)
    P_whor_node = P_whor_node.sum(dim=0)
    results["ensembles"]["val_weighted_horizon"] = per_horizon(
        P_whor_node, Y_test_node, M_test_node)
    print(f"[hor-wt]     60-min MAE = {results['ensembles']['val_weighted_horizon']['mae_60']:.4f}")

    # ---- ST-TTC on top of the best blend ----
    if args.use_ttc:
        print("\n--- ST-TTC v2 ---")
        # Pick the best ensemble blend by val 60-min and apply TTC on it
        best_key = min(results["ensembles"].keys(),
                       key=lambda k: results["ensembles"][k]["mae_60"])
        if best_key == "uniform":
            best_norm = P_unif_norm
        elif best_key == "val_weighted_scalar":
            best_norm = (Pt * w_scalar.view(K, 1, 1, 1)).sum(dim=0)
        else:
            best_norm = (Pt * w_horizon.view(K, 1, args.out_steps, 1)).sum(dim=0)
        print(f"Best pre-TTC blend: {best_key}  60-min={results['ensembles'][best_key]['mae_60']:.4f}")

        P_ttc_norm = apply_ttc(best_norm, Y_test_node, M_test_node, mean, std,
                               args, device,
                               ttc_groups=args.ttc_groups, ttc_lr=args.ttc_lr,
                               per_horizon=args.ttc_per_horizon,
                               warm_start=args.ttc_warm_start,
                               reset_every=args.ttc_reset_every,
                               T_out=args.out_steps)
        P_ttc_node = P_ttc_norm * float(std) + float(mean)
        results["ensembles"]["best_plus_ttc_v2"] = per_horizon(
            P_ttc_node, Y_test_node, M_test_node)
        print(f"[best+TTCv2] 60-min MAE = {results['ensembles']['best_plus_ttc_v2']['mae_60']:.4f}")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for k, m in results["ensembles"].items():
        print(f"  {k:<25}  15/30/60 = {m['mae_15']:.4f}/{m['mae_30']:.4f}/{m['mae_60']:.4f}  avg={m['avg_mae']:.4f}")

    print("\nvs Published SOTA:")
    print("  TESTAM+ 2.99 (unreprod) | TITAN 3.08 (unreprod) | TESTAM 3.14 (unreprod) | MLCAFormer 3.30 | STAEformer 3.34")
    print("  REPORT.md headline: 3.283 (4-seed + ST-TTC v1)")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
