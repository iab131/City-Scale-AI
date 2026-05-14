"""
R16 — Per-horizon top-K ensemble.

For each horizon h ∈ {0..11}, find the K models that minimize VAL MAE at that
specific horizon, then uniform-average their predictions for that horizon.
Different K_h per horizon is possible.

To avoid per-horizon overfitting, also report a simpler variant: same K for
all horizons but per-horizon model selection.

Adds the non-STAE diverse models (GWNet, Hybrid, stae_pre) as well.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--T_long", type=int, default=2016)
    p.add_argument("--use_ttc", action="store_true")
    p.add_argument("--out_json", type=str, default="results/R16_per_horizon_topk.json")
    return p.parse_args()


def masked_mae(p, y, m, eps=1e-6):
    mm = m.mean().clamp(min=eps); return ((p - y).abs() * m).mean() / mm


def masked_rmse(p, y, m, eps=1e-6):
    mm = m.mean().clamp(min=eps); return torch.sqrt(((p - y) ** 2 * m).mean() / mm)


def masked_mape(p, y, m, eps=1e-6):
    m2 = m * (y.abs() > 1e-3).float(); mm = m2.mean().clamp(min=eps)
    return ((p - y).abs() / y.abs().clamp(min=eps) * m2).mean() / mm


def per_horizon(pred, true, mask):
    out = {"avg_mae": masked_mae(pred, true, mask).item(),
           "avg_rmse": masked_rmse(pred, true, mask).item(),
           "avg_mape": masked_mape(pred, true, mask).item()}
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p_t, y_t, m_t = pred[:, t:t+1], true[:, t:t+1], mask[:, t:t+1]
            out[f"mae_{tag}"] = masked_mae(p_t, y_t, m_t).item()
            out[f"rmse_{tag}"] = masked_rmse(p_t, y_t, m_t).item()
            out[f"mape_{tag}"] = masked_mape(p_t, y_t, m_t).item()
    return out


def per_horizon_val_mae(P_models, Y, M):
    """P_models: [K, B, T, N] normalized predictions.
    Returns: [K, T] per-(model, horizon) MAE in raw mph.
    """
    K, B, T, N = P_models.shape
    out = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            p = P_models[k, :, t:t+1]
            y = Y[:, t:t+1]
            m = M[:, t:t+1]
            out[k, t] = masked_mae(p, y, m).item()
    return out


class SDCalibrator(torch.nn.Module):
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups; self.group_size = freq_bins // groups
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
            A_g = A[:, :, s:e] * (1 + self.lambda_amp[g].unsqueeze(0))
            P_g = P[:, :, s:e] + self.lambda_phi[g].unsqueeze(0)
            Yf_corr[:, :, s:e] = A_g * torch.exp(1j * P_g)
        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.permute(0, 2, 1)


def load_stae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False); a = ckpt["args"]
    if a.get("in_steps", 12) != 12:
        return None, None
    m = STAEformer(N=207, in_steps=12, out_steps=12,
                   input_embedding_dim=a["input_embedding_dim"],
                   tod_embedding_dim=a["tod_embedding_dim"],
                   dow_embedding_dim=a["dow_embedding_dim"],
                   adaptive_embedding_dim=a["adaptive_embedding_dim"],
                   feed_forward_dim=a["feed_forward_dim"],
                   num_heads=a["num_heads"], num_layers=a["num_layers"],
                   dropout=a["dropout"]).to(device).eval()
    m.load_state_dict(ckpt["model"]); return m, a


def load_gwnet(ckpt_path, adj_torch, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False); a = ckpt["args"]
    m = GraphWaveNet(N=207, adj_mx=adj_torch,
                     in_steps=12, out_steps=12,
                     in_dim=3, out_dim=1,
                     residual_channels=a["residual_channels"],
                     dilation_channels=a["dilation_channels"],
                     skip_channels=a["skip_channels"],
                     end_channels=a["end_channels"],
                     kernel_size=a["kernel_size"], blocks=a["blocks"],
                     layers=a["layers"], dropout=a["dropout"],
                     adaptive_adj=not a.get("no_adaptive_adj", False)).to(device).eval()
    m.load_state_dict(ckpt["model"]); return m, a


def collect_stae(model, loader, device, amp_dtype):
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


def collect_stae_pre(model, loader, device, amp_dtype):
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


def main():
    args = parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow, mask_arr = data["tod"], data["dow"], data["missing_mask"]
    mean, std = data["mean"], data["std"]
    _, _, A = load_adj_pkl(args.adj_path); A = symmetrize_adjacency(A)
    adj_torch = torch.from_numpy(A).float()

    arrs = split_train_val_test([X, X_norm, tod, dow, mask_arr], 0.7, 0.1)
    (_, X_va, X_te), (_, Xn_va, Xn_te), (_, tod_va, tod_te), \
        (_, dow_va, dow_te), (_, mk_va, mk_te) = arrs

    def mk(Xp, Xnp, tp, dp, mp):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=12, pred_len=12)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    short_va = mk(X_va, Xn_va, tod_va, dow_va, mk_va)
    short_te = mk(X_te, Xn_te, tod_te, dow_te, mk_te)

    # Long-history for stae_pre
    T = X.shape[0]
    tr_r, va_r, te_r = split_t0_for_pretrained(T, args.T_long, 12, 12)
    def mk_long(rng):
        ds = PretrainedSSSMDataset(X, X_norm, tod, dow, mask_arr,
                                   t0_start=rng[0], t0_end=rng[1],
                                   T_in=12, T_out=12, T_long=args.T_long)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    long_va = mk_long(va_r); long_te = mk_long(te_r)

    # Collect models
    stae_paths = sorted(glob.glob("results/staeformer/stae_R*/best_stae_s*.pth"))
    stae_paths += sorted(glob.glob("results/staeformer/stae_repro*/best_stae_s*.pth"))
    # Exclude R14 prior (different checkpoint naming + different forward signature)
    stae_paths = [p for p in stae_paths if "R14_prior" not in p]
    stae_paths = sorted(set(stae_paths))
    gwnet_paths = sorted(glob.glob("results/gwnet/gwnet_s*/best_gwnet_s*.pth"))
    hybrid_paths = sorted(glob.glob("results/hybrid/hybrid_s*/best_hybrid_s*.pth"))
    pre_paths = sorted(glob.glob("results/stae_pretrained/*/best_stae_pre_s*.pth"))
    prior_paths = sorted(glob.glob("results/staeformer/stae_R14_prior_s*/best_stae_prior_s*.pth"))

    val_preds = []; test_preds = []; names = []
    Y_val = M_val = Y_test = M_test = None

    for path in stae_paths:
        m, a = load_stae(path, device)
        if m is None: continue
        Pv, Yv, Mv = collect_stae(m, short_va, device, amp_dtype)
        Pt, Yt, Mt = collect_stae(m, short_te, device, amp_dtype)
        val_preds.append(Pv); test_preds.append(Pt)
        names.append(f"stae:{os.path.basename(path)}")
        if Y_val is None:
            Y_val, M_val = Yv, Mv; Y_test, M_test = Yt, Mt
        del m; torch.cuda.empty_cache()

    for path in gwnet_paths:
        m, a = load_gwnet(path, adj_torch, device)
        Pv, _, _ = collect_stae(m, short_va, device, amp_dtype)
        Pt, _, _ = collect_stae(m, short_te, device, amp_dtype)
        val_preds.append(Pv); test_preds.append(Pt)
        names.append(f"gwnet:{os.path.basename(path)}")
        del m; torch.cuda.empty_cache()

    for path in hybrid_paths:
        try:
            from models.hybrid import HybridSTAEMamba
            ckpt = torch.load(path, map_location=device, weights_only=False); a = ckpt["args"]
            U_np = data["U"]; evals_np = data["evals"]
            U = torch.from_numpy(U_np).float().to(device); evals_t = torch.from_numpy(evals_np).float().to(device)
            m = HybridSTAEMamba(N=207, U=U, evals=evals_t,
                                in_steps=12, out_steps=12,
                                adaptive_embedding_dim=a["adaptive_embedding_dim"],
                                feed_forward_dim=a["feed_forward_dim"],
                                num_heads=a["num_heads"], num_layers=a["num_layers"],
                                dropout=a["dropout"], spec_d=a["spec_d"],
                                spec_layers=a["spec_layers"]).to(device).eval()
            m.load_state_dict(ckpt["model"])
            Pv, _, _ = collect_stae(m, short_va, device, amp_dtype)
            Pt, _, _ = collect_stae(m, short_te, device, amp_dtype)
            val_preds.append(Pv); test_preds.append(Pt)
            names.append(f"hybrid:{os.path.basename(path)}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  skip hybrid {path}: {e}")

    for path in prior_paths:
        try:
            from models.staeformer_prior import STAEformerWithPrior
            ckpt = torch.load(path, map_location=device, weights_only=False); a = ckpt["args"]
            m = STAEformerWithPrior(N=207, dropout=a.get("dropout", 0.15)).to(device).eval()
            m.load_state_dict(ckpt["model"])
            # Need prior_norm in dataset
            def collect_prior(loader):
                preds = []; ys = []; ms = []
                with torch.no_grad():
                    for batch in loader:
                        x_norm = batch["x_norm"].to(device, non_blocking=True)
                        tod_b = batch["tod"].to(device, non_blocking=True)
                        dow_b = batch["dow"].to(device, non_blocking=True)
                        prior_n = batch["prior_norm_in"].to(device, non_blocking=True)
                        with torch.amp.autocast('cuda', dtype=amp_dtype):
                            yn = m(x_norm, tod_b, dow_b, prior_n)
                        preds.append(yn.float().cpu())
                        ys.append(batch["y_node"]); ms.append(batch["y_mask"])
                return torch.cat(preds), torch.cat(ys), torch.cat(ms)
            # Build prior-aware loaders inline
            from src.dataset_v2 import SSSMDataset as DS
            ds_va = DS(X_va, Xn_va, tod_va, dow_va, mk_va, input_len=12, pred_len=12,
                       prior_norm=data["prior_norm"][int(0.7*T):int(0.8*T)])
            ds_te = DS(X_te, Xn_te, tod_te, dow_te, mk_te, input_len=12, pred_len=12,
                       prior_norm=data["prior_norm"][int(0.8*T):])
            from torch.utils.data import DataLoader as DL
            short_va_p = DL(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            short_te_p = DL(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            Pv, _, _ = collect_prior(short_va_p)
            Pt, _, _ = collect_prior(short_te_p)
            val_preds.append(Pv); test_preds.append(Pt)
            names.append(f"stae_prior:{os.path.basename(path)}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  skip stae_prior {path}: {e}")

    for path in pre_paths:
        try:
            from models.staeformer_pretrained import STAEformerPretrained
            ckpt = torch.load(path, map_location=device, weights_only=False); a = ckpt["args"]
            m = STAEformerPretrained(
                N=207, tmae_ckpt_path=a["tmae_ckpt"], smae_ckpt_path=a.get("smae_ckpt"),
                in_steps=12, out_steps=12,
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
            Pv, _, _ = collect_stae_pre(m, long_va, device, amp_dtype)
            Pt, _, _ = collect_stae_pre(m, long_te, device, amp_dtype)
            # Align to short val/test
            n_short_va = len(short_va.dataset); n_short_te = len(short_te.dataset)
            if Pv.shape[0] < n_short_va:
                pad = torch.zeros(n_short_va - Pv.shape[0], *Pv.shape[1:]); Pv = torch.cat([pad, Pv])
            elif Pv.shape[0] > n_short_va:
                Pv = Pv[-n_short_va:]
            if Pt.shape[0] != n_short_te:
                if Pt.shape[0] < n_short_te:
                    pad = torch.zeros(n_short_te - Pt.shape[0], *Pt.shape[1:]); Pt = torch.cat([pad, Pt])
                else:
                    Pt = Pt[-n_short_te:]
            val_preds.append(Pv); test_preds.append(Pt)
            names.append(f"stae_pre:{os.path.basename(path)}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  skip stae_pre {path}: {e}")

    K_total = len(val_preds)
    Pv = torch.stack(val_preds, dim=0)
    Pt = torch.stack(test_preds, dim=0)
    print(f"Loaded {K_total} models")

    # Per-horizon val MAE
    Pv_node = Pv * float(std) + float(mean)
    Pt_node = Pt * float(std) + float(mean)
    val_per_horizon = per_horizon_val_mae(Pv_node, Y_val, M_val)   # [K, T]

    # Per-horizon top-K (different K per horizon)
    T = 12
    best_preds_node = torch.zeros_like(Pt_node[0])
    K_per_horizon = []
    # For each horizon, try K ∈ {1..K_total}, pick the one with lowest val MAE
    val_per_horizon_blend = np.zeros((K_total, T))
    for t in range(T):
        order = np.argsort(val_per_horizon[:, t])
        best_K_t = K_total; best_val_t = float("inf")
        for k in range(1, K_total + 1):
            sel = order[:k]
            P_blend_v = Pv_node[sel, :, t:t+1].mean(dim=0)  # [B, 1, N]
            v_mae = masked_mae(P_blend_v, Y_val[:, t:t+1], M_val[:, t:t+1]).item()
            val_per_horizon_blend[k-1, t] = v_mae
            if v_mae < best_val_t - 1e-5:
                best_val_t = v_mae; best_K_t = k
        K_per_horizon.append(best_K_t)
        # Apply best K to test
        sel = order[:best_K_t]
        best_preds_node[:, t, :] = Pt_node[sel, :, t, :].mean(dim=0)

    print(f"Per-horizon best Ks (chosen on val): {K_per_horizon}")
    metrics_perhor = per_horizon(best_preds_node, Y_test, M_test)
    print(f"Per-horizon top-K (val-chosen) test 60-min: {metrics_perhor['mae_60']:.4f}")

    # Uniform (all models) baseline
    P_unif_te = Pt_node.mean(dim=0)
    metrics_unif = per_horizon(P_unif_te, Y_test, M_test)
    print(f"Uniform all-{K_total} test 60-min: {metrics_unif['mae_60']:.4f}")

    results = {
        "n_models": K_total,
        "model_names": names,
        "K_per_horizon_val": K_per_horizon,
        "uniform_all": metrics_unif,
        "per_horizon_topk": metrics_perhor,
    }

    if args.use_ttc:
        sdc = SDCalibrator(num_nodes=207, freq_bins=T // 2 + 1, groups=4).to(device)
        ttc_opt = torch.optim.Adam(sdc.parameters(), lr=1e-4)
        Q = deque(maxlen=T)
        # Apply to per-horizon best
        best_norm = best_preds_node / float(std) - float(mean) / float(std)
        cal_preds = []
        for i in range(best_norm.shape[0]):
            yp = best_norm[i:i+1].to(device); yn = Y_test[i:i+1].to(device); ym = M_test[i:i+1].to(device)
            sdc.eval()
            with torch.no_grad(): yc = sdc(yp)
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
        P_cal_node = P_cal_norm * float(std) + float(mean)
        m_ttc = per_horizon(P_cal_node, Y_test, M_test)
        results["per_horizon_topk_plus_ttc"] = m_ttc
        print(f"Per-horizon top-K + ST-TTC v2: {m_ttc['mae_60']:.4f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
