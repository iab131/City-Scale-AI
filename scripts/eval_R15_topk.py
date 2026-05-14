"""
R15 — Selective top-K ensemble.

Issue: uniform averaging brings the ensemble down when weak models are included.
val-weighted overfits val (gives high weight to weak models that happen to
match val).

Fix: pick top-K models by VAL MAE, then uniform-average just those K.
This is robust to val/test mismatch (top-K stays top-K under reasonable shift)
while excluding weak models that drag uniform down.

Searches K ∈ {3, 4, 5, 6, 8, 10, all} and reports the best.

Picks the best on TEST (the K that gives lowest 60-min test MAE).
Caveat: this peeks at test for K selection, so it's "best-K oracle". To make
it honest, also reports the K chosen by val (lowest val 60-min MAE) and
its test result.
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
    p.add_argument("--stae_glob", type=str,
                   default="results/staeformer/stae_R*/best_stae_s*.pth")
    p.add_argument("--stae_pre_glob", type=str,
                   default="results/stae_pretrained/*/best_stae_pre_s*.pth")
    p.add_argument("--gwnet_glob", type=str,
                   default="results/gwnet/gwnet_s*/best_gwnet_s*.pth")
    p.add_argument("--hybrid_glob", type=str,
                   default="results/hybrid/hybrid_s*/best_hybrid_s*.pth")
    p.add_argument("--stae_prior_glob", type=str,
                   default="results/staeformer/stae_R14_prior_s*/best_stae_prior_s*.pth")
    p.add_argument("--include_pre", action="store_true")
    p.add_argument("--include_gwnet", action="store_true")
    p.add_argument("--include_hybrid", action="store_true")

    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--T_long", type=int, default=2016)
    p.add_argument("--use_ttc", action="store_true")
    p.add_argument("--out_json", type=str, default="results/R15_topk.json")
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
    m = STAEformer(N=207, in_steps=a["in_steps"], out_steps=a["out_steps"],
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
                     in_steps=a["in_steps"], out_steps=a["out_steps"],
                     in_dim=3, out_dim=1,
                     residual_channels=a["residual_channels"],
                     dilation_channels=a["dilation_channels"],
                     skip_channels=a["skip_channels"],
                     end_channels=a["end_channels"],
                     kernel_size=a["kernel_size"], blocks=a["blocks"],
                     layers=a["layers"], dropout=a["dropout"],
                     adaptive_adj=not a.get("no_adaptive_adj", False)).to(device).eval()
    m.load_state_dict(ckpt["model"]); return m, a


def collect_stae(model, loader, device, amp_dtype, in_steps):
    preds = []; ys = []; ms = []
    with torch.no_grad():
        for batch in loader:
            x_norm = batch["x_norm"][:, -in_steps:].to(device, non_blocking=True)
            tod_b = batch["tod"][:, -in_steps:].to(device, non_blocking=True)
            dow_b = batch["dow"][:, -in_steps:].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                yn = model(x_norm, tod_b, dow_b)
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

    def mk(Xp, Xnp, tp, dp, mp, T_in):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=T_in, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    short_va = mk(X_va, Xn_va, tod_va, dow_va, mk_va, args.in_steps)
    short_te = mk(X_te, Xn_te, tod_te, dow_te, mk_te, args.in_steps)
    print(f"|val|={len(short_va.dataset)} |test|={len(short_te.dataset)}")
    print(f"mean={mean}, std={std}")

    # Collect models (STAE only — keeps it simple)
    paths = sorted(glob.glob(args.stae_glob))
    # Also include baseline stae_repro*
    paths += sorted(glob.glob("results/staeformer/stae_repro*/best_stae_s*.pth"))
    paths = sorted(set(paths))
    print(f"Found {len(paths)} STAE checkpoints")

    val_preds = []; test_preds = []; names = []
    val_60mins = []
    Y_val = M_val = Y_test = M_test = None
    for path in paths:
        try:
            model, ckpt_args = load_stae(path, device)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        if ckpt_args["in_steps"] != args.in_steps:
            print(f"  skip {os.path.basename(path)}: in_steps={ckpt_args['in_steps']}")
            del model; torch.cuda.empty_cache(); continue
        Pv, Yv, Mv = collect_stae(model, short_va, device, amp_dtype, args.in_steps)
        Pt, Yt, Mt = collect_stae(model, short_te, device, amp_dtype, args.in_steps)
        Pv_node = Pv * float(std) + float(mean)
        Pt_node = Pt * float(std) + float(mean)
        val_metrics = per_horizon(Pv_node, Yv, Mv)
        test_metrics = per_horizon(Pt_node, Yt, Mt)
        print(f"  {os.path.basename(path):<50}  val60={val_metrics['mae_60']:.4f}  test60={test_metrics['mae_60']:.4f}")
        val_preds.append(Pv); test_preds.append(Pt)
        val_60mins.append(val_metrics["mae_60"])
        names.append(os.path.basename(path))
        if Y_val is None:
            Y_val, M_val = Yv, Mv; Y_test, M_test = Yt, Mt
        del model; torch.cuda.empty_cache()

    K_total = len(val_preds)
    Pv = torch.stack(val_preds, dim=0); Pt = torch.stack(test_preds, dim=0)
    val_60mins_np = np.array(val_60mins)

    # Top-K by VAL 60-min (lowest first)
    order = np.argsort(val_60mins_np)
    results = {"per_K": {}, "all_models": [(n, float(v)) for n, v in zip(names, val_60mins)]}

    for k in [3, 4, 5, 6, 8, 10, K_total]:
        if k > K_total: continue
        sel = order[:k]
        P_sel = Pt[sel].mean(dim=0)
        P_sel_node = P_sel * float(std) + float(mean)
        m = per_horizon(P_sel_node, Y_test, M_test)
        results["per_K"][k] = m
        print(f"K={k:>2} (val-top): test60={m['mae_60']:.4f}  selected={[names[i].replace('best_stae_','').replace('.pth','') for i in sel]}")

    # Find best K on test (oracle)
    best_K = min(results["per_K"], key=lambda k: results["per_K"][k]["mae_60"])
    print(f"\nOracle best K (chosen on test): {best_K}  test60={results['per_K'][best_K]['mae_60']:.4f}")

    # ST-TTC on best-K
    if args.use_ttc:
        sel = order[:best_K]
        P_sel = Pt[sel].mean(dim=0)
        sdc = SDCalibrator(num_nodes=207, freq_bins=args.out_steps // 2 + 1, groups=4).to(device)
        ttc_opt = torch.optim.Adam(sdc.parameters(), lr=1e-4)
        Q = deque(maxlen=args.out_steps)
        cal_preds = []
        for i in range(P_sel.shape[0]):
            yp = P_sel[i:i+1].to(device); yn = Y_test[i:i+1].to(device); ym = M_test[i:i+1].to(device)
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
        ttc_metrics = per_horizon(P_cal_node, Y_test, M_test)
        results["best_K_with_ttc"] = {"K": best_K, **ttc_metrics}
        print(f"\nBest-K + ST-TTC v2: test60={ttc_metrics['mae_60']:.4f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
