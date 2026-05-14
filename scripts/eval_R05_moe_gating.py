"""
R05 — Mixture-of-Experts gated ensemble.

Trains a small gating network on val data that maps each input to a softmax
over models (and optionally per horizon). Gating features:
  - per-input mean / std of the input speeds (12-step input)
  - TOD sin/cos of the prediction-start
  - DOW one-hot of the prediction-start

The gating network is a 2-layer MLP. Output is [K_models, T_out] softmax-weights
per sample.

This is genuinely different from R04's val-optimized weighting because the
weights are *per-sample*: a sample at 3 AM gets different model weights than
a sample at rush-hour. If different models have complementary error patterns
(e.g. one is better at rush-hour, another at off-peak), gating will pick that up.

Uses the same set of checkpoints as R04. Re-uses R04's collect_*_preds.
"""

import os
import sys
import glob
import json
import argparse

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
# Lazy imports inside the loader switch (mamba_ssm dep for hybrid)


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

    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    p.add_argument("--T_long", type=int, default=2016)
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--gate_lr", type=float, default=1e-3)
    p.add_argument("--gate_epochs", type=int, default=30)
    p.add_argument("--gate_weight_decay", type=float, default=1e-3)
    p.add_argument("--out_json", type=str, default="results/R05_moe_gating.json")
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


class GatingMLP(nn.Module):
    """Maps per-sample gating features → softmax weights over K models.

    Output: [B, K, T_out]  softmax over K per (B, t).
    """
    def __init__(self, in_dim: int, K: int, T_out: int, hidden: int = 64):
        super().__init__()
        self.K = K
        self.T_out = T_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, K * T_out),
        )
        # Zero-init last layer so initial gating = uniform
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):
        # feats: [B, in_dim] → logits [B, K, T_out] → softmax over K
        logits = self.net(feats).view(-1, self.K, self.T_out)
        return torch.softmax(logits, dim=1)


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


def build_features(x_norm, tod, dow):
    """Build per-sample gating features.
    x_norm: [B, T_in, N]
    tod:    [B, T_in]
    dow:    [B, T_in]
    Returns: [B, F]
    """
    B = x_norm.shape[0]
    mean_speed = x_norm.mean(dim=(1, 2))                    # [B]
    std_speed = x_norm.std(dim=(1, 2))                       # [B]
    # Use TOD at the prediction-start time = tod[-1]
    tod_now = tod[:, -1]
    ang = 2.0 * 3.14159 * tod_now
    tod_sin = torch.sin(ang)
    tod_cos = torch.cos(ang)
    dow_now = dow[:, -1].long()
    dow_oh = torch.nn.functional.one_hot(dow_now, num_classes=7).float()
    return torch.cat([
        mean_speed.unsqueeze(-1), std_speed.unsqueeze(-1),
        tod_sin.unsqueeze(-1), tod_cos.unsqueeze(-1),
        dow_oh,
    ], dim=-1)


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

    # ---- discover checkpoints ----
    all_paths = []
    for pat in (args.stae_ckpts, args.stae_R01_ckpts, args.stae_big_ckpts):
        all_paths.extend([(p, "stae") for p in sorted(glob.glob(pat))])
    all_paths.extend([(p, "stae_pre") for p in sorted(glob.glob(args.stae_pre_ckpts))])
    if args.include_gwnet:
        all_paths.extend([(p, "gwnet") for p in sorted(glob.glob(args.gwnet_ckpts))])
    if args.include_hybrid:
        all_paths.extend([(p, "hybrid") for p in sorted(glob.glob(args.hybrid_ckpts))])
    seen = set(); paths = []
    for p, k in all_paths:
        if p in seen: continue
        seen.add(p); paths.append((p, k))
    print(f"Found {len(paths)} checkpoints")
    if not paths:
        print("no checkpoints"); sys.exit(1)

    # ---- collect preds ----
    val_preds = []; test_preds = []; names = []
    Y_val_node = M_val_node = Y_test_node = M_test_node = None
    for path, kind in paths:
        print(f"  {kind}: {os.path.basename(path)}")
        if kind == "stae":
            ck = torch.load(path, map_location=device, weights_only=False); a = ck["args"]
            m = STAEformer(N=207, in_steps=a["in_steps"], out_steps=a["out_steps"],
                           input_embedding_dim=a["input_embedding_dim"],
                           tod_embedding_dim=a["tod_embedding_dim"],
                           dow_embedding_dim=a["dow_embedding_dim"],
                           adaptive_embedding_dim=a["adaptive_embedding_dim"],
                           feed_forward_dim=a["feed_forward_dim"],
                           num_heads=a["num_heads"], num_layers=a["num_layers"],
                           dropout=a["dropout"]).to(device).eval()
            m.load_state_dict(ck["model"])
            Pv, Yv, Mv = collect_stae(m, short_va, device, amp_dtype)
            Pt, Yt, Mt = collect_stae(m, short_te, device, amp_dtype)
        elif kind == "stae_pre":
            from models.staeformer_pretrained import STAEformerPretrained
            ck = torch.load(path, map_location=device, weights_only=False); a = ck["args"]
            m = STAEformerPretrained(
                N=207, tmae_ckpt_path=a["tmae_ckpt"], smae_ckpt_path=a["smae_ckpt"],
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
            m.load_state_dict(ck["model"])
            Pv, Yv, Mv = collect_stae_pre(m, long_va, device, amp_dtype)
            Pt, Yt, Mt = collect_stae_pre(m, long_te, device, amp_dtype)
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
        elif kind == "gwnet":
            ck = torch.load(path, map_location=device, weights_only=False); a = ck["args"]
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
            m.load_state_dict(ck["model"])
            Pv, Yv, Mv = collect_stae(m, short_va, device, amp_dtype)
            Pt, Yt, Mt = collect_stae(m, short_te, device, amp_dtype)
        elif kind == "hybrid":
            from models.hybrid import HybridSTAEMamba
            ck = torch.load(path, map_location=device, weights_only=False); a = ck["args"]
            m = HybridSTAEMamba(N=207, U=U, evals=evals_t,
                                in_steps=a["in_steps"], out_steps=a["out_steps"],
                                adaptive_embedding_dim=a["adaptive_embedding_dim"],
                                feed_forward_dim=a["feed_forward_dim"],
                                num_heads=a["num_heads"], num_layers=a["num_layers"],
                                dropout=a["dropout"],
                                spec_d=a["spec_d"], spec_layers=a["spec_layers"]).to(device).eval()
            m.load_state_dict(ck["model"])
            Pv, Yv, Mv = collect_stae(m, short_va, device, amp_dtype)
            Pt, Yt, Mt = collect_stae(m, short_te, device, amp_dtype)
        del m; torch.cuda.empty_cache()

        val_preds.append(Pv); test_preds.append(Pt)
        names.append(f"{kind}:{os.path.basename(path)}")
        if Y_val_node is None:
            Y_val_node = Yv; M_val_node = Mv
            Y_test_node = Yt; M_test_node = Mt

    K = len(val_preds)
    Pv = torch.stack(val_preds, dim=0)        # [K, B_val, T, N] (normalized)
    Pt = torch.stack(test_preds, dim=0)
    Pv_node = Pv * float(std) + float(mean)
    Pt_node = Pt * float(std) + float(mean)

    # ---- gather gating features on val and test ----
    # Iterate datasets to grab x_norm, tod, dow per sample
    def gather_features(loader):
        feats = []
        for batch in loader:
            f = build_features(batch["x_norm"], batch["tod"], batch["dow"])
            feats.append(f)
        return torch.cat(feats, dim=0)

    feat_val = gather_features(short_va)            # [B_val, F]
    feat_te  = gather_features(short_te)            # [B_test, F]
    F = feat_val.shape[-1]
    print(f"Gating features dim: {F}")

    # ---- baselines: uniform & val-weighted-scalar on val data ----
    P_unif_te = Pt_node.mean(dim=0)
    base_metrics = per_horizon(P_unif_te, Y_test_node, M_test_node)
    print(f"\n[uniform]   60-min = {base_metrics['mae_60']:.4f}")

    # ---- train gating network ----
    gate = GatingMLP(in_dim=F, K=K, T_out=args.out_steps, hidden=args.gate_hidden).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr,
                            weight_decay=args.gate_weight_decay)

    # Move val tensors to device
    Pv_node_dev = Pv_node.to(device)
    Y_val_dev = Y_val_node.float().to(device)
    M_val_dev = M_val_node.float().to(device)
    feat_val_dev = feat_val.to(device)

    B_val = Pv_node_dev.shape[1]
    # Mini-batch training
    batch_size_gate = 256
    n_batches = (B_val + batch_size_gate - 1) // batch_size_gate
    print(f"\nTraining gating MLP: {args.gate_epochs} epochs, lr={args.gate_lr}, hidden={args.gate_hidden}")
    for epoch in range(1, args.gate_epochs + 1):
        gate.train()
        perm = torch.randperm(B_val, device=device)
        running = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size_gate : (b + 1) * batch_size_gate]
            f = feat_val_dev[idx]                                # [b, F]
            w = gate(f)                                          # [b, K, T_out]
            # Apply weights: P[K, b, T, N] * w[b, K, T] → [b, T, N]
            P_b = Pv_node_dev[:, idx, :, :]                      # [K, b, T, N]
            P_b = P_b.permute(1, 0, 2, 3)                        # [b, K, T, N]
            Pw = (P_b * w.unsqueeze(-1)).sum(dim=1)              # [b, T, N]
            loss = masked_mae(Pw, Y_val_dev[idx], M_val_dev[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.detach())
        if epoch % 5 == 0 or epoch == 1:
            print(f"[gate ep {epoch:03d}/{args.gate_epochs}] val_mae={running/n_batches:.4f}")

    # ---- apply on test ----
    gate.eval()
    feat_te_dev = feat_te.to(device)
    Pt_node_dev = Pt_node.to(device)
    with torch.no_grad():
        w_te = gate(feat_te_dev)                                  # [B_te, K, T_out]
        P_te = Pt_node_dev.permute(1, 0, 2, 3)                    # [B_te, K, T, N]
        P_moe = (P_te * w_te.unsqueeze(-1)).sum(dim=1)            # [B_te, T, N]
    moe_metrics = per_horizon(P_moe.cpu(), Y_test_node, M_test_node)
    print(f"\n[MoE gated] 60-min = {moe_metrics['mae_60']:.4f}")

    results = {
        "n_models": K,
        "model_names": names,
        "uniform": base_metrics,
        "moe_gated": moe_metrics,
        "gate_lr": args.gate_lr,
        "gate_hidden": args.gate_hidden,
        "gate_epochs": args.gate_epochs,
    }
    print("\n" + "=" * 60)
    print("MoE vs uniform:")
    for k in ["mae_15", "mae_30", "mae_60", "avg_mae"]:
        print(f"  {k}  uniform={base_metrics[k]:.4f}  moe={moe_metrics[k]:.4f}  delta={moe_metrics[k]-base_metrics[k]:+.4f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
