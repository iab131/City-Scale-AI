"""
Probabilistic STAEformer with horizon-aware Gaussian NLL output.

Hypothesis (see paper §8.5 + appendix):
The METR-LA plateau at validation MAE 2.74 is NOT a bandwidth limitation
on graph spectrum (oracle K=128 achievable val 2.07) but a *capacity-
allocation* limitation: STAEformer with masked-MAE point prediction
spends equal capacity on every horizon, including those that are
intrinsically unpredictable from 12-step input. Replacing the point head
with a Gaussian (μ, log σ²) head and training with Gaussian NLL lets
the model express low confidence on hard horizons (large σ → smaller
NLL contribution), reallocating capacity to predictable horizons —
where μ should track the conditional mean more precisely, improving the
point-prediction MAE we report at inference.

Implementation differences vs. train_staeformer.py:
  - output_proj produces 2 * T_out values per sensor: (μ, log σ²)
  - loss is per-element Gaussian NLL with masking
  - inference uses μ alone; val/test MAE/RMSE/MAPE computed on μ
"""
from __future__ import annotations
import argparse, datetime, json, math, os, random, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.staeformer import STAEformer


# ---------------------------------------------------------------------------
class STAEformerGaussian(nn.Module):
    """STAEformer encoder + per-(sensor, horizon) Gaussian head."""

    def __init__(self, base: STAEformer, log_var_clamp=(-7.0, 7.0)):
        super().__init__()
        self.base = base
        # Replace the point head with a Gaussian head: output is 2 * T_out
        # per sensor, the first T_out being μ and the second log σ².
        in_features = base.in_steps * base.model_dim
        self.gauss_head = nn.Linear(in_features, 2 * base.out_steps)
        # Initialise the mean half to match the original point head so that
        # at step 1 the model produces sensible point predictions; the
        # log-variance half is initialised to zero (σ ≈ 1 normalized = std mph).
        with torch.no_grad():
            self.gauss_head.weight.zero_()
            self.gauss_head.bias.zero_()
            mu_w = base.output_proj.weight.detach().clone()  # [T_out, in_features]
            mu_b = base.output_proj.bias.detach().clone()    # [T_out]
            self.gauss_head.weight[:base.out_steps].copy_(mu_w)
            self.gauss_head.bias[:base.out_steps].copy_(mu_b)
        self.log_var_clamp = log_var_clamp

    def forward(self, x_norm, tod, dow):
        h = self.base.get_hidden(x_norm, tod, dow)              # [B, T_in, N, D]
        B, T_in, N, D = h.shape
        flat = h.transpose(1, 2).reshape(B, N, T_in * D)
        out = self.gauss_head(flat)                              # [B, N, 2 * T_out]
        mu, log_var = out.chunk(2, dim=-1)                       # each [B, N, T_out]
        log_var = log_var.clamp(self.log_var_clamp[0], self.log_var_clamp[1])
        mu = mu.transpose(1, 2).contiguous()                     # [B, T_out, N]
        log_var = log_var.transpose(1, 2).contiguous()
        return mu, log_var


# ---------------------------------------------------------------------------
def masked_gaussian_nll(mu, log_var, y_true_norm, mask, eps=1e-6):
    """Per-element Gaussian NLL on normalized scale, with masking.

    NLL = 0.5 * (log_var + (y - μ)^2 * exp(-log_var))
    (constants 0.5 * log(2π) dropped — only matters for absolute NLL.)
    Trains μ as the conditional MEAN under squared loss.
    """
    inv_var = torch.exp(-log_var)
    se = (mu - y_true_norm) ** 2
    nll = 0.5 * (log_var + se * inv_var)
    m_mean = mask.mean().clamp(min=eps)
    return (nll * mask).mean() / m_mean


def masked_laplace_nll(mu, log_b, y_true_norm, mask, eps=1e-6):
    """Per-element Laplace NLL on normalized scale, with masking.

    NLL = log(2b) + |y - μ| / b = log 2 + log_b + |y - μ| * exp(-log_b)
    (constant log 2 dropped.)
    Trains μ as the conditional MEDIAN — the MAE-optimal point predictor.
    The scale parameter exp(log_b) plays the same heteroscedastic role
    as σ² in the Gaussian case: large at high-uncertainty horizons,
    small where predictions are reliable.
    """
    inv_b = torch.exp(-log_b)
    ae = (mu - y_true_norm).abs()
    nll = log_b + ae * inv_b
    m_mean = mask.mean().clamp(min=eps)
    return (nll * mask).mean() / m_mean


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


def per_horizon(pred, true, mask):
    out = {"avg_mae": masked_mae(pred, true, mask).item(),
           "avg_rmse": masked_rmse(pred, true, mask).item(),
           "avg_mape": masked_mape(pred, true, mask).item()}
    for tag, t in [("15", 2), ("30", 5), ("60", 11)]:
        if pred.shape[1] > t:
            p, y, m = pred[:, t:t+1], true[:, t:t+1], mask[:, t:t+1]
            out[f"mae_{tag}"] = masked_mae(p, y, m).item()
            out[f"rmse_{tag}"] = masked_rmse(p, y, m).item()
            out[f"mape_{tag}"] = masked_mape(p, y, m).item()
    return out


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/staeformer_nll")
    p.add_argument("--tag", type=str, default="staeformer_nll")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in_steps", type=int, default=12)
    p.add_argument("--out_steps", type=int, default=12)
    # STAEformer hyperparams (match published defaults)
    p.add_argument("--input_embedding_dim", type=int, default=24)
    p.add_argument("--tod_embedding_dim", type=int, default=24)
    p.add_argument("--dow_embedding_dim", type=int, default=24)
    p.add_argument("--adaptive_embedding_dim", type=int, default=80)
    p.add_argument("--feed_forward_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[20, 30])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    # NLL-specific knobs
    p.add_argument("--log_var_min", type=float, default=-7.0)
    p.add_argument("--log_var_max", type=float, default=7.0)
    p.add_argument("--loss", choices=["gaussian", "laplace"], default="gaussian",
                   help="gaussian: train μ as conditional mean (squared-error). "
                        "laplace: train μ as conditional median (MAE-optimal).")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


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

    def mk(X_p, Xn_p, t_p, d_p, m_p, sh):
        ds = SSSMDataset(X_p, Xn_p, t_p, d_p, m_p,
                         input_len=args.in_steps, pred_len=args.out_steps)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                          num_workers=args.num_workers, pin_memory=True)

    tr = mk(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, True)
    va = mk(X_va, Xn_va, tod_va, dow_va, mk_va, False)
    te = mk(X_te, Xn_te, tod_te, dow_te, mk_te, False)
    print(f"[data] |tr|={len(tr.dataset)} |va|={len(va.dataset)} "
          f"|te|={len(te.dataset)}", flush=True)

    base = STAEformer(
        N=X.shape[1], in_steps=args.in_steps, out_steps=args.out_steps,
        input_embedding_dim=args.input_embedding_dim,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = STAEformerGaussian(base, log_var_clamp=(args.log_var_min,
                                                     args.log_var_max)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    nll_fn = masked_laplace_nll if args.loss == "laplace" else masked_gaussian_nll
    scale_name = "log_b" if args.loss == "laplace" else "log_var"
    print(f"[model] STAEformer+{args.loss} params={n_params/1e6:.3f}M "
          f"{scale_name}_clamp=({args.log_var_min}, {args.log_var_max})",
          flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=args.lr_milestones, gamma=args.lr_gamma)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_nll_s{args.seed}.pth")

    best_val_mae = float("inf")
    no_improve = 0
    max_gpu_mb = 0
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        run_nll = 0.0; nb = 0
        run_mae = 0.0
        for b in tr:
            x_norm = b["x_norm"].to(device, non_blocking=True)
            tod_b = b["tod"].to(device, non_blocking=True)
            dow_b = b["dow"].to(device, non_blocking=True)
            y_node = b["y_node"].to(device, non_blocking=True)
            y_mask = b["y_mask"].to(device, non_blocking=True)
            # Target in normalized space, masked by validity mask
            y_norm_true = (y_node - mean_t) / std_t.clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                mu_norm, log_var = model(x_norm, tod_b, dow_b)
                nll = nll_fn(mu_norm, log_var, y_norm_true, y_mask)
            nll.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.gradient_clip)
            opt.step()
            run_nll += float(nll.detach()); nb += 1
            # Track raw-mph MAE on μ for monitoring
            with torch.no_grad():
                y_pred = mu_norm.float() * std_t + mean_t
                run_mae += float(masked_mae(y_pred, y_node, y_mask))
            if torch.cuda.is_available():
                max_gpu_mb = max(max_gpu_mb,
                                  int(torch.cuda.max_memory_allocated() / 1024 / 1024))
        scheduler.step()
        tr_nll = run_nll / max(1, nb)
        tr_mae = run_mae / max(1, nb)

        # ---- val ----
        model.eval()
        mu_all, y_all, m_all, lv_all = [], [], [], []
        with torch.no_grad():
            for b in va:
                x_norm = b["x_norm"].to(device, non_blocking=True)
                tod_b = b["tod"].to(device, non_blocking=True)
                dow_b = b["dow"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    mu_norm, log_var = model(x_norm, tod_b, dow_b)
                y_pred = mu_norm.float() * std_t + mean_t
                mu_all.append(y_pred.cpu())
                y_all.append(b["y_node"]); m_all.append(b["y_mask"])
                lv_all.append(log_var.float().cpu())
        P = torch.cat(mu_all); Y = torch.cat(y_all); M = torch.cat(m_all)
        LV = torch.cat(lv_all)
        val_m = per_horizon(P, Y, M)
        v_mae = val_m["avg_mae"]
        lr_now = opt.param_groups[0]["lr"]
        # Diagnostic: mean per-horizon log_var
        lv_per_h = LV.mean(dim=(0, 2)).tolist()
        lv_15, lv_30, lv_60 = lv_per_h[2], lv_per_h[5], lv_per_h[11]
        print(f"[ep {ep:03d}/{args.epochs}] lr={lr_now:.2e} "
              f"tr_nll={tr_nll:.3f} tr_mae={tr_mae:.3f} "
              f"val avg/15/30/60 = {v_mae:.3f} / {val_m['mae_15']:.3f} / "
              f"{val_m['mae_30']:.3f} / {val_m['mae_60']:.3f}  "
              f"lv15/30/60={lv_15:+.2f}/{lv_30:+.2f}/{lv_60:+.2f}  "
              f"gpu={max_gpu_mb}MB", flush=True)
        if v_mae < best_val_mae - 1e-4:
            best_val_mae = v_mae
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_metrics": val_m, "epoch": ep}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs",
                      flush=True)
                break

    # ---- test ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mu_all, y_all, m_all = [], [], []
    with torch.no_grad():
        for b in te:
            x_norm = b["x_norm"].to(device, non_blocking=True)
            tod_b = b["tod"].to(device, non_blocking=True)
            dow_b = b["dow"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                mu_norm, _ = model(x_norm, tod_b, dow_b)
            y_pred = mu_norm.float() * std_t + mean_t
            mu_all.append(y_pred.cpu())
            y_all.append(b["y_node"]); m_all.append(b["y_mask"])
    P = torch.cat(mu_all); Y = torch.cat(y_all); M = torch.cat(m_all)
    te_m = per_horizon(P, Y, M)
    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(te_m, indent=2)}", flush=True)
    print(f"[done] elapsed={elapsed:.1f}s best_val_mae={best_val_mae:.3f} "
          f"max_gpu={max_gpu_mb}MB", flush=True)

    row = {"model": "STAEformer_Gaussian_NLL",
           "seed": args.seed, "epochs": ckpt["epoch"],
           "params_M": round(n_params / 1e6, 3),
           "elapsed_sec": round(elapsed, 1),
           **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
           **{f"test_{k}": v for k, v in te_m.items()},
           "tag": args.tag,
           "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    csv_path = os.path.join(args.out_dir, "staeformer_nll_results.csv")
    pd.DataFrame([row]).to_csv(csv_path, mode="a",
                                header=not os.path.exists(csv_path), index=False)
    print(f"[done] appended to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
