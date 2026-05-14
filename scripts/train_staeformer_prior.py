"""Train STAEformer-with-prior variant. Uses calendar prior as per-(t,n) input."""
import os, sys, time, json, argparse, datetime, random
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([ROOT, os.path.join(ROOT, "src")])

from preprocess_v2 import get_cached_v2_data
from dataset_v2 import SSSMDataset, split_train_val_test
from models.staeformer_prior import STAEformerWithPrior


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft")
    p.add_argument("--out_dir", type=str, default="results/staeformer")
    p.add_argument("--tag", type=str, default="stae_prior")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[20, 30])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=4)
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


def main():
    args = parse_args()
    os.chdir(ROOT)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} seed={args.seed}")

    data = get_cached_v2_data(args.data_path, args.adj_path, k=207, cache_dir=args.cache_dir)
    X, X_norm = data["X"], data["X_norm"]
    tod, dow = data["tod"], data["dow"]
    mask = data["missing_mask"]
    prior_norm = data["prior_norm"]
    mean, std = data["mean"], data["std"]

    arrs = split_train_val_test([X, X_norm, tod, dow, mask, prior_norm], 0.7, 0.1)
    (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), (tod_tr, tod_va, tod_te), \
        (dow_tr, dow_va, dow_te), (mk_tr, mk_va, mk_te), \
        (prn_tr, prn_va, prn_te) = arrs

    def mk(Xp, Xnp, tp, dp, mp, prn_p, shuffle):
        ds = SSSMDataset(Xp, Xnp, tp, dp, mp, input_len=12, pred_len=12,
                         prior_norm=prn_p)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    tr_loader = mk(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, prn_tr, True)
    va_loader = mk(X_va, Xn_va, tod_va, dow_va, mk_va, prn_va, False)
    te_loader = mk(X_te, Xn_te, tod_te, dow_te, mk_te, prn_te, False)

    model = STAEformerWithPrior(N=X.shape[1], dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M model_dim={model.model_dim}")

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones,
                                                     gamma=args.lr_gamma)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mean_t = torch.tensor(mean, device=device); std_t = torch.tensor(std, device=device)

    out_dir = os.path.join(args.out_dir, args.tag); os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_stae_prior_s{args.seed}.pth")

    best_val = float("inf"); epochs_no_improve = 0; t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; nb = 0
        for batch in tr_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            prior_n = batch["prior_norm_in"].to(device, non_blocking=True)
            y_node = batch["y_node"].to(device, non_blocking=True)
            y_mask = batch["y_mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                y_pred_norm = model(x_norm, tod_b, dow_b, prior_n)
                y_pred = y_pred_norm * std_t + mean_t
                loss = masked_mae(y_pred, y_node, y_mask)
            loss.backward(); opt.step()
            running += float(loss.detach()); nb += 1
        train_mae = running / max(1, nb)

        model.eval(); all_p, all_y, all_m = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                x_norm = batch["x_norm"].to(device, non_blocking=True)
                tod_b = batch["tod"].to(device, non_blocking=True)
                dow_b = batch["dow"].to(device, non_blocking=True)
                prior_n = batch["prior_norm_in"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    yn = model(x_norm, tod_b, dow_b, prior_n)
                y_pred = yn.float() * std_t + mean_t
                all_p.append(y_pred.cpu()); all_y.append(batch["y_node"]); all_m.append(batch["y_mask"])
        P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
        val_metrics = per_horizon(P, Y, M)
        val_mae = val_metrics["avg_mae"]
        scheduler.step()
        print(f"[ep {epoch:03d}/{args.epochs}] lr={opt.param_groups[0]['lr']:.2e} "
              f"train_mae={train_mae:.3f} val_mae={val_mae:.3f} "
              f"val_15/30/60={val_metrics.get('mae_15',0):.3f}/"
              f"{val_metrics.get('mae_30',0):.3f}/{val_metrics.get('mae_60',0):.3f}", flush=True)

        if val_mae < best_val - 1e-4:
            best_val = val_mae; epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_metrics": val_metrics, "epoch": epoch}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement in {args.patience} epochs"); break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            x_norm = batch["x_norm"].to(device, non_blocking=True)
            tod_b = batch["tod"].to(device, non_blocking=True)
            dow_b = batch["dow"].to(device, non_blocking=True)
            prior_n = batch["prior_norm_in"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                yn = model(x_norm, tod_b, dow_b, prior_n)
            y_pred = yn.float() * std_t + mean_t
            all_p.append(y_pred.cpu()); all_y.append(batch["y_node"]); all_m.append(batch["y_mask"])
    P = torch.cat(all_p); Y = torch.cat(all_y); M = torch.cat(all_m)
    test_metrics = per_horizon(P, Y, M)
    elapsed = time.time() - t_start
    print(f"[test] {json.dumps(test_metrics, indent=2)}")
    print(f"[done] elapsed={elapsed:.1f}s best_val={best_val:.3f}")

    row = {"model": "STAEformer_prior", "seed": args.seed, "epochs": ckpt["epoch"],
           "params_M": round(n_params / 1e6, 3), "elapsed_sec": round(elapsed, 1),
           **{f"val_{k}": ckpt["val_metrics"][k] for k in ckpt["val_metrics"]},
           **{f"test_{k}": v for k, v in test_metrics.items()},
           "tag": args.tag,
           "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    csv_path = os.path.join(args.out_dir, "staeformer_results.csv")
    pd.DataFrame([row]).to_csv(csv_path, mode="a",
                                header=not os.path.exists(csv_path), index=False)


if __name__ == "__main__":
    main()
