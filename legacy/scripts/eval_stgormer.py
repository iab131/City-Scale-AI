"""
Evaluate a trained STGormer checkpoint on METR-LA test set with PER-HORIZON
metrics, matching our standard masked-MAE-in-raw-mph protocol.

Loads STGormer's saved best_model.pth + uses its dataloader, but computes
metrics our way so results are directly comparable to STAEformer/GWNet
ensemble numbers.

Outputs the [B, 12, 207] test predictions in RAW MPH which can be saved for
later ensembling with our other models.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, "/workspace/STGormer")
from model.models import MoESTar
from lib.dataloader import get_dataloader_from_index_data
from lib.utils import load_graph, init_seed
from model.utils.trans_utils import get_shortpath_num, get_num_degree

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to STGormer best_model.pth")
    p.add_argument("--config", type=str, default="/workspace/STGormer/configs/stgormer/METRLA.yaml")
    p.add_argument("--data_dir", type=str, default="/workspace/STGormer/data")
    p.add_argument("--save_preds", type=str, default=None,
                   help="Optional: path to save raw-mph test predictions [.npz]")
    p.add_argument("--device", type=str, default="cuda")
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


def main():
    args = parse_args()

    # Load STGormer's config
    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)
    # Convert to namespace-like
    class C: pass
    cfg = C()
    for k, v in cfg_yaml.items():
        setattr(cfg, k, v)
    cfg.device = args.device

    init_seed(cfg.seed)

    # Build dataloader exactly like STGormer's training script does
    dataloader = get_dataloader_from_index_data(
        data_dir=args.data_dir, dataset=cfg.dataset,
        d_input=cfg.d_input, d_output=cfg.d_output,
        batch_size=cfg.batch_size, test_batch_size=cfg.test_batch_size,
    )
    graph = load_graph(cfg.graph_file, device=cfg.device)
    cfg.num_shortpath, _ = get_shortpath_num(graph, cfg.dataset)
    cfg.num_node_deg = get_num_degree(graph)

    # Build model
    model = MoESTar(cfg).to(cfg.device)
    state = torch.load(args.ckpt, map_location=cfg.device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"[loaded] {args.ckpt}")

    # Inference
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in dataloader["test"]:
            repr_, *_ = model(data, graph)
            pred_output = model.predict(repr_)
            y_true.append(target)
            y_pred.append(pred_output)
    y_true = dataloader["scaler"].inverse_transform(torch.cat(y_true, dim=0))
    y_pred = dataloader["scaler"].inverse_transform(torch.cat(y_pred, dim=0))
    # Shapes: [N_samples, T_out, N_nodes, 1] for METRLA (single channel)
    y_pred = y_pred[..., 0]                     # [N_samples, T_out, N]
    y_true = y_true[..., 0]                     # [N_samples, T_out, N]

    # Mask = y_true != 0 (matches our STAEformer eval protocol)
    mask = (y_true != 0).float()

    pred = y_pred.cpu()
    true = y_true.cpu()
    mk = mask.cpu()
    metrics = per_horizon(pred, true, mk)
    print("\n--- STGormer test metrics (masked, per-horizon, raw mph) ---")
    print(json.dumps(metrics, indent=2))

    print("\n--- Compared to published ---")
    print(f"  STGormer paper H12: MAE 3.10  | ours: {metrics['mae_60']:.3f}")
    print(f"  STGormer paper H12: RMSE 6.34 | ours: {metrics['rmse_60']:.3f}")
    print(f"  STAEformer paper H12: MAE 3.34 | ours: 3.34 (matches)")

    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or ".", exist_ok=True)
        np.savez(args.save_preds, pred=pred.numpy(), true=true.numpy(), mask=mk.numpy())
        print(f"[saved] predictions -> {args.save_preds}")


if __name__ == "__main__":
    main()
