"""
Generate paper-ready plots from one or more DiSR-Mamba run outputs.

Produces (per run, in `results/disr/<tag>/plots/`):
  - train_val_curves.png
  - per_horizon_mae.png
  - per_speed_regime_mae.png

If multiple runs share a base tag (e.g., q-sweep), also writes a sweep
sensitivity plot at the root output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_run(run_dir: str):
    plt = _import_mpl()
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    log_csv = os.path.join(run_dir, "log.csv")
    if os.path.exists(log_csv):
        df = pd.read_csv(log_csv)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(df["epoch"], df["train_loss"], label="train_loss")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
        ax[0].set_title("Train loss")
        ax[1].plot(df["epoch"], df["val_mae"], label="val_avg_mae")
        ax[1].plot(df["epoch"], df["val_mae_60"], label="val_mae_60", linestyle="--")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MAE"); ax[1].legend()
        ax[1].set_title("Val MAE")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "train_val_curves.png"), dpi=120)
        plt.close()

    per_h = os.path.join(run_dir, "per_horizon.json")
    if os.path.exists(per_h):
        d = json.load(open(per_h))
        keys = sorted(d.keys(), key=lambda k: int(k.lstrip("h")))
        mae = [d[k]["mae_15"] if "mae_15" in d[k] else d[k]["avg_mae"] for k in keys]
        # Per-horizon: each json contains a single-step view, use "avg_mae" of that horizon.
        mae = [d[k]["avg_mae"] for k in keys]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(mae) + 1), mae, marker="o")
        ax.set_xlabel("forecast horizon (5-min steps)")
        ax.set_ylabel("MAE (mph)")
        ax.set_title("Per-horizon test MAE")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_horizon_mae.png"), dpi=120)
        plt.close()

    psr = os.path.join(run_dir, "per_speed_regime.json")
    if os.path.exists(psr):
        d = json.load(open(psr))
        names = ["lt20", "20_40", "40_60", "ge60"]
        labels = ["<20", "20-40", "40-60", ">=60"]
        vals = [d.get(f"mae_{n}", 0.0) for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals)
        ax.set_xlabel("ground-truth speed (mph)")
        ax.set_ylabel("masked MAE (mph)")
        ax.set_title("Per-speed-regime test MAE")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_speed_regime_mae.png"), dpi=120)
        plt.close()
    print(f"[plots] {run_dir} -> {out_dir}")


def plot_q_sensitivity(root: str):
    """
    If we find runs named stageD_q*_K48, plot 60-min test MAE vs q.
    """
    plt = _import_mpl()
    summaries = sorted(glob(os.path.join(root, "stageD_q*", "summary.json")))
    if not summaries:
        return
    qs, maes = [], []
    for p in summaries:
        s = json.load(open(p))
        tag = s.get("tag", "")
        try:
            q = float(tag.split("q")[1].split("_")[0]) / 100.0  # "q010" -> 0.10
        except Exception:
            try:
                q = float(tag.split("_q")[1].split("_")[0])
            except Exception:
                continue
        qs.append(q)
        maes.append(float(s.get("test_metrics", {}).get("mae_60", float("nan"))))
    if not qs:
        return
    order = np.argsort(qs)
    qs = np.array(qs)[order]
    maes = np.array(maes)[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(qs, maes, marker="o")
    ax.set_xlabel("magnetic charge q")
    ax.set_ylabel("60-min test MAE (mph)")
    ax.set_title("q-charge sensitivity (Stage D)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(root, "q_sensitivity.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plots] {out}")


def plot_K_sensitivity(root: str):
    plt = _import_mpl()
    summaries = sorted(glob(os.path.join(root, "stageC_K*", "summary.json")))
    if not summaries:
        return
    ks, maes = [], []
    for p in summaries:
        s = json.load(open(p))
        tag = s.get("tag", "")
        try:
            k = int(tag.split("K")[1].split("_")[0])
        except Exception:
            continue
        ks.append(k)
        maes.append(float(s.get("test_metrics", {}).get("mae_60", float("nan"))))
    if not ks:
        return
    order = np.argsort(ks)
    ks = np.array(ks)[order]
    maes = np.array(maes)[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, maes, marker="o")
    ax.set_xlabel("K spectral modes")
    ax.set_ylabel("60-min test MAE (mph)")
    ax.set_title("K-mode sensitivity (Stage C)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(root, "K_sensitivity.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plots] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=os.path.join(ROOT, "results", "disr"))
    p.add_argument("--runs", type=str, default="",
                   help="Optional comma-separated subset of run dirs")
    args = p.parse_args()

    if args.runs:
        run_dirs = [os.path.join(args.root, r) for r in args.runs.split(",") if r]
    else:
        run_dirs = sorted(glob(os.path.join(args.root, "*", "log.csv")))
        run_dirs = [os.path.dirname(p) for p in run_dirs]
    for d in run_dirs:
        plot_run(d)
    plot_q_sensitivity(args.root)
    plot_K_sensitivity(args.root)


if __name__ == "__main__":
    main()
