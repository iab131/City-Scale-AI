"""
Aggregate DiSR-Mamba run summaries into a single ablation table.

Reads `results/disr/<tag>/summary.json` for each completed run and emits:
  * ``results/disr/ablation_table.csv`` (one row per run)
  * ``results/disr/ablation_table.md``  (human-readable table)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default=os.path.join(ROOT, "results", "disr"))
    return p.parse_args()


def main():
    args = parse_args()
    summaries = sorted(glob.glob(os.path.join(args.root, "*", "summary.json")))
    if not summaries:
        print(f"[aggregate] no summaries under {args.root}")
        return
    rows = []
    for p in summaries:
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception as e:
            print(f"[aggregate] skip {p}: {e}")
            continue
        m = s.get("test_metrics", {})
        rows.append({
            "tag": s.get("tag"),
            "seed": s.get("seed"),
            "experts": "|".join(s.get("expert_names", [])),
            "trunk_ckpt": os.path.basename(s.get("trunk_ckpt", "")),
            "best_val_mae": round(float(s.get("best_val_mae", float("nan"))), 4),
            "test_mae_15": round(float(m.get("mae_15", float("nan"))), 4),
            "test_mae_30": round(float(m.get("mae_30", float("nan"))), 4),
            "test_mae_60": round(float(m.get("mae_60", float("nan"))), 4),
            "test_avg_mae": round(float(m.get("avg_mae", float("nan"))), 4),
            "test_rmse_60": round(float(m.get("rmse_60", float("nan"))), 4),
            "elapsed_sec": int(s.get("elapsed_sec", 0)),
            "max_gpu_mb": int(s.get("max_gpu_mb", 0)),
        })
    rows.sort(key=lambda r: (r["test_mae_60"]
                              if not isinstance(r["test_mae_60"], float)
                                 or r["test_mae_60"] == r["test_mae_60"]
                              else 1e9, r["tag"]))

    csv_path = os.path.join(args.root, "ablation_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {csv_path} ({len(rows)} rows)")

    md_path = os.path.join(args.root, "ablation_table.md")
    cols = ["tag", "seed", "experts", "test_mae_15", "test_mae_30",
            "test_mae_60", "test_avg_mae", "best_val_mae"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {md_path}")

    # Pretty-print top 10 in stdout
    print("\nTop 10 by 60-min test MAE:")
    header = " ".join(f"{c:>14}" for c in cols)
    print(header)
    for r in rows[:10]:
        print(" ".join(f"{str(r[c]):>14}" for c in cols))


if __name__ == "__main__":
    main()
