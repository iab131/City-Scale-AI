"""Read all results CSVs + ensemble JSONs and print a single leaderboard
sorted by 60-min test MAE. Run this at any point to see overall status.
"""

import os
import sys
import json
import glob
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = []

    # STAEformer (and STAEformer mixup) — single models
    for r in read_csv_rows("results/staeformer/staeformer_results.csv"):
        rows.append({
            "kind": "stae" if r["model"] != "STAEformer_mixup" else "stae_mixup",
            "tag": r["tag"],
            "seed": r.get("seed", "-"),
            "mae_15": float(r.get("test_mae_15", 0)),
            "mae_30": float(r.get("test_mae_30", 0)),
            "mae_60": float(r.get("test_mae_60", 0)),
            "params_M": r.get("params_M", "-"),
            "epochs": r.get("epochs", "-"),
        })
    for r in read_csv_rows("results/gwnet/gwnet_results.csv"):
        rows.append({
            "kind": "gwnet", "tag": r["tag"], "seed": r.get("seed", "-"),
            "mae_15": float(r.get("test_mae_15", 0)),
            "mae_30": float(r.get("test_mae_30", 0)),
            "mae_60": float(r.get("test_mae_60", 0)),
            "params_M": r.get("params_M", "-"), "epochs": r.get("epochs", "-"),
        })
    for r in read_csv_rows("results/hybrid/hybrid_results.csv"):
        rows.append({
            "kind": "hybrid", "tag": r["tag"], "seed": r.get("seed", "-"),
            "mae_15": float(r.get("test_mae_15", 0)),
            "mae_30": float(r.get("test_mae_30", 0)),
            "mae_60": float(r.get("test_mae_60", 0)),
            "params_M": r.get("params_M", "-"), "epochs": r.get("epochs", "-"),
        })
    for r in read_csv_rows("results/stae_pretrained/stae_pretrained_results.csv"):
        rows.append({
            "kind": "stae_pre", "tag": r["tag"], "seed": r.get("seed", "-"),
            "mae_15": float(r.get("test_mae_15", 0)),
            "mae_30": float(r.get("test_mae_30", 0)),
            "mae_60": float(r.get("test_mae_60", 0)),
            "params_M": "-", "epochs": r.get("epochs", "-"),
        })

    # Ensemble JSONs
    for path in sorted(glob.glob("results/R0*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            for name, m in d.get("ensembles", {}).items():
                rows.append({
                    "kind": "ensemble",
                    "tag": f"{os.path.basename(path).replace('.json', '')}:{name}",
                    "seed": "-",
                    "mae_15": m.get("mae_15", 0),
                    "mae_30": m.get("mae_30", 0),
                    "mae_60": m.get("mae_60", 0),
                    "params_M": "-", "epochs": "-",
                })
            if "moe_gated" in d:
                rows.append({
                    "kind": "moe",
                    "tag": os.path.basename(path).replace('.json', '') + ":moe_gated",
                    "seed": "-",
                    "mae_15": d["moe_gated"].get("mae_15", 0),
                    "mae_30": d["moe_gated"].get("mae_30", 0),
                    "mae_60": d["moe_gated"].get("mae_60", 0),
                    "params_M": "-", "epochs": "-",
                })
            # R08 stacking output has 'base' and 'stacked' keys
            for k in ("base", "stacked"):
                if k in d and isinstance(d[k], dict) and "mae_60" in d[k]:
                    rows.append({
                        "kind": "stacking",
                        "tag": os.path.basename(path).replace('.json', '') + ":" + k,
                        "seed": "-",
                        "mae_15": d[k].get("mae_15", 0),
                        "mae_30": d[k].get("mae_30", 0),
                        "mae_60": d[k].get("mae_60", 0),
                        "params_M": "-", "epochs": "-",
                    })
        except Exception as e:
            print(f"warn: failed to read {path}: {e}")

    # Sort by 60-min MAE
    rows = [r for r in rows if r["mae_60"] > 0]
    rows.sort(key=lambda r: r["mae_60"])

    # Print
    print(f"\n{'='*100}")
    print(f"{'rank':>4}  {'kind':<10}  {'mae_60':>7}  {'mae_30':>7}  {'mae_15':>7}  {'tag':<60}")
    print(f"{'='*100}")
    for i, r in enumerate(rows[:50], 1):
        print(f"{i:>4}  {r['kind']:<10}  {r['mae_60']:7.4f}  {r['mae_30']:7.4f}  "
              f"{r['mae_15']:7.4f}  {r['tag'][:60]}")
    print(f"{'='*100}")
    print(f"\nTotal entries: {len(rows)}")
    print(f"Reference: REPORT.md headline = 3.283 (4-seed STAE + ST-TTC v1)")
    print(f"Target:    < 3.20 (clearly SOTA) — or below 3.14 (TESTAM, unreproducible)")

if __name__ == "__main__":
    main()
