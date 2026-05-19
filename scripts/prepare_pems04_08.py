"""
Download (if missing) and prepare PEMS04 / PEMS08 traffic-flow data for use
with our STAE-Spectral-Magma pipeline.

Outputs into data/ (matching the convention used for METR-LA and PEMS-BAY):
  - data/pems04.npz         raw traffic tensor [T, N, 3] (flow, occupancy, speed)
  - data/distance_pems04.csv per-edge distance file (from, to, cost)
  - data/adj_PEMS04.pkl     DCRNN-style 3-tuple (ids, id_to_idx, adj_mx)

Same three files for PEMS08.

Usage:
    python scripts/prepare_pems04_08.py            # both
    python scripts/prepare_pems04_08.py --only pems04
"""
from __future__ import annotations
import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from data_utils import (
    load_pems_npz, build_pems_adj_from_csv, save_pems_adj_pkl,
)


# Public mirrors of the AGCRN-distributed PEMS04/08 files. ASTGCN/AGCRN/STSGCN
# all use the same files; we point at a stable raw-content mirror that has
# been around since 2022.
SOURCES = {
    "pems04": {
        "npz": "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/PEMS04.npz",
        "csv": "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/distance.csv",
        "n_nodes": 307,
    },
    "pems08": {
        "npz": "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS08/PEMS08.npz",
        "csv": "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS08/distance.csv",
        "n_nodes": 170,
    },
}


def _download(url: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"  [skip] {dst} already exists ({os.path.getsize(dst):,} bytes)")
        return
    print(f"  [get ] {url}\n         -> {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(url, dst)
    print(f"  [ok  ] {os.path.getsize(dst):,} bytes")


def prepare(name: str, data_dir: str = "data") -> None:
    spec = SOURCES[name]
    npz_path = os.path.join(data_dir, f"{name}.npz")
    csv_path = os.path.join(data_dir, f"distance_{name}.csv")
    adj_path = os.path.join(data_dir, f"adj_PEMS{name[-2:]}.pkl")

    print(f"=== {name.upper()} ===")
    _download(spec["npz"], npz_path)
    _download(spec["csv"], csv_path)

    # Sanity-check the data tensor
    X_full = np.load(npz_path)["data"]
    print(f"  [data] shape={X_full.shape} dtype={X_full.dtype}")
    X = load_pems_npz(npz_path, feature="flow")
    print(f"  [flow] shape={X.shape}  min={X.min():.1f} max={X.max():.1f} "
          f"mean={X.mean():.1f}")

    # Build the symmetric Gaussian-kernel adjacency once and cache it.
    n_nodes = spec["n_nodes"]
    if not os.path.exists(adj_path):
        W = build_pems_adj_from_csv(csv_path, n_nodes=n_nodes)
        edges = int((W > 0).sum() - n_nodes)
        print(f"  [adj ] n={n_nodes}, edges_after_threshold={edges}")
        save_pems_adj_pkl(adj_path, W)
        print(f"  [save] {adj_path}")
    else:
        print(f"  [skip] {adj_path} already built")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["pems04", "pems08"], default=None,
                   help="prepare a single dataset instead of both")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    args = p.parse_args()
    names = [args.only] if args.only else ["pems04", "pems08"]
    for n in names:
        prepare(n, data_dir=args.data_dir)
    print("\ndone.")


if __name__ == "__main__":
    main()
