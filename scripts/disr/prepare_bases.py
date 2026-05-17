"""
Pre-compute and cache all DiSR-Mamba spectral bases & cluster assignments.

Runs once at the start of the campaign so individual training jobs don't pay
the eigendecomposition / lagged-correlation cost.

Caches written to:
    cache/gft/disr/sym_k{K}_{side}.npz
    cache/gft/disr/mag_k{K}_q{q:.2f}_{side}.npz
    cache/gft/disr/clusters_n{N}_v1.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_utils import load_metr_la_h5, load_adj_pkl  # noqa: E402
from models.disr.spectral_basis import load_or_build_symmetric_basis  # noqa: E402
from models.disr.magnetic_laplacian import magnetic_basis_from_adjacency  # noqa: E402
from models.disr.residual_router import build_sensor_clusters  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/METR-LA.h5")
    p.add_argument("--adj_path", type=str, default="data/adj_METR-LA.pkl")
    p.add_argument("--cache_dir", type=str, default="cache/gft/disr")
    p.add_argument("--K", type=int, nargs="+", default=[32, 48, 64])
    p.add_argument("--q", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.25])
    p.add_argument("--sides", type=str, nargs="+", default=["low"])
    p.add_argument("--n_clusters", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--train_frac", type=float, default=0.7)
    args = p.parse_args()

    os.chdir(ROOT)
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[prep] cache_dir={args.cache_dir}")

    # Load adjacency once
    _, _, A = load_adj_pkl(args.adj_path)
    A = np.asarray(A, dtype=np.float32)
    print(f"[prep] adjacency {A.shape}, symmetric? "
          f"{np.allclose(A, A.T, atol=1e-6)}")

    # Load training portion
    X = load_metr_la_h5(args.data_path)
    n_train = int(args.train_frac * X.shape[0])
    X_train = X[:n_train].astype(np.float32)
    print(f"[prep] X_train={X_train.shape}")

    # ----- Symmetric bases -----
    for K in args.K:
        for side in args.sides:
            cache = os.path.join(args.cache_dir, f"sym_k{K}_{side}.npz")
            t0 = time.time()
            ev, U = load_or_build_symmetric_basis(A, k=K, side=side,
                                                  cache_path=cache)
            dt = time.time() - t0
            print(f"[prep] sym K={K} side={side}: {U.shape}  ({dt:.1f}s)  -> {cache}")

    # ----- Magnetic bases -----
    for K in args.K:
        for side in args.sides:
            for q in args.q:
                cache = os.path.join(args.cache_dir,
                                      f"mag_k{K}_q{q:.2f}_{side}.npz")
                t0 = time.time()
                ev, U = magnetic_basis_from_adjacency(
                    A_sym=A, X_train=X_train, k=K, q=q, side=side,
                    cache_path=cache,
                )
                dt = time.time() - t0
                print(f"[prep] mag K={K} q={q:.2f} side={side}: "
                      f"{U.shape} complex={U.is_complex()}  ({dt:.1f}s)  -> {cache}")

    # ----- Cluster assignments -----
    for n in args.n_clusters:
        cache = os.path.join(args.cache_dir, f"clusters_n{n}_v1.npy")
        if os.path.exists(cache):
            cids = np.load(cache)
            print(f"[prep] clusters n={n}: cached -> {cache} (sizes "
                  f"{np.bincount(cids, minlength=n).tolist()})")
            continue
        t0 = time.time()
        cids = build_sensor_clusters(A, n_clusters=n, X_train=X_train,
                                       use_history=True, random_state=0)
        np.save(cache, cids)
        dt = time.time() - t0
        print(f"[prep] clusters n={n}: ({dt:.1f}s) sizes="
              f"{np.bincount(cids, minlength=n).tolist()}  -> {cache}")


if __name__ == "__main__":
    main()
