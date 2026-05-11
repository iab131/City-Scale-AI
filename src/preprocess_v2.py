"""
v2 preprocessing: masked normalization, time-of-day & day-of-week features.

Backwards-compatible with the existing cache layout but writes new files
(prefixed `v2_`) to avoid clobbering the original artifacts. The GFT
eigenvectors/eigenvalues are reused as-is from the existing cache (the
proposal says "GFT is good, don't touch") - only the normalization stats
and the windowed inputs change.
"""

import os
import numpy as np

from data_utils import load_metr_la_h5, load_adj_pkl
from graph_utils import normalized_laplacian
from gft import compute_gft_basis, gft


METR_LA_T_PER_DAY = 288  # 5-min cadence -> 24*60/5 = 288
SECONDS_PER_DAY = 24 * 3600


def compute_masked_stats(X, missing_val=0.0):
    """Mean/std computed across (T, N), ignoring entries equal to missing_val."""
    mask = X != missing_val  # [T, N] bool
    # per-sensor stats are robust because each sensor has different scale
    # but METR-LA convention is global stats. Use global to match published baselines.
    valid = X[mask]
    mean = valid.mean()
    std = valid.std() + 1e-6
    return float(mean), float(std), mask


def build_time_features(T, start_idx=0):
    """
    METR-LA spans 2012-03-01 to 2012-06-30 at 5-min cadence.
    We don't have absolute timestamps in the loaded array, but the dataset
    is contiguous so we can derive (time-of-day, day-of-week) from index.

    Returns:
      tod: [T] float in [0, 1)
      dow: [T] int in {0..6}
    """
    idx = np.arange(T, dtype=np.int64) + start_idx
    tod = (idx % METR_LA_T_PER_DAY).astype(np.float32) / METR_LA_T_PER_DAY
    # METR-LA officially starts on Thursday 2012-03-01, dow=3 (Mon=0).
    dow = ((idx // METR_LA_T_PER_DAY + 3) % 7).astype(np.int64)
    return tod, dow


def compute_calendar_prior(X_train, tod_train, dow_train, missing_mask_train,
                            n_tod=METR_LA_T_PER_DAY, n_dow=7):
    """
    Compute per-sensor mean speed conditioned on (TOD-bin, DOW).

    For each (sensor n, tod_idx t in [0..287], dow d in [0..6]) compute the
    mean of X[t', n] over all train timesteps t' such that:
      - tod_idx(t') == t  (same 5-min slot in the day)
      - dow(t') == d
      - X[t', n] != 0     (valid reading)

    Returns:
      prior_table  [N, n_tod, n_dow]   mean speed in raw mph (zeros = no data)
    """
    T, N = X_train.shape
    tod_idx = (tod_train * n_tod).astype(np.int64) % n_tod  # [T]
    # accumulators
    sums = np.zeros((N, n_tod, n_dow), dtype=np.float64)
    cnts = np.zeros((N, n_tod, n_dow), dtype=np.float64)
    for t in range(T):
        ti, di = int(tod_idx[t]), int(dow_train[t])
        mask = missing_mask_train[t]                 # [N], 1.0 where valid
        x = X_train[t]                               # [N]
        sums[:, ti, di] += x * mask
        cnts[:, ti, di] += mask
    # Avoid div-by-zero: where no obs, fall back to global mean
    global_mean = X_train[missing_mask_train.astype(bool)].mean()
    prior = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), global_mean)
    return prior.astype(np.float32)


def lookup_calendar_prior(prior_table, tod, dow, n_tod=METR_LA_T_PER_DAY):
    """
    prior_table: [N, n_tod, n_dow]
    tod: [T] float in [0,1)
    dow: [T] int
    returns: [T, N] in raw mph
    """
    T = len(tod)
    tod_idx = (tod * n_tod).astype(np.int64) % n_tod  # [T]
    # prior_table is [N, n_tod, n_dow] -> grab [T, N]
    return prior_table[:, tod_idx, dow.astype(np.int64)].T.astype(np.float32)


def get_cached_v2_data(data_path, adj_path, k, cache_dir="cache/gft"):
    """
    Loads (or computes + caches) all artifacts required for the v2 pipeline.

    Returns dict with:
      X            [T, N]            raw speeds
      X_norm       [T, N]            (X - mean) / std,  but only over valid (X!=0) entries
      mean, std    scalars           global stats over valid entries
      U            [N, k]            eigenvectors (reused from existing cache)
      evals        [k]               eigenvalues  (reused)
      tod          [T]               time-of-day in [0,1)
      dow          [T]               day-of-week int
      missing_mask [T, N]            True where X != 0 (valid reading)
      prior        [T, N]            calendar prior in raw mph
      prior_norm   [T, N]            calendar prior in normalized space
    """
    os.makedirs(cache_dir, exist_ok=True)

    mean_p = os.path.join(cache_dir, "v2_mean_train.npy")
    std_p = os.path.join(cache_dir, "v2_std_train.npy")
    evals_p = os.path.join(cache_dir, f"evals_k{k}_train.npy")  # reuse v1
    U_p = os.path.join(cache_dir, f"U_k{k}_train.npy")          # reuse v1
    tod_p = os.path.join(cache_dir, "v2_tod.npy")
    dow_p = os.path.join(cache_dir, "v2_dow.npy")

    X = load_metr_la_h5(data_path)
    T, N = X.shape

    if os.path.exists(mean_p) and os.path.exists(std_p):
        mean = float(np.load(mean_p))
        std = float(np.load(std_p))
    else:
        n_train = int(0.7 * T)
        mean, std, _ = compute_masked_stats(X[:n_train])
        np.save(mean_p, np.array(mean, dtype=np.float32))
        np.save(std_p, np.array(std, dtype=np.float32))
        print(f"[v2] computed masked train stats: mean={mean:.4f}, std={std:.4f}")

    # Eigenvectors/eigenvalues: reuse existing cache; compute if absent.
    if os.path.exists(evals_p) and os.path.exists(U_p):
        evals = np.load(evals_p)
        U = np.load(U_p)
    else:
        _, _, A = load_adj_pkl(adj_path)
        L = normalized_laplacian(A)
        evals, U = compute_gft_basis(L, k=k)
        np.save(evals_p, evals)
        np.save(U_p, U)

    # Time features
    if os.path.exists(tod_p) and os.path.exists(dow_p):
        tod = np.load(tod_p)
        dow = np.load(dow_p)
    else:
        tod, dow = build_time_features(T)
        np.save(tod_p, tod)
        np.save(dow_p, dow)

    missing_mask = (X != 0).astype(np.float32)  # 1 where valid
    # Important: use masked-normalization on the whole series (eval uses train stats)
    X_norm = (X - mean) / std

    # ---- calendar prior (per-sensor speed conditioned on TOD-bin, DOW) ----
    prior_table_p = os.path.join(cache_dir, "v2_calendar_prior_table.npy")
    prior_p = os.path.join(cache_dir, "v2_prior_full.npy")
    if os.path.exists(prior_table_p) and os.path.exists(prior_p):
        prior_table = np.load(prior_table_p)
        prior = np.load(prior_p)
    else:
        n_train = int(0.7 * T)
        prior_table = compute_calendar_prior(
            X[:n_train], tod[:n_train], dow[:n_train], missing_mask[:n_train]
        )                                                        # [N, 288, 7]
        prior = lookup_calendar_prior(prior_table, tod, dow)     # [T, N]
        np.save(prior_table_p, prior_table)
        np.save(prior_p, prior)
        print(f"[v2] calendar prior: range "
              f"[{prior.min():.2f}, {prior.max():.2f}], "
              f"mean={prior.mean():.2f}")

    prior_norm = ((prior - mean) / std).astype(np.float32)

    return {
        "X": X.astype(np.float32),
        "X_norm": X_norm.astype(np.float32),
        "mean": mean,
        "std": std,
        "U": U.astype(np.float32),
        "evals": evals.astype(np.float32),
        "tod": tod,
        "dow": dow,
        "missing_mask": missing_mask,
        "prior": prior.astype(np.float32),
        "prior_norm": prior_norm,
        "prior_table": prior_table,
    }
