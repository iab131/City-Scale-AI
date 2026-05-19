import os
import pickle
import math
import pandas as pd
import numpy as np


def load_metr_la_h5(h5_path: str) -> np.ndarray:
    """
    Load METR-LA / PEMS-BAY-style traffic data from HDF5.

    Supports both pandas HDFStore group conventions:
      - METR-LA distribution stores the table under group key "df"
      - PEMS-BAY distribution stores it under group key "data"
    Both expose `<group>/block0_values` with shape [T, N] (after a possible
    transpose for METR-LA's legacy [N, T] layout).

    Returns:
        data: shape [T, N]
            T = number of timesteps
            N = number of sensors
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        top_keys = list(f.keys())
        if "df" in top_keys:
            group = "df"
        elif "data" in top_keys:
            group = "data"
        else:
            raise ValueError(f"Unknown HDF5 layout (top-level keys: {top_keys})")
        data = f[group]["block0_values"][:]
        # METR-LA legacy layout: [N=207, T=34272]; transpose to [T, N].
        if data.shape[0] == 207 and data.shape[1] == 34272:
            data = data.T
    return data.astype(np.float32)


def load_adj_pkl(pkl_path: str):
    """
    Load adjacency info from METR-LA pickle file.

    Expected common structure:
        sensor_ids, sensor_id_to_ind, adj_mx
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # Common DCRNN-style format
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        sensor_ids, sensor_id_to_ind, adj_mx = obj
        return sensor_ids, sensor_id_to_ind, np.asarray(adj_mx, dtype=np.float32)

    raise ValueError("Unexpected adjacency pickle format.")


# ---------------------------------------------------------------------------
# PEMS04 / PEMS08 (.npz + distance CSV) — different convention from METR-LA
#   - data array: [T, N, F=3] with features (flow, occupancy, speed). The
#     canonical traffic-forecasting protocol uses flow only.
#   - no zero-missing convention: data is complete (no mask)
#   - adjacency given as a CSV (from, to, cost) over integer node indices
#     directly (no sensor-id translation step like DCRNN)
# ---------------------------------------------------------------------------
def load_pems_npz(npz_path: str, feature: str = "flow") -> np.ndarray:
    """Load PEMS04/PEMS08 .npz traffic data.

    Args:
        npz_path: path to PEMS04.npz / PEMS08.npz (key 'data', [T, N, F]).
        feature: one of {flow, occupancy, speed}. Default flow.

    Returns:
        X: float32 array [T, N] with the chosen feature.
    """
    feat_idx = {"flow": 0, "occupancy": 1, "speed": 2}[feature]
    arr = np.load(npz_path)["data"]                                    # [T, N, F]
    if arr.ndim == 2:
        X = arr
    elif arr.ndim == 3:
        X = arr[..., feat_idx]
    else:
        raise ValueError(f"Unexpected PEMS .npz shape: {arr.shape}")
    return X.astype(np.float32)


def build_pems_adj_from_csv(csv_path: str, n_nodes: int) -> np.ndarray:
    """Build the symmetric thresholded-Gaussian adjacency used by the DCRNN/
    STAEformer family, from PEMS04/08-style (from, to, cost) CSV files.

    The convention matches what we did for PEMS-BAY in build_pems_bay_adj —
    Gaussian kernel exp(-d^2/sigma^2), then threshold at 0.1.
    """
    df = pd.read_csv(csv_path)
    # PEMS04/08 distance CSVs use either ('from','to','cost') or
    # ('from','to','distance'); accept either.
    cols = {c.lower(): c for c in df.columns}
    f_col = cols["from"]
    t_col = cols["to"]
    d_col = cols.get("cost", cols.get("distance"))
    if d_col is None:
        raise ValueError(f"CSV needs a cost/distance column, got: {list(df.columns)}")

    dist_mx = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
    np.fill_diagonal(dist_mx, 0.0)
    for f_, t_, d_ in zip(df[f_col], df[t_col], df[d_col]):
        i, j = int(f_), int(t_)
        if 0 <= i < n_nodes and 0 <= j < n_nodes:
            dist_mx[i, j] = float(d_)

    finite = dist_mx[(dist_mx > 0) & np.isfinite(dist_mx)]
    if finite.size == 0:
        raise ValueError("no finite distances found in CSV")
    sigma = float(finite.std())
    W = np.exp(-(dist_mx ** 2) / (sigma ** 2 + 1e-9))
    W[W < 0.1] = 0.0
    return W.astype(np.float32)


def save_pems_adj_pkl(out_path: str, W: np.ndarray) -> None:
    """Save a PEMS adjacency in DCRNN-compatible 3-tuple format so
    load_adj_pkl picks it up unchanged."""
    n = W.shape[0]
    sensor_ids = list(range(n))
    sensor_id_to_ind = {i: i for i in sensor_ids}
    with open(out_path, "wb") as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, W.astype(np.float32)), f)