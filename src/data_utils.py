import pickle
import pandas as pd
import numpy as np


def load_metr_la_h5(h5_path: str) -> np.ndarray:
    """
    Load METR-LA traffic data from HDF5.
    
    Returns:
        data: shape [T, N]
            T = number of timesteps
            N = number of sensors
    """
    df = pd.read_hdf(h5_path)
    # Usually rows are timestamps and columns are sensors
    data = df.to_numpy(dtype=np.float32)
    return data


def load_adj_pkl(pkl_path: str):
    """
    Load adjacency info from METR-LA pickle file.

    Expected common structure:
        sensor_ids, sensor_id_to_ind, adj_mx
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # Common DCRNN-style format
    if isinstance(obj, tuple) and len(obj) == 3:
        sensor_ids, sensor_id_to_ind, adj_mx = obj
        return sensor_ids, sensor_id_to_ind, np.asarray(adj_mx, dtype=np.float32)

    raise ValueError("Unexpected adjacency pickle format.")