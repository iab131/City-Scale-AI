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
    import h5py
    with h5py.File(h5_path, "r") as f:
        # Pytables/pandas often stores the 2D array in 'df/block0_values'
        # The shape is usually [N, T] or [T, N]. 
        data = f["df"]["block0_values"][:]
        # In pandas HDFStore, if shape is (N, T), it might be transposed when read as DataFrame
        # METR-LA typically has T=34272, N=207. Let's ensure it returns [T, N].
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