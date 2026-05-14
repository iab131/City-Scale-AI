"""
Prepare STGormer's expected data layout from our METR-LA preprocessing.

STGormer expects:
  data/METRLA/data.npz  with key "data" of shape [T, N, 3]
                        = (raw_speed, tod_in_[0,1), dow_int)
  data/METRLA/index.npz with keys train/val/test each [num_samples, 3]
                        = (x_start, y_start, y_end)
  data/METRLA/adj_mx.npz with the symmetric adjacency

Uses our existing preprocess_v2 to get raw X and tod/dow features.
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

from data_utils import load_metr_la_h5, load_adj_pkl
from graph_utils import symmetrize_adjacency
from preprocess_v2 import build_time_features


def main():
    data_dir = "/workspace/STGormer/data/METRLA"
    os.makedirs(data_dir, exist_ok=True)

    # Load our data
    X = load_metr_la_h5("/workspace/city-scale-ai/data/METR-LA.h5")  # [T, N]
    _, _, A = load_adj_pkl("/workspace/city-scale-ai/data/adj_METR-LA.pkl")
    A = symmetrize_adjacency(A).astype(np.float32)

    T, N = X.shape
    tod, dow = build_time_features(T)               # tod: [T] in [0,1), dow: [T] int

    # data: [T, N, 3] with channels (raw_speed, tod, dow)
    data = np.zeros((T, N, 3), dtype=np.float32)
    data[..., 0] = X
    data[..., 1] = tod[:, None]
    data[..., 2] = dow[:, None].astype(np.float32)

    # Build train/val/test sliding-window indices (70/10/20 chronological)
    input_len = 12
    output_len = 12
    history_seq_len = input_len
    future_seq_len = output_len
    # Indexes follow STGormer's convention: x_start, y_start (= x_end), y_end
    # x = data[x_start : x_start + input_len], y = data[y_start : y_start + output_len]
    indices = []
    for t in range(T - input_len - output_len + 1):
        x_start = t
        y_start = t + input_len
        y_end = y_start + output_len
        indices.append((x_start, y_start, y_end))
    indices = np.array(indices, dtype=np.int64)        # [num_samples, 3]
    num_samples = len(indices)
    n_train = int(0.7 * num_samples)
    n_val = int(0.1 * num_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"T={T}, N={N}, samples={num_samples}")
    print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    np.savez(os.path.join(data_dir, "data.npz"), data=data)
    np.savez(os.path.join(data_dir, "index.npz"),
             train=train_idx, val=val_idx, test=test_idx)
    np.savez(os.path.join(data_dir, "adj_mx.npz"), adj_mx=A)
    print(f"[done] wrote to {data_dir}")


if __name__ == "__main__":
    main()
