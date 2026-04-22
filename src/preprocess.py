import os
import numpy as np

from data_utils import load_metr_la_h5, load_adj_pkl
from graph_utils import normalized_laplacian
from gft import compute_gft_basis, gft

def get_cached_gft_data(data_path, adj_path, k, cache_dir="cache/gft"):
    """
    Loads GFT artifacts from disk if they exist; otherwise computes and saves them.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    mean_path = os.path.join(cache_dir, "mean.npy")
    std_path = os.path.join(cache_dir, "std.npy")
    L_path = os.path.join(cache_dir, "L.npy")
    evals_path = os.path.join(cache_dir, f"evals_k{k}.npy")
    U_path = os.path.join(cache_dir, f"U_k{k}.npy")
    X_hat_path = os.path.join(cache_dir, f"X_hat_k{k}.npy")
    
    # Check if all files for this k exist
    if (os.path.exists(mean_path) and os.path.exists(std_path) and 
        os.path.exists(L_path) and os.path.exists(evals_path) and 
        os.path.exists(U_path) and os.path.exists(X_hat_path)):
        print(f"Loading cached GFT artifacts for k={k}...")
        mean = np.load(mean_path)
        std = np.load(std_path)
        L_arr = np.load(L_path, allow_pickle=True)
        L = L_arr.item() if L_arr.shape == () else L_arr
        evals = np.load(evals_path)
        U = np.load(U_path)
        X_hat = np.load(X_hat_path)
        return mean, std, L, evals, U, X_hat
        
    print(f"Computing and saving new GFT artifacts for k={k}...")
    # Load raw data if needed
    X = load_metr_la_h5(data_path)
    _, _, A = load_adj_pkl(adj_path)
    
    # Compute mean and std if not cached
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        np.save(mean_path, mean)
        np.save(std_path, std)
        
    X_norm = (X - mean) / std
    
    # Compute L if not cached
    if os.path.exists(L_path):
        L_arr2 = np.load(L_path, allow_pickle=True)
        L = L_arr2.item() if L_arr2.shape == () else L_arr2
    else:
        L = normalized_laplacian(A)
        np.save(L_path, L)
        
    # Compute U, evals
    evals, U = compute_gft_basis(L, k=k)
    np.save(evals_path, evals)
    np.save(U_path, U)
    
    # Compute X_hat
    X_hat = gft(X_norm, U)
    np.save(X_hat_path, X_hat)
    
    return mean, std, L, evals, U, X_hat
