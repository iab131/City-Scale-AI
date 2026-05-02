# City-Scale Traffic Forecasting with Graph Fourier Transform (GFT)

This project implements a traffic forecasting pipeline on the **METR-LA** dataset using the **Graph Fourier Transform (GFT)** as the spatial representation step.

The idea is:

1. Load traffic signals from METR-LA
2. Build the road sensor graph from the provided adjacency matrix
3. Compute the normalized graph Laplacian
4. Perform eigendecomposition to obtain the graph Fourier basis
5. Transform traffic signals into the spectral domain
6. Train a temporal model on spectral coefficients
7. Reconstruct predictions back to sensor space with inverse GFT

---

## Project Goal

Standard graph models often rely on local message passing.  
This project instead moves node signals into the **spectral domain**, where traffic can be represented as global graph modes.

This makes it easier to model:

- city-wide congestion patterns
- long-range dependencies
- smooth vs. abrupt traffic variations

---

## Dataset

We use the **METR-LA** traffic forecasting dataset.

Expected files:

```text
data/metr_la/
├── metr-la.h5
├── adj_METR-LA.pkl
```

---

## Running Mamba k-Sweep Experiments

The repository includes a Spectral Mamba model. A critical hyperparameter is `k` (the number of graph Fourier components kept). You can easily run experiments to sweep over `k`.

**Important**: The comparison should be fair: only `k` changes during the sweep unless you explicitly change other hyperparameters (`d_model`, `num_layers`, `batch_size`, or `epochs`). The model still uses the same pipeline: data is GFT-transformed using `get_cached_gft_data(...)`, and the Mamba model (`SpectralMambaReal`) trains on these spectral coefficients.

### 1. Run a smoke test
Quick test to make sure everything runs without errors:
```bash
python scripts/train_mamba.py --device cuda --k 8 --epochs 1 --batch_size 4
```

### 2. Manual single run
Run with a specific `k` value:
```bash
python scripts/train_mamba.py --device cuda --k 64 --epochs 50 --batch_size 16
```

### 3. Run the full sweep interactively
Run the k-sweep script directly (sweeps k = 8, 16, 32, 64, 96, 128):
```bash
bash scripts/run_mamba_k_sweep.sh
```

### 4. Submit the sweep on Skynet (Slurm)
Submit the job to the cluster. Make sure to update the repo path and virtual environment path in `scripts/slurm_train_mamba_k_sweep.sh` before running!
```bash
sbatch scripts/slurm_train_mamba_k_sweep.sh
```

### 5. Monitor and Manage Skynet Job
Check queue:
```bash
squeue -u $USER
```

Watch logs (replace `<JOBID>` with the actual job ID):
```bash
tail -f logs/mamba_k_sweep_<JOBID>.out
```

Cancel job:
```bash
scancel <JOBID>
```

### Results & Checkpoints
- Results (metrics and configs) are saved and appended to: `results/mamba/k_sweep/mamba_k_sweep_results.csv`
- Model checkpoints are saved separately by `k` at: `results/mamba/k_sweep/checkpoints/best_mamba_k_<k>.pth`

To download the results back to your local computer, run from your **local machine**:
```bash
scp -r skynet:~/City-Scale-AI/results/mamba/k_sweep ./local_results
```