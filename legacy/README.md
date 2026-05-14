# Legacy Code

This directory contains code from **before our work** plus exploratory scripts that did **not** make it into the final pipeline. We preserve it here so the original work (and our negative results) is recoverable.

**Nothing here is imported by the active pipeline.** If you delete this directory, `scripts/train_staeformer.py`, `scripts/eval_stae_ensemble.py`, and every other active script still works.

## What lives here

### `src/` — original project's data + model code
The pre-our-work codebase used the **Graph Fourier Transform + GRU** pipeline (`SpectralGRU`).
- `data_utils.py`, `graph_utils.py`, `gft.py` — were originally here; we moved them up to `src/` because the active pipeline still uses them. They are no longer in this folder.
- `preprocess.py` — old preprocessing (unmasked normalization). Superseded by `src/preprocess_v2.py`.
- `dataset.py` — old sliding-window dataset (spectral-space targets). Superseded by `src/dataset_v2.py`.
- `model.py` — `SpectralGRU` baseline.
- `train.py` — original training loop for `SpectralGRU`.
- `run_experiment.py` — original k-sweep driver.
- `plot_results.py`, `recover_results.py` — original plotting/result-restoration scripts.

### `models/` — original models
- `mamba_model.py` — original `SpectralMambaReal` (the project's first Mamba attempt, achieved 60-min test MAE 4.18). Superseded by our `models/spectral_ssm.py` and ultimately by `models/staeformer.py`.
- `fallback_mamba.py` — CPU-only fallback that was never wired in.

### `scripts/` — historical / exploratory training & eval scripts
- `train_mamba.py` — trains the original `SpectralMambaReal`.
- `run_mamba_k_sweep.sh`, `slurm_train_mamba_k_sweep.sh` — Skynet cluster scripts.
- `train_sssm_v7.py` — older script for the SSSM v7 multi-window variant. Superseded by `scripts/train_sssm.py` with `--version v7`.
- `eval_multi_ensemble.py` — earlier multi-arch ensemble eval. Superseded by `scripts/eval_full_ensemble.py`.
- `run_stae_seeds.sh` — superseded by `run_stae_seeds_v2.sh`.
- `run_v4_seeds.sh`, `sweep_sssm.sh` — exploration during the SSSM v1-v8 phase.
- `eval_stgormer.py`, `prep_stgormer_data.py` — attempted reproduction of STGormer (arXiv 2408.10822). Came in at 60-min MAE 3.58 vs paper's 3.10. Kept for documentation of the negative result.

### `scratch/`
- `scratch_h5.py`, `scratch_pkl.py` — quick inspection scripts the original team used to understand the dataset layout.

### Result artifacts (preserved from earlier runs)
- `checkpoints_gru/` — 11 `SpectralGRU` checkpoints (one per k in 1..207). Used by `src/train.py` to load best model after training.
- `outputs/` — original team's k-sweep result CSVs, plots, and markdown summaries.
- `mamba/` — original `SpectralMambaReal` k-sweep checkpoints + result CSV.
- `results_mamba/` — duplicate set of `SpectralMambaReal` checkpoints + results (the original team kept both `mamba/` and `results/mamba/`).
- `city-scale-ai-clean.tar.gz` — early 30 MB snapshot tarball of the project. Kept for archival.

## How to run the legacy code

Each legacy script expects the original project's directory layout. To run them, work inside `legacy/`:
```bash
cd legacy/
python src/train.py    # original SpectralGRU baseline
python scripts/train_mamba.py --device cuda --k 64 --epochs 50  # SpectralMambaReal
```

For the active pipeline, see the top-level `README.md`.
