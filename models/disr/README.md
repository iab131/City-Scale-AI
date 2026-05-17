# DiSR-Mamba: Directed Spectral Residual Mamba

A residual branch trained on top of a frozen STAEformer trunk for METR-LA
traffic forecasting. The branch predicts a structured correction Δ_Y so the
final forecast is

```
Y_hat = Y_STAEformer + α · Δ_Y
```

## Modules

| File | What it does |
|---|---|
| `losses.py` | Masked MAE/RMSE/MAPE, congestion mask, composite L_main + λ_res L_res + λ_cong L_cong. |
| `spectral_basis.py` | Symmetric normalized Laplacian eigendecomposition + projection helpers. |
| `magnetic_laplacian.py` | Hermitian magnetic Laplacian L_q = D_s − A_s ⊙ exp(i Θ_q), eigendecomposition, complex projection/unprojection. Also infers a directed adjacency from a symmetric one via lagged cross-correlation. |
| `biaxis_mamba.py` | Bi-axis (temporal + mode) Mamba block. Real-valued; complex spectra are split into (real, imag) channels. Falls back to bidirectional GRU on CPU. |
| `residual_router.py` | Low-capacity router over (horizon, sensor-cluster, calendar, recent context) → per-expert weights + α scale. Also builds the spectral-clustering cluster assignment. |
| `disr_mamba.py` | Composite model orchestrating the temporal, sym-spectral, mag-spectral, and learned-basis experts. |
| `staeformer_wrapper.py` | Frozen-trunk wrapper around `models.staeformer.STAEformer` with optional partial-unfreeze for Stage F. |

## Switches (config flags)

The model is fully assembled from `cfg.model.*` flags. Each may be toggled
independently:

```yaml
model:
  use_temporal_residual: true
  use_symmetric_spectral: false
  use_magnetic_spectral: false
  use_learned_basis: false
  use_horizon_cluster_router: false
  biaxis_mode_axis: true
  k_modes: 48
  q_charge: 0.10
  spectral_side: "low"
  d_model: 64
  n_layers: 2
  alpha_init: 0.10
  alpha_max: 1.5
```

## Tensor conventions

All shapes follow the spec:

| name | shape | space |
|---|---|---|
| `x_norm` | `[B, T_in, N]` | normalized speed |
| `tod` | `[B, T_in]` | time-of-day ∈ [0, 1) |
| `dow` | `[B, T_in]` | day-of-week ∈ {0..6} |
| `y_node` | `[B, T_out, N]` | raw mph |
| `y_mask` | `[B, T_out, N]` | 1 = valid |
| `Y_base_norm` | `[B, T_out, N]` | STAEformer output (normalized) |
| `Delta_Y_norm` | `[B, T_out, N]` | residual (normalized) |
| `Y_pred = (Y_base_norm + α · Delta_Y_norm) · std + mean` | `[B, T_out, N]` | raw mph |

## Caches

Spectral bases and clusters are cached under `cache/gft/disr/`:

```
sym_k{K}_low.npz                 symmetric Laplacian, K modes
mag_k{K}_q{q:.2f}_low.npz        magnetic Laplacian, K modes, charge q
clusters_n{N}_v1.npy             sensor cluster assignment
```

Run `python scripts/disr/prepare_bases.py` once to populate them.

## Running

```bash
# 1. Train STAEformer trunk (paper-faithful).
python scripts/train_staeformer.py --tag stae_trunk --seed 42

# 2. Pre-compute spectral bases and cluster assignments.
python scripts/disr/prepare_bases.py

# 3. Run a stage.
python scripts/disr/train_disr.py \
    --config configs/disr/stage_d_magspec.yaml \
    --trunk_ckpt results/staeformer/stae_trunk/best_stae_s42.pth \
    --seed 0

# 4. Aggregate all results.
python scripts/disr/aggregate_results.py
python scripts/disr/make_plots.py

# 5. Final ensemble + ST-TTC.
python scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/stageE_router_*_s*/best_disr.pth" --use_ttc
```

## Hardware notes

- bf16 AMP on the H200 H200 (`torch.cuda.is_bf16_supported() == True`).
- `mamba_ssm`'s `selective_scan_cuda` requires inputs on CUDA. The
  `_make_mamba` helper falls back to a bidirectional GRU on CPU so unit
  tests run anywhere.
- `causal_conv1d` is optional; without it Mamba falls back to `nn.Conv1d`
  for the conv pre-step (~10–20% slower, no quality impact).

## Tests

```bash
PYTHONPATH=. python -m pytest tests/disr -q
```

Covers: Hermitian property, eigenshape, projection roundtrip, real/imag
split parity, module output shape, masked-MAE correctness, bf16 AMP step,
sensor-cluster sanity.
