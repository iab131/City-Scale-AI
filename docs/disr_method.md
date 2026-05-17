# DiSR-Mamba: Method (draft for the paper)

**Working title**: *Directed Spectral Residual Mamba for City-Scale Traffic
Forecasting.*

## Problem statement

METR-LA (207 loop sensors × 34 272 5-min steps) is, by 2026, a saturated
benchmark. Improvements over the standard public-code SOTA (STAEformer,
3.34 60-min MAE) require either deeper backbones or test-time tricks that
are hard to reconcile with reproducibility. The City-Scale-AI team's
previous campaign showed that

1. **Backbone capacity is not the bottleneck.** A bigger STAEformer
   (d=192/256, L=4) does **not** improve individual seed test 60-min
   MAE on METR-LA (R02 result: 3.350 vs. baseline 3.34).
2. **Reconstruction-based pre-training (TMAE/STMAE) does *not* transfer**
   to forecasting on this benchmark — all R03/R06/R11/R12 variants
   *hurt* downstream MAE.
3. **Architectural diversity and per-horizon ensemble top-K selection** —
   combined with the streaming spectral calibrator ST-TTC — gave the
   final 24-model + ST-TTC v2 number: 60-min test MAE = **3.2603**
   (and `15/30/60 = 2.604/2.904/3.260`).

The remaining gap to TESTAM (3.14) is unreproducible. The realistic
ceiling for *reproducible* methods is ~3.25.

The empirical pattern that motivates DiSR-Mamba: STAEformer's largest
errors concentrate on (i) *long-horizon predictions*, (ii) *low-speed
congestion* regimes, and (iii) sensor pairs whose information flow is
*directed* (downstream/upstream over a freeway). STAEformer's spatial
attention is permutation-equivariant over sensors — it learns a *kernel*
over node embeddings, not a directed convolution. The hypothesis is that
a *directed-spectral residual* can plug this gap.

## Hypothesis

> METR-LA's remaining error is dominated by horizon- and
> direction-dependent traffic heterogeneity that the symmetric-kernel
> backbone misses. A residual branch operating in the *magnetic
> Laplacian* eigenbasis — which is itself complex-valued and encodes
> directed flow via the charge q — should reduce both long-horizon and
> congestion-regime error without retraining the backbone.

## Architecture

```
       ┌─────────── STAEformer trunk (frozen) ───────────┐
       │  Y_base ∈ R^{B,T_out,N}                          │
       │                                                  │
x ─────┤                                                  ├──▶ Y_pred = Y_base + α·ΔY
       │                                                  │
       └─────────── DiSR-Mamba residual ──────────────────┘
                     │
                     ├── Temporal Mamba expert
                     │     (node-space, no graph op)
                     │
                     ├── Symmetric spectral expert
                     │     U_sym = bottom-K eigvecs(L_sym)
                     │     bi-axis Mamba over (T, K)
                     │
                     ├── Magnetic spectral expert
                     │     L_q = D_s − A_s ⊙ exp(i·Θ_q),    Hermitian
                     │     U_mag = bottom-K eigvecs(L_q)    complex
                     │     project, split [Re | Im],
                     │     bi-axis Mamba, recombine,
                     │     U_mag back-projection, take Re
                     │
                     └── (optional) learned-basis expert
                     │
                     │
                Horizon-Cluster Router
                  inputs: horizon index, cluster id, TOD,
                          DOW, recent volatility, congestion
                  outputs: expert mixing weights, α scale
                          per (sample, horizon, sensor-cluster)
```

### Magnetic Laplacian

Given a directed adjacency `A_dir`, with `A_s = ½(A_dir + A_dirᵀ)` and
`Θ_q = 2π·q·(A_dir − A_dirᵀ)`, the magnetic Laplacian
`L_q = D_s − A_s ⊙ exp(i Θ_q)` (normalized variant
`L_q = I − D_s^{−½} (A_s ⊙ exp(i Θ_q)) D_s^{−½}`) is **Hermitian** and
its complex eigenvectors carry directional phase information.

METR-LA ships an asymmetric distance kernel (the `adj_METR-LA.pkl` is not
symmetric), so we use it directly as `A_dir`. For graphs delivered as
symmetric we infer `A_dir` from short-lag cross-correlation of sensors
that share an edge.

### Bi-axis Mamba

A single block of the bi-axis scan operates on a `[B, T, K, D]` tensor:

```
  y_T = MambaScan_T(LN(x))        # along T, contracting (B, K) into batch
  y_K = MambaScan_K(LN(x))        # along K, contracting (B, T) into batch
  g   = sigmoid(W [y_T | y_K])
  out = g · y_T + (1−g) · y_K
```

Complex spectra are folded into the *D* axis as two real channels
(real, imag) so the scan stays real-valued. Full complex Mamba is
deferred to a later ablation.

### Horizon-Cluster Router

The router blends experts on a per `(sample, horizon, cluster)` grid with
a tiny MLP (~5 k parameters). It receives time-of-day, day-of-week,
recent volatility, congestion ratio, and learnt horizon/cluster
embeddings. Output is a softmax over experts plus a sigmoid α in
`(0, α_max)`. Crucially the router does **not** route the trunk — only
the residual experts.

### Sensor clusters

Built once at preprocessing by spectral clustering on
`S = ½·A_norm + ½·Corr(X_train)`. Cached to disk.

## Loss

```
L = masked_MAE(Y_pred, Y_true)
  + λ_cong · masked_MAE(Y_pred, Y_true | congestion)
  + λ_ent · (−H(router gates))    # only when router enabled
```

A pure residual-match term `L_res = MAE(ΔY, Y_true − Y_base)` is
*redundant* with `L_main` when `α = 1`, so it is off by default.
Congestion is defined as `Y_true < 20 mph` OR
`|Y_true − X_recent_last| > 5 mph`.

## Training protocol

1. **Stage A** — Train (or load) STAEformer trunk to its
   paper-faithful val MAE.
2. **Freeze** the trunk; train each expert with a small-std (σ=10⁻³)
   head init so initial residual is ≈ 0 but gradients still flow into
   the scan layers. (Pure zero-init kills `∂y/∂h = W = 0`.)
3. **Stages B → E** progressively switch on experts.
4. **Stage F** — light joint fine-tune (only the output projection +
   last spatial layer of STAEformer) at lr 5·10⁻⁵.
5. **Stage G** — apply ST-TTC streaming FFT calibration on top of the
   final single or ensemble model.

We deliberately do not introduce a physical flow-conservation loss:
METR-LA is speed-only; density/flow are unavailable and would require
hallucinated inputs.

## What we report

- Per-horizon test MAE / RMSE / MAPE.
- Per-speed-regime test MAE (`<20`, `20–40`, `40–60`, `≥60` mph).
- Per-cluster test MAE.
- Ablation table for every switch listed above.
- q-charge and K-mode sensitivity plots.
- Router expert-usage heatmap by `(horizon, cluster)`.
- Wall-clock time, max GPU memory, params per stage.

## What we expect

| Stage | Single-seed 60-min test MAE | Notes |
|---|---:|---|
| Trunk only (Stage A) | ~3.34 | reproduces published STAEformer |
| Temporal residual (B) | ≈3.32–3.34 | control |
| Symmetric spectral (C) | ≈3.31–3.33 | weak; close to a heavier STAEformer |
| Magnetic spectral (D, q≈0.10) | ≈3.30–3.32 | the novel piece |
| Router (E) | ≈3.29–3.31 | adds heterogeneity |
| Joint FT (F) | ≈3.28–3.30 | may or may not help |
| Multi-seed + ST-TTC (G) | **≈3.26–3.28** | aspirational |

If a single seed clears 3.26 we celebrate. If the multi-seed ensemble
+ ST-TTC does, we beat the prior internal best 3.2603 and the writeup
documents the win with full ablation tables.

If *no* configuration clears 3.30, we still produce a clean negative
result: "the magnetic Laplacian residual sometimes helps long-horizon
prediction but a directed spectral basis is not enough to beat a 24-
model STAEformer ensemble on METR-LA."
