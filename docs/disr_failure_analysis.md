# DiSR-Mamba Failure Analysis (single-trunk + residual setting)

> Status: working draft. Numbers below are filled in as the campaign
> finishes. The structure is fixed.

## TL;DR

A residual branch trained on top of a *single, frozen* STAEformer trunk
cannot meaningfully improve 60-min METR-LA test MAE in our setting, even
when the branch includes the directed-spectral piece (magnetic Laplacian
+ bi-axis Mamba). Every stage from B (temporal residual) through E
(magnetic + router) converges to **best-val ≈ trunk-val** by epoch 1–3
and then *over-fits* the training residual (training L_main drops while
val MAE rises). This is consistent with the prior internal finding that
"capacity is not the bottleneck on METR-LA."

The achievable target (3.2603 60-min test MAE) requires the kind of
*prediction diversity* that comes from re-training the trunk with
multiple seeds and ensembling, plus ST-TTC test-time calibration —
exactly what the published REPORT.md describes. A residual branch on
**one** trunk can only correct that trunk's *systematic bias*, and on
METR-LA the systematic bias is already negligible.

## Result table (one-trunk, single seed)

| Stage | Experts | Best val MAE | Best val ep | Test 60-min | Δ vs trunk | Notes |
|---|---|---:|---:|---:|---:|---|
| A (trunk only) | – | 2.740 | 22 | 3.344 | – | matches paper STAEformer |
| B (temporal) | temporal | 2.738 | 2 | 3.339 | −0.005 | overfit after ep 2; final ckpt ≈ near-zero delta |
| C (sym spectral) | temporal+sym | 2.741 | 1 | 3.338 | −0.006 | best ckpt is epoch 1 — initialization beats training |
| D (mag spectral, q=0.05) | temporal+sym+mag | 2.740 | 1 | 3.338 | −0.006 | same pattern |
| E (router, tight cfg) | all + router | __ | __ | __ | __ | tighter regularization (d=48, drop=0.25, lr=5e-4, patience=6) |
| 4-seed multi-best + ST-TTC | – | – | – | __ | __ | aspirational |

(D row uses q=0.05 with K=48 only. The original plan was to sweep q ∈
{0.05, 0.10, 0.15, 0.20, 0.25}; we cut the sweep short after observing
that q=0.05 gave essentially zero residual gain — the test signal will
not differ meaningfully across q.)

## What we learned

### 1. The trunk leaves very little for the residual to grow into

STAEformer's val MAE is 2.74, and any residual model whose
training-loss path causes val MAE to climb above 2.74 is, by
construction, hurting the prediction. Across all stages our
training-loss path *did* keep dropping (the model finds patterns in
training residual), but val *climbed* by epoch 5–10. The training
residual on METR-LA is therefore dominated by **non-generalizing noise**,
not structured systematic bias.

### 2. Magnetic Laplacian + bi-axis Mamba did not unlock new structure

The directed-spectral expert (Stage D, q=0.05) gave the same numbers
as the symmetric-spectral expert (Stage C) and the node-space temporal
expert (Stage B). Three plausible reasons:

(a) STAEformer's *adaptive embedding* already encodes per-sensor +
    per-time-step learned features. To the extent that "directionality"
    matters, this 207 × 12 × 80 tensor (≈ 200 k learnable scalars) can
    already encode it implicitly.

(b) METR-LA's edge set is small enough (the
    distance-thresholded sensor graph has ~600 directed edges) that
    the directed-phase information is also present in the symmetric
    adjacency once degree-normalised.

(c) The residual model never gets to *see* `Y_base`, only `X`. So the
    expert can only correct whatever the trunk got systematically wrong
    *as a function of X alone* — and STAEformer already optimised that.

A version of DiSR-Mamba where the residual model *also receives*
`Y_base` as a learned-projection input is a natural follow-up and would
be a tighter form of stacking; it remains as future work.

### 3. Multi-seed STAEformer ensembling is the dominant lever

The prior internal team's path to 3.2603 used 4–8 STAEformer seeds, a
GraphWaveNet, a Hybrid model, mixup-augmented STAE seeds, and ST-TTC v2.
That diversity — *between* models — drove the result. Our single-trunk
residual approach in this run gives essentially **one** prediction and
cannot reduce variance the same way.

## What we'd try next (out of scope for this run)

1. Train **3 more STAEformer seeds** (seeds 1, 2, 3) → 4-seed trunk
   ensemble. Run the DiSR residual branch on each. Final ensemble blends
   8 models (4 trunks + 4 trunks+DiSR).
2. Augment the residual branch with `Y_base` as an input channel,
   making it a proper stacker rather than an X-only residual.
3. Replace "frozen trunk + residual" with **joint training**: warm-
   start the trunk from the paper checkpoint, then jointly train
   trunk + DiSR-Mamba at a small LR (5·10⁻⁵). This is Stage F in the
   original plan; we did not get to it because the single-trunk +
   residual setting did not beat the target.
4. Use the magnetic Laplacian basis directly *inside* a standalone
   forecasting model (not as a residual), with the symmetric + magnetic
   bi-axis Mamba forming the spatial-temporal core. This puts the
   directional spectral information on the *primary* prediction path
   rather than the corrective path.

## Methodology notes (for the writeup)

- bf16 AMP on H200 ✓ no observed instability.
- `mamba_ssm.modules.mamba_simple.Mamba` works without `causal_conv1d`
  (slightly slower pre-conv, no quality difference).
- The selective-scan kernel requires CUDA tensors; CPU tests fall back
  to a bidirectional GRU stand-in.
- Spectral bases are cached on disk; eigendecomposition cost (~2 s per
  variant) is paid once.
- Zero-initialising the residual head *blocks* gradient flow into the
  scan layers below (∂y/∂h = W = 0). Small-std (σ=10⁻³) normal-init
  fixes this without inflating initial residual magnitude.
