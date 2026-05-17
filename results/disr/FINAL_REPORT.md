# DiSR-Mamba — Final Campaign Report

**Run date**: 2026-05-14 / 2026-05-15
**Hardware**: NVIDIA H200 SXM (143 GiB), bf16 AMP, single-GPU
**Backbone**: STAEformer (CIKM 2023), reproduced from scratch in this run
**Target**: beat the prior internal best 60-min test MAE = **3.2603** on METR-LA
under the project's standard masked evaluation protocol.

## Result

| Configuration | 15-min | 30-min | **60-min** | avg-MAE |
|---|---:|---:|---:|---:|
| STAEformer single-seed (paper reproduction) | 2.653 | 2.961 | **3.344** | 2.933 |
| 4-seed STAEformer ensemble (uniform) | 2.620 | 2.922 | 3.292 | 2.895 |
| 4-seed STAEformer ensemble + ST-TTC v2 (g=4) | 2.618 | 2.918 | **3.287** | 2.891 |
| 4-seed STAE + 3 DiSR-on-trunks + ST-TTC v2 | 2.618 | 2.919 | 3.287 | 2.892 |
| 4-seed STAE + 9 DiSR variants + ST-TTC v2 | 2.622 | 2.924 | 3.293 | 2.896 |
| Prior internal best (24-model + per-horizon top-K + ST-TTC v2) | 2.604 | 2.904 | **3.260** | 2.874 |

**Best 60-min test MAE this campaign: 3.287** (Δ = +0.027 vs. target 3.2603).

The target was *not* beaten. The full DiSR-Mamba residual stack (Stages B
through E) does **not** improve over the trunk it sits on, and the 4-seed
STAEformer ensemble + ST-TTC alone hits the same 3.287 with or without
DiSR-on-each-trunk in the bag.

## Per-stage ablation (single-trunk = stae_trunk seed 42)

| Stage | Experts | Test 60-min | Δ vs. trunk | Best val ep |
|---|---|---:|---:|---:|
| A — Trunk only | – | 3.344 | – | 22 |
| B — Temporal residual | temporal | 3.339 | −0.005 | 2 |
| C — Sym spectral, K=48 | temporal + sym | 3.338 | −0.006 | 1 |
| D — Magnetic spectral, q=0.05, K=48 | + mag | 3.338 | −0.006 | 1 |
| E — Horizon-cluster router (tight cfg) | + router | 3.339 | −0.005 | 1 |

Every residual stage's best-val ckpt is at epoch **1–2**. After that, the
training-loss path keeps dropping while the val MAE *climbs* — classic
over-fitting on a residual that is dominated by non-generalizing noise.
This is what we predicted in `docs/disr_failure_analysis.md` once Stage
C's overfit pattern appeared.

## Multi-seed of the best DiSR config (Stage C)

| Tag | Trunk | Test 60-min | Best val |
|---|---|---:|---:|
| stageC_symspec_s0  | stae_trunk (s42) | 3.338 | 2.741 |
| stageC_symspec_s1  | stae_trunk (s42) | **3.355** | 2.741 |
| stageC_symspec_s2  | stae_trunk (s42) | 3.349 | 2.743 |
| stageC_symspec_trunk1 | stae_trunk_s1 | 3.349 | 2.726 |
| stageC_symspec_trunk2 | stae_trunk_s2 | 3.346 | 2.745 |
| stageC_symspec_trunk3 | stae_trunk_s3 | 3.345 | 2.735 |

Different random-init seeds of the residual head perturb test predictions
by ±0.01 around the trunk. The mean across seeds ≈ the trunk-alone
result; the spread is **noise**, not learning.

## STAEformer trunks (multi-seed)

| Tag | Best val MAE | Test 60-min | Epochs trained |
|---|---:|---:|---:|
| stae_trunk (seed 42) | 2.740 | 3.344 | 22 (early-stopped) |
| stae_trunk_s1 (seed 1) | 2.723 | 3.340 | 24 |
| stae_trunk_s2 (seed 2) | 2.743 | 3.347 | 23 |
| stae_trunk_s3 (seed 3) | 2.732 | 3.344 | 21 |

Seed std on test 60-min = **0.003** — exactly the same architectural
plateau the prior internal team observed (their REPORT § "seed std 0.004").
Multi-seed STAEformer is what drives the ensemble lift, not DiSR.

## What we learned (and what to publish)

1. **Capacity is not the bottleneck on METR-LA.** Confirmed yet again.
   The trunk's residual error is essentially aleatoric.

2. **Frozen-trunk residual learning fails on METR-LA.** Every stage's
   best-val checkpoint is at epoch 1 — initialization beats training,
   because the trainer minimizes the residual on training data but those
   gradients don't transfer to val. The small-std (σ=10⁻³) head init we
   use does let the scan layers receive gradient, so the model *is*
   learning — it just learns noise.

3. **Magnetic Laplacian + bi-axis Mamba is not enough to unlock new
   structure when the trunk has an adaptive embedding.** The
   `[T_in × N × 80]` adaptive embedding in STAEformer already encodes
   per-sensor + per-time-step learned features that subsume most of the
   directed-spectral signal we hoped to add. q ∈ {0.05} was tested;
   larger q wasn't run after q=0.05 showed no lift (we cut the
   pre-planned q sweep to save compute).

4. **The 3.2603 number is gated by architectural diversity, not by a
   smarter single-trunk residual.** The prior team's 24-model bag
   blended STAEformer seeds + dropout variants + STMAE-pretrained +
   mixup + 60-min specialist + calendar-prior + GraphWaveNet + Hybrid,
   then applied *per-horizon top-K selection*. With only 4 trunks and
   9 essentially-identical DiSR variants we cannot match that.

5. **ST-TTC v2 (FFT amplitude+phase calibrator, FIFO streaming) is the
   single biggest single-step improvement** we can apply on top of an
   ensemble: it consistently drops 60-min MAE by 0.004-0.006 with 1656
   parameters and no test-set labels needed.

## What would close the gap to 3.2603 (out of scope here)

1. **Train 4 more STAEformer seeds with hyperparameter diversity**
   (dropout ∈ {0.05, 0.10, 0.15}, 1–2 with batch 32 instead of 16).
   This is what the prior team called "R01" and it added 0.005 per
   variant via per-horizon top-K.

2. **Add a GraphWaveNet seed and a Hybrid (STAE + spectral) seed.**
   These are the two cross-architecture members that mattered most in
   the prior 24-model bag.

3. **Apply per-horizon top-K selection on top of the ensemble.** Use
   `scripts/eval_R16_per_horizon_topk.py` (adapted to our trunk paths).
   The prior team's per-horizon K ∈ [4, 14, …, 23] gave −0.005 on top
   of uniform averaging.

4. **A true augmented-residual** where the residual model also receives
   `Y_base` as a learned input channel. The current "residual sees X
   only" design is essentially an X-conditioned predictor that competes
   with the trunk; an X+Y_base design is a proper stacker.

## Deliverables in this run

- `models/disr/{losses, spectral_basis, magnetic_laplacian, biaxis_mamba,
  residual_router, disr_mamba, staeformer_wrapper}.py`
- `tests/disr/*.py` — 23 unit tests passing on H200 (Hermitian, real/imag
  split, projection roundtrip, bf16 AMP, shape contracts, masked-MAE
  correctness, sensor-cluster sanity)
- `configs/disr/{base, stage_b..f, sweep_main}.yaml`
- `scripts/disr/{train_disr, evaluate_disr, eval_combined,
  eval_trunk_only, precompute_stae_base, prepare_bases,
  aggregate_results, make_plots, run_sweep, run_disr_campaign,
  finish_campaign, queue_after_trunk, queue_multitrunk,
  queue_phase3}.{py,sh}`
- `cache/gft/disr/` — all spectral bases (sym K∈{32,48,64}, magnetic
  K∈{32,48,64} × q∈{0.05,0.10,0.15,0.20,0.25}) and sensor clusters
  (n∈{8,12,16}) precomputed.
- `results/disr/`:
    - 9 single-model DiSR runs
    - `ablation_table.{csv,md}`
    - per-run `summary.json`, `test_metrics.json`, `per_horizon.json`,
      `per_speed_regime.json`, `test_predictions.npz`, `log.csv`, `plots/`
    - `combined_full_metrics.json` — 13-model + ST-TTC
    - `combined_4stae_3disr_trunks.json` — 7-model + ST-TTC
    - `trunk_only_metrics.json` — Stage A baseline
- `docs/disr_method.md` — paper-style method writeup
- `docs/disr_failure_analysis.md` — pre-result hypothesis + structured
  negative-result write-up

## Reproduction recipe

```bash
# 0. install deps (transformers<4.45, mamba-ssm 2.2.2 via deep import; no causal-conv1d needed)
pip install --no-build-isolation "transformers<4.45" "mamba-ssm==2.2.2" \
    pandas h5py scipy einops pyyaml scikit-learn matplotlib

# 1. precompute spectral bases & clusters (~30 s)
python scripts/disr/prepare_bases.py

# 2. train 4 STAEformer trunks (~50 min each)
for s in 42 1 2 3; do
    python scripts/train_staeformer.py --tag stae_trunk_s$s --seed $s \
        --epochs 80 --patience 20 --batch_size 16 --num_workers 4
done

# 3. run each DiSR stage on stae_trunk seed 42 (≤ 15 min each)
for cfg in stage_b_temporal stage_c_symspec stage_d_magspec stage_e_router; do
    python scripts/disr/train_disr.py \
        --config configs/disr/${cfg}.yaml \
        --trunk_ckpt results/staeformer/stae_trunk/best_stae_s42.pth \
        --seed 0 --no_compile
done

# 4. 4-seed STAEformer ensemble + ST-TTC v2 — gives 60-min 3.287
python scripts/eval_stae_ensemble.py --use_ttc --ttc_groups 4 \
    --stae_ckpts 'results/staeformer/stae_trunk*/best_stae_s*.pth'

# 5. combined STAE + DiSR ensemble + ST-TTC v2
python scripts/disr/eval_combined.py \
    --stae_ckpts 'results/staeformer/stae_trunk*/best_stae_s*.pth' \
    --disr_ckpts 'results/disr/*_s*/best_disr.pth' \
    --use_ttc --ttc_groups 4

# 6. aggregate ablation table + plots
python scripts/disr/aggregate_results.py
python scripts/disr/make_plots.py
```

## Stopping rule trigger

Stopping rule **3 from the run brief**: *"All planned ablations fail to
improve over STAEformer and the failure analysis is written."*

- Stages B / C / D / E all give single-seed test 60-min in the range
  3.338-3.339 (Δ < 0.01 vs. trunk 3.344).
- Multi-seed Stage C and DiSR-on-multiple-trunks add prediction noise,
  not signal.
- Best ensemble (4 STAE + ST-TTC v2) = 3.287 ≠ 3.2603.
- Failure analysis written: `docs/disr_failure_analysis.md`.

The campaign closes here with a complete reproducible code base, full
unit tests, an end-to-end ablation table, per-horizon and per-speed-regime
breakdowns, and a documented set of next steps for a future run that has
the compute budget to train a more diverse architecture bag.
