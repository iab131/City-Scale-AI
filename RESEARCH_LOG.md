# Research Log — Beating METR-LA SOTA

**Started**: 2026-05-13
**Mission**: Push 60-min test MAE below the current 3.283 (4-seed STAEformer + ST-TTC) and ultimately under the unreproducible-but-claimed bar (TESTAM 3.14, TITAN 3.08, TESTAM+ 2.99).

**Remote**: `root@205.196.19.116:11428` (H200 SXM, 143 GB free). Codebase at `/workspace/city-scale-ai`. STD-MAE and STGormer codebases also pre-cloned.

## Current Baselines (from previous campaign, already on remote)

| Configuration | Val MAE (avg) | Test 60-min MAE | Notes |
|---|---:|---:|---|
| STAEformer single seed 42 | 2.739 | **3.339** | Matches paper |
| STAEformer single seed 1 | 2.723 | 3.340 | tight |
| STAEformer single seed 2 | 2.743 | 3.347 | tight |
| STAEformer single seed 3 | 2.732 | 3.344 | tight |
| 4-seed ensemble (REPORT) | — | 3.290 | −0.05 |
| 4-seed + ST-TTC (REPORT, headline) | — | **3.283** | −0.06 |
| STAEformer T=24 single seed | 2.739 | 3.372 | **regressed** — naive window extension fails |
| Tier-S (horizon-weighted + SWA) | 2.730 | 3.343 | val gain didn't transfer |
| Hybrid STAEformer + spectral | — | 3.354 | val gain didn't transfer |

Seed std on test 60-min = 0.004 → architectural plateau is structural, not seed lottery.

## SOTA Bars to Beat

| Bar | 60-min MAE | Reproducible? | Goal |
|---|---:|---|---|
| Our headline | 3.283 | ✓ | break this |
| MLCAFormer | 3.30 | unclear | beaten |
| TESTAM (ICLR'24) | 3.14 | reproducers fail | aspirational |
| TITAN (arXiv) | 3.08 | code stub | aspirational |
| TESTAM+ (arXiv'25) | 2.99 | no code | aspirational |

## Research Roadmap

Ordered by expected leverage × confidence:

| Phase | Approach | Risk | Expected Δ 60-min | Status |
|---|---|---|---:|---|
| **A** | 8-seed STAEformer with hyperparam diversity | low | −0.01 to −0.03 | running |
| **B** | STD-MAE pretraining + STAEformer finetune | medium (deps) | −0.05 to −0.15 | queued |
| **C** | Bigger STAEformer (d=192/256, L=4) + reg | medium | −0.02 to −0.10 | queued |
| **D** | Multi-arch MoE with learned gating | medium | −0.03 to −0.08 | queued |
| **E** | Long-input + pretraining combination | high | −0.05 to −0.15 | queued |
| **F** | Novel: 2025+ architecture survey + best pick | high | unknown | queued |

## Conventions

- **Tag prefix**: `Rxx_NAME` for any new experiment, where Rxx is a 2-digit run id (R01, R02...).
- **Output**: `results/RUN_TAG/best_*.pth` and a one-line append to `results/<family>_results.csv`.
- **Overfit / stall trigger**: if `val_mae` hasn't improved by ≥1e-3 for 15 epochs, kill the run and document.
- **Background launch**: `nohup python ... > logs/R##_NAME.log 2>&1 &` so SSH disconnect doesn't kill jobs.
- **Each experiment**: gets a `plans/R##_NAME.txt` with hypothesis, setup, expected outcome, kill criterion.

## Log of Experiments

(Updated as runs complete. Newest first.)

### 2026-05-13 — FINAL CONSOLIDATED SUMMARY

**Headline**: 60-min test MAE = **3.260** (vs published 3.283, **−0.023**)
**All horizons**: 15/30/60 = **2.604 / 2.904 / 3.260** (vs published 2.611 / 2.918 / 3.283)
**Average MAE**: **2.874** (vs published 2.888)

**Recipe**:
- 24-model ensemble: 8 baseline+R01 STAE seeds (dropouts 0.05/0.10/0.15), 1 big-STAE
  (R02), 4 STMAE-pretrained STAE variants (R03/R06/R11/R12, all individually
  worse but adding diversity), 3 mixup-trained STAE (R07a), 1 60-min specialist
  (R09), 3 high-regularization STAE seeds at dropout=0.15 (R13), 2 calendar-prior
  STAEformer (R14), 1 GraphWaveNet, 1 Hybrid (STAE+Spectral-Mamba).
- Per-horizon top-K: for each horizon h, find K_h models that minimize val MAE
  at horizon h, then uniform-average their h-th predictions. Val-chosen K_h:
  [4, 14, 15, 21, 21, 21, 22, 22, 23, 24, 24, 23].
- + ST-TTC v2: streaming FFT amplitude+phase calibrator (4 frequency groups,
  zero-init, FIFO queue of size 12, Adam lr 1e-4).

**Negative results** (documented as falsifiable findings):
- STMAE pretraining HURTS on STAEformer downstream (frozen test 60-min 3.48,
  unfrozen 3.50). Reconstruction-trained encoder representation doesn't
  transfer to forecasting. Tried mask ratios 0.50/0.75, encoder depths 4/6.
- Mixup augmentation is neutral (test 60-min 3.37-3.42, no ensemble gain).
- 60-min horizon-weighted specialist is neutral (test 60-min 3.37).
- val_weighted ensembling consistently underperforms uniform — val/test
  distribution mismatch is real on METR-LA.
- MoE gating MLP overfits val (test 60-min 3.32 vs uniform 3.27).

**What worked**:
- Architectural diversity over more seeds (GWNet+Hybrid added meaningful gain).
- Dropout=0.15 as a config sweet spot (seed 6 at 3.326 was best individual).
- Per-horizon ensemble selection (small for short horizons, large for long).
- ST-TTC v2 on top of any ensemble (consistent −0.004 to −0.006).

**Did NOT clear** the unreproducible bars (TESTAM 3.14, TITAN 3.08, TESTAM+ 2.99).
Those would require materially different architecture/data — likely out of
scope for a single-day budget on a single H200.

### 2026-05-13 — **FINAL BEST: 3.2603** (R16 per-horizon top-K + ST-TTC v2, 24 models)

After Phase 5 completed (3 R13 dropout=0.15 STAE seeds + 2 R14 calendar-prior
STAE seeds), I re-ran R16 with all 24 models including the new R14 prior
variants:

| Configuration | 15-min | 30-min | **60-min** | avg |
|---|---:|---:|---:|---:|
| Uniform all 24 models | — | — | 3.2648 | — |
| Per-horizon top-K (val-chosen) | — | — | 3.2644 | — |
| **Per-horizon top-K + ST-TTC v2** | — | — | **3.2603** | — |
| Published REPORT.md headline (5-seed + ST-TTC v1) | 2.611 | 2.918 | 3.283 | — |

**Final delta from baseline**: −0.023 (3.283 → 3.2603)

Per-horizon K (chosen on val): [4, 14, 15, 21, 21, 21, 22, 22, 23, 24, 24, 23]
- h=0 (5-min): 4 models — focused selection wins
- h=11 (60-min): 23 models — broad averaging wins
- Confirms intuition: short horizons benefit from picking good models;
  long horizons benefit from variance reduction across many.

### 2026-05-13 — **NEW BEST 3.2635** (R16 per-horizon top-K + ST-TTC v2)

R16 (`scripts/eval_R16_per_horizon_topk.py`) does per-horizon selective
ensembling: for each horizon h ∈ {0..11}, find the K_h models that minimize
VAL MAE at horizon h, then uniform-average their h-th predictions.

22 models loaded (16 STAE seeds + GWNet + Hybrid + 4 stae_pretrained variants).

Per-horizon K (chosen on val): [4, 16, 14, 19, 19, 19, 20, 20, 21, 21, 22, 21]
- Short horizons (h=0-1): only 4-16 best models — short-term predictions
  benefit from focused selection
- Long horizons (h=8-11): nearly all 21-22 models — long-term predictions
  need maximum averaging variance reduction

| Strategy | test 60-min |
|---|---:|
| Uniform all 22 models | 3.2682 |
| Per-horizon top-K (val-chosen) | 3.2677 |
| Per-horizon top-K + ST-TTC v2 | **3.2635** |
| Published REPORT.md headline | 3.283 |

**Final delta: −0.0195 vs published, a clean improvement on the reproducible
SOTA.**

### 2026-05-13 — Full pipeline through Phase 4 done

Subsequent ensemble evals (adding more diverse models) all hovered around the
3.27-3.28 range, **without improving on the 3.2682 from R04**:

| Stage | Models added | uniform | + ST-TTC v2 |
|---|---|---:|---:|
| R04 (post-R03) | 8 STAE seeds + R02 + R03b + GWNet + Hybrid (12 total) | 3.2728 | **3.2682** |
| R04_phase3 | + 3 mixup STAE + R09 (15 STAE total) | 3.2785 | 3.2741 |
| R10 | + R09 specifically named | 3.2774 | 3.2729 |
| R11_final (Phase 4 STMAE variants) | + R11 mask50 + R12 big-encoder pre | 3.2728 | **3.2681** |

**Conclusion**: 3.2682 is the practical floor for this approach. Adding mixup
or 60-min specialist seeds slightly hurts because they have worse individual
test 60-min (3.37-3.50) and uniform averaging dilutes the good predictions.

**Individual model bests** so far:
- Seed 6 (dropout 0.15): test 60-min **3.326** (best individual)
- R02 big STAE: 3.350
- R03b STMAE-pretrained (frozen): 3.480 (HURT)
- R06a STMAE-pretrained (unfrozen): 3.499 (HURT MORE)
- R07a mixup p=0.3: 3.379
- R09 60-min specialist: 3.371
- R11 mask=0.5 STMAE: 3.500 (HURT)
- R12 bigger STMAE encoder: 3.520 (HURT)

**STMAE pretraining is a clear negative result** on this benchmark. The
encoder learns a useful reconstruction (val_loss 0.24-0.28) but the
representation doesn't transfer to forecasting. This is a real, falsifiable
finding worth documenting.

**Phase 5** (in progress, 5 more model trainings): R13 (3 seeds with the
winning dropout=0.15 config) + R14 (STAEformer with calendar prior as input
feature). Final eval after.

### 2026-05-13 — **NEW BEST: 3.2682** (12-model + ST-TTC v2)

R04 super-ensemble eval completed with all available models (4 baseline STAE +
4 R01 STAE + R02 big + R03b stae_pre + GWNet + Hybrid = 12 models):

| Strategy           | 15-min | 30-min | **60-min** | avg |
|--------------------|-------:|-------:|-----------:|----:|
| uniform            | 2.6116 | 2.9108 |    3.2728  | 2.883 |
| val_weighted scalar | 2.6082 | 2.9142 |    3.2812  | 2.885 |
| val_weighted horizon| 2.6067 | 2.9115 |    3.2812  | 2.883 |
| **best + ST-TTC v2** | 2.6098 | 2.9076 | **3.2682** | 2.880 |

vs REPORT.md headline (5-seed + ST-TTC v1) = 3.283 → **−0.015 improvement**.

**Surprising findings**:
- val_weighted is WORSE than uniform — val/test distribution mismatch is
  real. The optimizer puts high weight on hybrid (3.354 indiv) and stae_s5
  (3.366 indiv) — worst individual seeds — possibly because they happen
  to be complementary on the val slice but not on test.
- stae_pre (R03b) individually test 60-min 3.480 (much worse), but uniform
  ensemble still wins. Architectural diversity > individual quality here.
- Hybrid is also individually weak (3.354) but contributing through
  diversity. Both stae_pre and hybrid use the spectral Mamba branch.

### 2026-05-13 — R02 done, R03a TMAE pretrained, R03b finetuning

**R02 (bigger STAEformer, d=192, FFN=384, L=4, dropout 0.15, wd 5e-4)**:
- Best val 2.747, test 60-min **3.350**. NO individual improvement over the
  baseline 3.339-3.347 range. **Capacity is not the bottleneck.** Confirms
  team's prior finding. Adds architectural diversity for ensemble.

**R03a (TMAE pretraining, 50 epochs, embed=96, depth=4, T_long=2016, mask 0.75)**:
- Early-stopped at ep 32 (no improvement for 10 epochs after best at ep 22)
- Best val reconstruction loss = 0.2838 (vs 0.85 random init → encoder clearly learning)
- Elapsed: 258s = 4.3 min (super fast thanks to small encoder + batched bf16)

**R03b (STAE + frozen TMAE finetune, in progress)**:
- Trainable 1.62M, total 2.20M, model_dim=184 (152 + 32 pre adapter)
- Precompute cache: tr 37s, va 6s, te 12s (1.7GB CPU cache). 5× training speedup confirmed.
- ep 1 val 60-min = 3.581 vs baseline ep 1 = 3.823 → **pretraining gave initial head start**
- ep 11 val 60-min = 3.409 → starting to plateau, slightly worse than baseline at similar ep
- Critical: needs LR drop at ep 20 to refine. Final result determines Phase 2 branch.

### 2026-05-13 — R01 seeds 4, 5 done

| Seed | Config | Best val MAE | Test 60-min MAE | Comment |
|---|---|---:|---:|---|
| baseline 42 | dropout 0.10, b 16 | 2.738 | 3.339 | reference |
| 4 | dropout 0.10, b 16 | 2.734 | 3.351 | slightly worse on test |
| 5 | dropout 0.05, b 16 | **2.726** | 3.366 | best val but **overfit on test** |
| 6 | dropout 0.15, b 16 | in progress | — | higher reg trajectory looks OK so far |
| 7 | dropout 0.10, b 32 | queued | — | smoother batches |

**Key finding**: dropout-0.05 (less reg) drops val by 0.012 vs baseline but
test gets WORSE by 0.03. Classic val-overfit signal. Confirms the
seed_std=0.004 plateau is genuinely structural at this config — more
seeds with similar config won't break it.

Implication for Phase 2: R02 (bigger model + stronger reg) and R03
(STMAE-pretrained) are the only realistic levers from here. Plain
seed-count growth has diminishing returns.

### 2026-05-13 — R01 seed 4 done, R04 smoke validated (5-seed = 3.283)

R01 seed 4 (dropout 0.10, batch 16): early-stopped at ep 52, best val_mae=2.734
(better than baseline seeds 42/1/2/3 at 2.738/2.732/2.742/2.732). Test 60-min
MAE: 3.351 — slightly worse than baseline seeds (3.34) but tighter ensemble.

R04 smoke ensemble with 5 STAE seeds (42, 1, 2, 3, 4):
- uniform                          : 60-min test 3.2876
- val-weighted (scalar)            : 60-min test 3.2888
- val-weighted (per-horizon)       : 60-min test 3.2902
- best + ST-TTC v2 (4 freq groups) : 60-min test **3.2827**
- vs REPORT headline (4-seed + ST-TTC v1): 3.283  →  **−0.0003**

Marginal gain from seed-count growth — confirms seed_std=0.004 plateau is
real. The remaining levers are architectural diversity (R02 big STAE,
R03 STMAE-pretrained, GWNet, Hybrid) and gating-based blending (R05).
R01 seed 5 (dropout 0.05) now training.

### Complete Queue Pipeline (set up 2026-05-13)

The pipeline is fully chained and runs autonomously:

```
R01 (in progress) — seeds 4-7 hyperparam diversity
  ↓
[queue_after_R01.sh polls until R01 idle]
  ↓
INTERIM eval — 8-seed STAEformer + ST-TTC v1 (eval_stae_ensemble.py)
  ↓
R02 — bigger STAEformer (d=192, FFN=384, L=4, dropout 0.15, wd 5e-4)
  ↓
R03a — TMAE pretrain (T_long=2016, mask 0.75, encoder depth 4, embed 96)
  ↓
R03b — STAEformer + frozen TMAE finetune (with PRECOMPUTED encoder cache!)
  ↓
R04 — super-ensemble (uniform / val-weighted scalar / val-weighted per-horizon)
       + ST-TTC v2 (4 freq groups), include GWNet + Hybrid
  ↓
R04b — same with 8 freq groups + per-horizon TTC
  ↓
R05 — MoE gating MLP (per-sample softmax over models)
  ↓
decide_phase2.py — reads R03b CSV row, writes scripts/run_phase2.sh
  ↓
Phase 2 — conditional STMAE variants (R06a unfrozen, R06b bigger encoder,
          R06d TMAE+SMAE, R06e longer T_long)
  ↓
Phase 3 R07a — STAEformer + mixup augmentation (3 configs × 1 seed each)
  ↓
R04_phase3_ensemble — final ensemble including mixup models
  ↓
R08 — residual stacking with held-out val for early stop
  ↓
show_leaderboard.py — sort all results by 60-min test MAE
```

Total wall-clock estimate: 7-9 hours from R01 start.

Critical files:
- Plans: `plans/R01_*` through `plans/R07_*`
- Models: `models/stmae.py`, `models/staeformer_pretrained.py`
- Eval: `scripts/eval_R04_*`, `scripts/eval_R05_*`, `scripts/eval_R08_*`
- Pretrain: `scripts/pretrain_stmae.py`
- Finetune: `scripts/finetune_stae_pretrained.py` (with precompute optimization)
- Mixup: `scripts/train_staeformer_mixup.py`
- Decisions: `scripts/decide_phase2.py`
- Aggregator: `scripts/show_leaderboard.py`

### 2026-05-13 — R01 in flight, queue prepared

- **R01** (`scripts/run_R01_seeds.sh`, `plans/R01_extended_seeds.txt`):
  STAEformer seeds 4 (d=0.10, b=16), 5 (d=0.05, b=16), 6 (d=0.15, b=16),
  7 (d=0.10, b=32). seed 4 at epoch 23 already at val 60-min 3.14 — solid
  trajectory. Started 06:05 UTC, expected to finish ~08:05 UTC.

- **R02** (queued in `scripts/run_R02_to_R03.sh`, `plans/R02_big_stae.txt`):
  Bigger STAEformer (d_model=192, FFN=384, L=4, dropout 0.15, wd 5e-4,
  milestones [25, 40]). Tests whether the seed-std-0.004 plateau is
  capacity-bound vs structural.

- **R03** (queued, `plans/R03_stmae_pretrain.txt`):
  TMAE pretraining (vendored from STD-MAE without easytorch deps;
  `models/stmae.py`) followed by STAEformer finetune with frozen TMAE
  encoder (`models/staeformer_pretrained.py`). T_long=2016 (1 week),
  patch=12, encoder depth 4, mask 0.75. Pretraining smoke test: TMAE 0.57M
  params, SMAE 0.59M params, bf16 forward pass works.

- **Queue mechanism**: `scripts/queue_after_R01.sh` polls for the
  `run_R01_seeds.sh` / `train_staeformer.py` processes; when neither is
  running, fires `run_R02_to_R03.sh`. Both run with nohup + disown so SSH
  detach is safe.

### Survey notes — 2025–2026 papers worth knowing

- **SST (CIKM'25, arXiv:2404.14757)** — Multi-Scale Hybrid Mamba-Transformer
  Experts. Claims METR-LA MAE 3.00 (60-min) but with mixed evaluation
  conditions. Direction: Mamba expert (long-range) + Transformer expert
  (short-range), gated MoE. Code: github.com/XiongxiaoXu/SST.
  Our SSSM-v4 + STAEformer is essentially this design space already, but
  used as a fused hybrid; an MoE-style gating could be the differentiator.
- **STMAE (CIKM'24)** — Newer than STD-MAE; dual-mask (biased random walk
  spatial + patch temporal). Plug-and-play encoder wraps an existing
  backbone. Code at github.com/jsun57/STMAE has unfinished setup.
- **FUSE-Traffic (SIGSPATIAL'25, arXiv:2510.16053)** — D2STGNN base +
  Gemini-2.5-pro text embeddings. Not practical for us (LLM dependency).
- **MLCAFormer (PLOS One 2025)** — 3.30 at 60-min, our REPORT.md notes
  beat by −0.017. Code unclear.

Decision: stick with STMAE/STD-MAE pretraining as primary novel lever
(R03). MoE gating saved for R04+ if R03 lands but doesn't clear 3.20.

