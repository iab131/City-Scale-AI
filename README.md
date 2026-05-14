# City-Scale Traffic Forecasting on METR-LA

**Team City Scale AI** — Nengjia Li, Udula Abeykoon, Anirudh Bharadwaj Vangara, Enhe Bai, Ryan Rana
University of Waterloo × Queen's University · Borealis AI / Let's Solve It · 2026

## Result

**60-min test MAE = 3.283** on METR-LA — **#1 among reproducible methods** as of May 2026.

| Horizon | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 15 min | **2.611** | 4.970 | 6.78 % |
| 30 min | **2.918** | 5.834 | 8.06 % |
| 60 min | **3.283** | 6.812 | 9.61 % |

Beats every other public method with working code, including MLCAFormer (3.30), STAEformer (3.34), STD-MAE (3.40), ST-SSDL (3.37), and FUSE-Traffic (3.39). The published numbers above this (TITAN 3.08, TESTAM+ 2.99) have no working public code; see `REPORT.md` for the full audit.

## Architecture

A 4-seed ensemble of **STAEformer** (CIKM 2023) reproduced on our pipeline, followed by the **ST-TTC** test-time spectral calibrator (NeurIPS 2025 Spotlight).

```
4 × STAEformer (seeds 42, 1, 2, 3) ─┐
                                     ├─► uniform-average normalized predictions
                                     │     ↓
                                     │   ST-TTC FFT amplitude+phase calibration
                                     │     (streaming flash-update at test time)
                                     ▼
                          test predictions in raw mph
```

The full STAEformer architecture (152-dim model, 3 temporal + 3 spatial Transformer layers, 80-dim adaptive embedding, mixed-projection output) is in `models/staeformer.py`. The ST-TTC calibrator (1 656 parameters total) is in `scripts/eval_stae_ensemble.py`.

## Reproduce the Result (2.5 hours on a single H200)

```bash
# 1. Install dependencies
pip install --no-build-isolation \
    "transformers<4.45" causal-conv1d==1.4.0 mamba-ssm==2.2.2 \
    h5py tables pandas scipy einops

# 2. Train 4 STAEformer seeds sequentially (~2 h)
python scripts/train_staeformer.py --tag stae_repro --seed 42 --batch_size 16
bash scripts/run_stae_seeds_v2.sh   # seeds 1, 2, 3

# 3. Final 4-seed ensemble + ST-TTC eval (~5 min)
python scripts/eval_stae_ensemble.py --use_ttc \
    --stae_ckpts "results/staeformer/stae_*/best_stae_s*.pth"
# Expected output: 60-min MAE 3.283
```

## Repository Structure

```
├── REPORT.md                          full technical report
├── README.md                          this file
├── requirements.txt
│
├── data/                              METR-LA raw files (.h5, .pkl)
├── cache/                             cached GFT artifacts (auto-generated)
│
├── src/                               core pipeline
│   ├── data_utils.py                  H5 + adjacency loaders
│   ├── graph_utils.py                 Laplacian / adjacency utilities
│   ├── gft.py                         Graph Fourier Transform
│   ├── preprocess_v2.py               masked z-score + TOD/DOW features
│   └── dataset_v2.py                  sliding-window dataset
│
├── models/                            architectures
│   ├── staeformer.py                  the headline backbone
│   ├── spectral_ssm.py                our novel SSSM v1–v8 (bi-axis Mamba)
│   ├── graph_wavenet.py               GWNet ensemble member
│   └── hybrid.py                      STAEformer + spectral branch (ablation)
│
├── scripts/                           training & eval
│   ├── train_staeformer.py            STAEformer training (paper-faithful + ablation flags)
│   ├── train_sssm.py                  Spectral State Space Model variants
│   ├── train_gwnet.py                 GraphWaveNet training
│   ├── train_hybrid.py                hybrid training
│   ├── eval_stae_ensemble.py          HEADLINE: 4-seed STAE + ST-TTC eval
│   ├── eval_full_ensemble.py          multi-architecture ensemble with val-weighting
│   ├── eval_ensemble.py               older v4 SSSM ensemble (legacy)
│   └── run_stae_seeds_v2.sh           sequential 4-seed training driver
│
├── legacy/                            preserved earlier work + failed exploration
│   └── README.md                      what's there and why
│
└── docs/
    └── ssh-note.md                    Skynet cluster notes (original team)
```

## What This Repository Contains

**Active pipeline** (gives the 3.283 result):
- `src/`, `models/{staeformer,spectral_ssm,graph_wavenet,hybrid}.py`, `scripts/train_*.py`, `scripts/eval_*.py`.

**Novel research contribution** (not used by the headline result, but documented in the report):
- `models/spectral_ssm.py` — the bi-axis spectral Mamba family v1–v8. Mamba scanning along both time and spectral-mode axes is, to our knowledge, absent from every other published METR-LA model. It plateaued at 60-min test MAE 3.82, which we documented as a structural ceiling and then pivoted away from.

**Preserved earlier work and negative results** in `legacy/`:
- The original project's SpectralGRU + SpectralMambaReal pipeline.
- Attempted reproductions that did NOT work (STGormer reproduction came in at 3.58 vs paper's 3.10).
- Older exploratory scripts (multi-window, calendar prior, etc.).

## The Bigger Picture

We document a **reproducibility crisis** on METR-LA: every "above-3.30" published claim from 2024–2025 has a serious issue (paper withdrawn from venue, code never released, baseline numbers disagree with peer papers, or reproducers fail to match). Our 3.283 is the best result that anyone can actually run end-to-end from public code. See `REPORT.md` § 6 for the full audit.

## Citation

If you use this work, please cite:
```
City Scale AI / Borealis AI Let's Solve It 2026, "Spectral State Space Models and a Reproducible
SOTA Pipeline for METR-LA". Internal report. University of Waterloo × Queen's University, 2026.
```
