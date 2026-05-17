# DiSR-Mamba Campaign Notes

**Run started**: 2026-05-14
**Hardware**: NVIDIA H200 SXM (143 GiB), bf16 AMP
**Trunk**: STAEformer reproduced from scratch on this pod
**Target**: beat 60-min test MAE 3.2603 (current best from 24-model + ST-TTC ensemble per [REPORT.md](../../REPORT.md))

## Trunk state

| Seed | Best Val MAE (avg) | Val MAE 60 | Best Ep | Notes |
|---|---:|---:|---:|---|
| 42 | 2.7395 | 3.147 | 22 | Killed at ep 29 once plateau confirmed; matches paper baseline |

## Pipeline confirmed working

- `models/disr/` modules wired together; all 23 unit tests pass on H200.
- `mamba_ssm` imported via deep path (`mamba_ssm.modules.mamba_simple`)
  bypassing the brittle `transformers.generation` dependency. Falls back
  to bidirectional GRU on CPU.
- `causal-conv1d` not installable (CUDA 12.4 / Torch 2.4.1 wheel missing).
  Mamba forward+backward works without it; pre-conv uses `nn.Conv1d`.
- Symmetric Laplacian + magnetic Laplacian + cluster bases all
  pre-computed in `cache/gft/disr/` (30 s total). Adjacency is asymmetric
  by construction on METR-LA, so we use it directly as A_dir.
- 1-epoch smoke for both Stage B (temporal residual) and Stage D
  (mag+sym+temporal) succeeds end-to-end; pipeline produces test
  predictions and saves NPZ for downstream stacking.

## Open questions for the writeup

1. **Why does the trunk plateau at 2.74 val instead of paper 2.72?**
   This is the same single-seed deviation observed in the prior campaign
   (R01 seeds 4–7 sat at 2.72–2.74). Likely seed-and-init variance; not
   a methodological gap. Multi-seed averaging closes it.

2. **What's the realistic target for a single DiSR run?**
   We bias the prior toward the residual learning ≤0.05–0.08 60-min MAE
   improvement over the trunk per seed. With single STAEformer baseline
   at 3.34, single DiSR best is plausibly 3.28–3.30. Ensemble + ST-TTC
   adds another 0.005–0.015 → 3.27–3.29. To clear 3.2603 we likely need
   multi-seed plus ST-TTC.

3. **What kills the q-sweep first?**
   Compute: each magnetic-K48 run is ~100 min. With patience 12 it
   should early-stop earlier. The campaign is configured to try q=0.10
   first and only sweep more if Stage D shows promise (i.e. beats Stage
   C).

## Campaign chain (running now)

```
Stage B (temporal residual, control)         ~30-40 min
Stage C (symmetric spectral)                 ~50-60 min
Stage D q=0.10 (magnetic spectral)           ~80-100 min
Stage E (horizon-cluster router, all experts) ~100-130 min
[best config × 2 more seeds]                  variable
Ensemble + ST-TTC                             ~5 min
```

Logs:
- `logs/queue_after_trunk.log`           overall driver
- `logs/disr_campaign/stage*_s*.log`     per-stage stdout
- `results/disr/<tag>_s<seed>/summary.json` per-run metrics
- `results/disr/disr_results.csv`        running CSV leaderboard
- `results/disr/ablation_table.{csv,md}` aggregated table (via `aggregate_results.py`)

## Stopping rule

- Halt the q-sweep and skip remaining variations once any single run
  beats 3.2603 60-min test MAE.
- Otherwise: run B → C → D(q=0.10) → E sequentially. After Stage E,
  pick the best config and re-train with 2 additional seeds; ensemble +
  ST-TTC.
