#!/bin/bash
# Chained campaign to populate the paper's ablation, multi-seed, and
# cross-dataset tables. Designed to run overnight via nohup; each stage
# writes its own log so partial failures don't lose earlier work.
#
# Order chosen so the highest-leverage stages finish first:
#   1. Ablation table on PEMS-BAY     (paper §4.3)
#   2. Multi-seed PEMS-BAY baselines  (paper §4.2 stddev)
#   3. Multi-seed PEMS-BAY hybrids    (paper §4.2 stddev)
#   4. PEMS04 baseline + hybrid       (paper §4.2 cross-dataset)
#   5. PEMS08 baseline + hybrid       (paper §4.2 cross-dataset)
#   6. Multi-seed METR-LA baselines + hybrids  (paper §4.2 saturation evidence)
#
# Total wall-clock target: ~12-15h on H200.
#
set -euo pipefail
cd /workspace/city-scale-ai
source venv/bin/activate
mkdir -p logs

stage_log () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE $1: $2" | tee -a logs/campaign_master.log; }

# Quiet warnings, unbuffer Python so logs stream live.
export PYTHONWARNINGS=ignore::FutureWarning,ignore::DeprecationWarning
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Stage 0: download + prepare PEMS04 / PEMS08 (cheap, no GPU).
# ---------------------------------------------------------------------------
stage_log 0 "Preparing PEMS04 / PEMS08 datasets"
python -u scripts/prepare_pems04_08.py > logs/stage0_prepare_pems.log 2>&1 || \
  stage_log 0 "FAILED (will skip PEMS04/08 stages)"
stage_log 0 "done"

# ---------------------------------------------------------------------------
# Stage 1: Ablation table on PEMS-BAY at seed 42 (6 variants).
# ---------------------------------------------------------------------------
stage_log 1 "Ablation table on PEMS-BAY (seed 42)"
bash scripts/run_ablations_stae_spec.sh pems_bay 42 \
  > logs/stage1_ablation_pems_bay.log 2>&1
stage_log 1 "done"

# ---------------------------------------------------------------------------
# Stage 2: extra STAEformer baselines on PEMS-BAY (seeds 1, 2). We already
# have seed 42 (saved as results/staeformer/pems_bay_stae_s42).
# ---------------------------------------------------------------------------
stage_log 2 "STAEformer PEMS-BAY seeds 1, 2"
for SEED in 1 2; do
  python -u scripts/train_staeformer.py \
    --tag "pems_bay_stae_s${SEED}" --seed "$SEED" \
    --data_path data/pems_bay.h5 --adj_path data/adj_PEMS-BAY.pkl \
    --cache_dir cache/gft_bay --epochs 60 \
    > "logs/stage2_pems_bay_stae_s${SEED}.log" 2>&1
done
stage_log 2 "done"

# ---------------------------------------------------------------------------
# Stage 3: STAE-Spectral-Magma on PEMS-BAY (seeds 1, 2). Seed 42 already
# saved as results/stae_spectral_magma/ablate_pems_bay_full from stage 1.
# ---------------------------------------------------------------------------
stage_log 3 "STAE-Spec hybrid PEMS-BAY seeds 1, 2"
for SEED in 1 2; do
  python -u scripts/train_stae_spectral_magma.py \
    --tag "pems_bay_hybrid_s${SEED}" --seed "$SEED" \
    --data_path data/pems_bay.h5 --adj_path data/adj_PEMS-BAY.pkl \
    --cache_dir cache/gft_bay --epochs 60 --patience 30 \
    --lr_milestones 20 30 --lr_gamma 0.1 \
    --weight_decay 3e-4 --gradient_clip 5.0 \
    > "logs/stage3_pems_bay_hybrid_s${SEED}.log" 2>&1
done
stage_log 3 "done"

# ---------------------------------------------------------------------------
# Stage 4: PEMS04 baseline + hybrid (one seed each — cheap; multi-seed
# saved for the final paper revision if the result is interesting).
# ---------------------------------------------------------------------------
if [[ -f data/pems04.npz && -f data/adj_PEMS04.pkl ]]; then
  stage_log 4 "PEMS04 baseline + hybrid (seed 42)"
  python -u scripts/train_staeformer.py \
    --tag "pems04_stae_s42" --seed 42 \
    --data_path data/pems04.npz --adj_path data/adj_PEMS04.pkl \
    --cache_dir cache/gft_pems04 --epochs 80 \
    > logs/stage4_pems04_stae.log 2>&1
  python -u scripts/train_stae_spectral_magma.py \
    --tag "pems04_hybrid_s42" --seed 42 \
    --data_path data/pems04.npz --adj_path data/adj_PEMS04.pkl \
    --cache_dir cache/gft_pems04 --epochs 80 --patience 30 \
    --lr_milestones 20 30 --lr_gamma 0.1 \
    --weight_decay 3e-4 --gradient_clip 5.0 \
    > logs/stage4_pems04_hybrid.log 2>&1
  stage_log 4 "done"
else
  stage_log 4 "SKIP (PEMS04 data missing)"
fi

# ---------------------------------------------------------------------------
# Stage 5: PEMS08 baseline + hybrid.
# ---------------------------------------------------------------------------
if [[ -f data/pems08.npz && -f data/adj_PEMS08.pkl ]]; then
  stage_log 5 "PEMS08 baseline + hybrid (seed 42)"
  python -u scripts/train_staeformer.py \
    --tag "pems08_stae_s42" --seed 42 \
    --data_path data/pems08.npz --adj_path data/adj_PEMS08.pkl \
    --cache_dir cache/gft_pems08 --epochs 80 \
    > logs/stage5_pems08_stae.log 2>&1
  python -u scripts/train_stae_spectral_magma.py \
    --tag "pems08_hybrid_s42" --seed 42 \
    --data_path data/pems08.npz --adj_path data/adj_PEMS08.pkl \
    --cache_dir cache/gft_pems08 --epochs 80 --patience 30 \
    --lr_milestones 20 30 --lr_gamma 0.1 \
    --weight_decay 3e-4 --gradient_clip 5.0 \
    > logs/stage5_pems08_hybrid.log 2>&1
  stage_log 5 "done"
else
  stage_log 5 "SKIP (PEMS08 data missing)"
fi

# ---------------------------------------------------------------------------
# Stage 6: 3-seed METR-LA baseline + hybrid. METR-LA is saturated, but the
# negative result needs multi-seed evidence to be defensible in §5.
# ---------------------------------------------------------------------------
stage_log 6 "METR-LA multi-seed baseline + hybrid"
for SEED in 42 1 2; do
  python -u scripts/train_staeformer.py \
    --tag "metr_la_stae_s${SEED}" --seed "$SEED" \
    --data_path data/METR-LA.h5 --adj_path data/adj_METR-LA.pkl \
    --cache_dir cache/gft --epochs 60 \
    > "logs/stage6_metr_la_stae_s${SEED}.log" 2>&1
  python -u scripts/train_stae_spectral_magma.py \
    --tag "metr_la_hybrid_s${SEED}" --seed "$SEED" \
    --data_path data/METR-LA.h5 --adj_path data/adj_METR-LA.pkl \
    --cache_dir cache/gft --epochs 60 --patience 30 \
    --lr_milestones 20 30 --lr_gamma 0.1 \
    --weight_decay 3e-4 --gradient_clip 5.0 \
    > "logs/stage6_metr_la_hybrid_s${SEED}.log" 2>&1
done
stage_log 6 "done"

stage_log COMPLETE "All stages finished."
