#!/bin/bash
# Phase 5: more STAEformer seeds with dropout=0.15 (the R01 seed 6 winning config)
# Trains 4 additional seeds for max ensemble diversity at the winning hyperparam.
set +e
cd /workspace/city-scale-ai
mkdir -p logs results/staeformer

echo "=== PHASE 5 START at $(date -u +%H:%M:%S) ==="

for SEED in 11 22 33; do
    TAG="stae_R13_d15_s${SEED}"
    echo "=== R13 $TAG ==="
    python3 -u scripts/train_staeformer.py \
        --tag "$TAG" --seed "$SEED" \
        --batch_size 16 --dropout 0.15 \
        --epochs 80 --patience 20 \
        > "logs/${TAG}.log" 2>&1
    echo "=== R13 $TAG done at $(date -u +%H:%M:%S) ==="
done

# R14: STAEformer-with-prior (calendar prior as input feature)
for SEED in 42 1; do
    TAG="stae_R14_prior_s${SEED}"
    echo "=== R14 $TAG ==="
    python3 -u scripts/train_staeformer_prior.py \
        --tag "$TAG" --seed "$SEED" \
        --batch_size 16 --dropout 0.15 \
        --epochs 80 --patience 20 \
        > "logs/${TAG}.log" 2>&1
    echo "=== R14 $TAG done at $(date -u +%H:%M:%S) ==="
done

# Final super-ensemble eval with everything
echo "=== Phase 5 final eval ==="
python3 -u scripts/eval_R04_super_ensemble.py \
    --use_ttc --ttc_groups 4 \
    --stae_ckpts "results/staeformer/stae_R*/best_stae_s*.pth" \
    --include_gwnet --include_hybrid \
    --out_json results/R13_phase5_ensemble.json \
    > logs/R13_phase5_ensemble.log 2>&1

python3 -u scripts/show_leaderboard.py > logs/phase5_leaderboard.log 2>&1
echo "=== PHASE 5 DONE at $(date -u +%H:%M:%S) ==="
