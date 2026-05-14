#!/bin/bash
# R01: 4 additional STAEformer seeds with hyperparam diversity (4, 5, 6, 7)
set -e
cd /workspace/city-scale-ai
mkdir -p logs results/staeformer

run_seed() {
    local SEED=$1
    local DROPOUT=$2
    local BATCH=$3
    local TAG="stae_R01_s${SEED}"
    echo "=== launching $TAG  dropout=$DROPOUT  batch=$BATCH  at $(date -u +%H:%M:%S) ==="
    python3 -u scripts/train_staeformer.py \
        --tag "$TAG" --seed "$SEED" \
        --batch_size "$BATCH" --dropout "$DROPOUT" \
        > "logs/${TAG}.log" 2>&1
    echo "=== done $TAG at $(date -u +%H:%M:%S) ==="
}

run_seed 4 0.10 16
run_seed 5 0.05 16
run_seed 6 0.15 16
run_seed 7 0.10 32

echo "R01 DONE at $(date -u +%H:%M:%S)"
