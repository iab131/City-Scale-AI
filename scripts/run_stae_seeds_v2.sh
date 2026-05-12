#!/bin/bash
# Run STAEformer with seeds 1, 2, 3 sequentially. seed 42 already done.
set -e
cd /workspace/city-scale-ai
mkdir -p logs results/staeformer

for SEED in 1 2 3; do
    TAG="stae_repro_s${SEED}"
    echo "=== launching $TAG at $(date -u +%H:%M:%S) ==="
    python3 -u scripts/train_staeformer.py \
        --tag "$TAG" --seed "$SEED" --batch_size 16 \
        > "logs/${TAG}.log" 2>&1
    echo "=== done $TAG at $(date -u +%H:%M:%S) ==="
done

echo "ALL STAE SEEDS DONE"
