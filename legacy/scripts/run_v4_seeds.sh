#!/bin/bash
# Run 3 additional seeds of v4 sequentially. seed=42 already in
# results/sssm/v4_d96_L3/. We add seeds {1, 2, 3} so the ensemble has 4 models.
set -e
cd /workspace/city-scale-ai
mkdir -p logs

for SEED in 1 2 3; do
    TAG="v4_d96_L3_s${SEED}"
    echo "=== launching $TAG at $(date -u +%H:%M:%S) ==="
    python3 -u scripts/train_sssm.py \
        --version v4 --k 207 --d_model 96 --num_layers 3 \
        --epochs 100 --patience 20 --batch_size 64 \
        --learning_rate 1e-3 --dropout 0.1 --warmup_epochs 3 \
        --tag "$TAG" --seed "$SEED" \
        > "logs/${TAG}.log" 2>&1
    echo "=== done $TAG at $(date -u +%H:%M:%S) ==="
done

echo "ALL SEEDS DONE"
