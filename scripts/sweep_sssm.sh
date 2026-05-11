#!/bin/bash
# Sweep across (d_model, num_layers, seed) on the 4090.
# Runs sequentially so they don't compete for VRAM.

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

run() {
    local tag=$1; shift
    echo "=== launching $tag at $(date -u +%H:%M:%S) ==="
    python3 -u scripts/train_sssm.py --tag "$tag" "$@" 2>&1 | tee "logs/$tag.log" | tail -2
    echo "=== done $tag at $(date -u +%H:%M:%S) ==="
}

# Larger model, single seed (the main run)
run d128_L4_s42 --k 207 --d_model 128 --num_layers 4 \
    --epochs 100 --patience 25 --batch_size 64 \
    --learning_rate 1e-3 --seed 42

# Multi-seed at headline config
for s in 1 2 3; do
    run d96_L3_s${s} --k 207 --d_model 96 --num_layers 3 \
        --epochs 100 --patience 20 --batch_size 64 \
        --learning_rate 1e-3 --seed $s
done

echo "ALL DONE"
