#!/bin/bash
# R02: Bigger STAEformer (d=192, L=4) + stronger reg
set -e
cd /workspace/city-scale-ai
mkdir -p logs results/staeformer

TAG="stae_R02_big_s42"
echo "=== launching $TAG at $(date -u +%H:%M:%S) ==="
python3 -u scripts/train_staeformer.py \
    --tag "$TAG" --seed 42 --batch_size 16 \
    --input_embedding_dim 32 --tod_embedding_dim 32 --dow_embedding_dim 32 \
    --adaptive_embedding_dim 96 \
    --feed_forward_dim 384 --num_layers 4 \
    --dropout 0.15 --weight_decay 5e-4 \
    --lr_milestones 25 40 \
    --epochs 200 --patience 30 \
    > "logs/${TAG}.log" 2>&1

echo "=== R02 DONE $TAG at $(date -u +%H:%M:%S) ==="
