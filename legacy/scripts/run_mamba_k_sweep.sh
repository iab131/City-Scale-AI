#!/bin/bash
set -e
# You can run this script interactively or as part of a Slurm job
echo "Starting Mamba k-sweep..."

python scripts/train_mamba.py --device cuda --k 1 --d_model 64 --num_layers 3 --batch_size 32 --learning_rate 0.0005  --seed 42
python scripts/train_mamba.py --device cuda --k 1 --d_model 64 --num_layers 3 --batch_size 32 --learning_rate 0.001  --seed 42
python scripts/train_mamba.py --device cuda --k 1 --d_model 64 --num_layers 3 --batch_size 32 --learning_rate 0.002  --seed 42

echo "k-sweep finished!"