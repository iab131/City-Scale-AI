#!/bin/bash

# You can run this script interactively or as part of a Slurm job
echo "Starting Mamba k-sweep..."

python scripts/train_mamba.py --device cuda --k 8 --epochs 50 --batch_size 16
python scripts/train_mamba.py --device cuda --k 16 --epochs 50 --batch_size 16
python scripts/train_mamba.py --device cuda --k 32 --epochs 50 --batch_size 16
python scripts/train_mamba.py --device cuda --k 64 --epochs 50 --batch_size 16
python scripts/train_mamba.py --device cuda --k 96 --epochs 50 --batch_size 16
python scripts/train_mamba.py --device cuda --k 128 --epochs 50 --batch_size 16

echo "k-sweep finished!"
