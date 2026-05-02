#!/bin/bash
set -e
#SBATCH --job-name=mamba_k_sweep
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mamba_k_sweep_%j.out
#SBATCH --error=logs/mamba_k_sweep_%j.err

echo "Job started"
hostname
whoami
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# TODO: Change this to your repository's path on Skynet
cd ~/City-Scale-AI
mkdir -p logs

# TODO: Change this to the path of your virtual environment on Skynet
ssource ~/miniconda3/bin/activate
conda activate mamba310

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

bash scripts/run_mamba_k_sweep.sh

echo "Job finished"
