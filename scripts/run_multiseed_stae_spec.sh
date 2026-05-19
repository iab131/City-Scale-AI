#!/bin/bash
# Multi-seed campaign: 3 hybrid seeds + 3 STAEformer-baseline seeds on a
# chosen dataset. Provides the error bars needed to claim that any
# observed STAE-Spectral-Magma gain is real and not seed lottery.
#
# Usage:
#   ./scripts/run_multiseed_stae_spec.sh <dataset>
#   dataset ∈ {metr_la, pems_bay, pems04, pems08}
#
set -euo pipefail

DATASET="${1:-pems_bay}"
SEEDS=(42 1 2)

case "$DATASET" in
  metr_la)
    DATA="data/METR-LA.h5"
    ADJ="data/adj_METR-LA.pkl"
    CACHE="cache/gft"
    ;;
  pems_bay)
    DATA="data/pems_bay.h5"
    ADJ="data/adj_PEMS-BAY.pkl"
    CACHE="cache/gft_bay"
    ;;
  pems04)
    DATA="data/pems04.npz"
    ADJ="data/adj_PEMS04.pkl"
    CACHE="cache/gft_pems04"
    ;;
  pems08)
    DATA="data/pems08.npz"
    ADJ="data/adj_PEMS08.pkl"
    CACHE="cache/gft_pems08"
    ;;
  *)
    echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source venv/bin/activate
mkdir -p logs

COMMON_HYBRID=(
  --data_path "$DATA" --adj_path "$ADJ" --cache_dir "$CACHE"
  --epochs 60 --patience 30
  --lr_milestones 20 30 --lr_gamma 0.1
  --weight_decay 3e-4 --gradient_clip 5.0
)
COMMON_BASELINE=(
  --data_path "$DATA" --adj_path "$ADJ" --cache_dir "$CACHE"
  --epochs 60
)

# Run STAEformer baseline first (faster to converge, gives the reference
# point we judge the hybrid against).
for SEED in "${SEEDS[@]}"; do
  TAG="baseline_${DATASET}_s${SEED}"
  LOG="logs/${TAG}.log"
  echo "[$(date +%H:%M:%S)] starting $TAG"
  PYTHONWARNINGS=ignore::FutureWarning,ignore::DeprecationWarning \
  PYTHONUNBUFFERED=1 \
    python -u scripts/train_staeformer.py \
      --tag "$TAG" --seed "$SEED" "${COMMON_BASELINE[@]}" \
      > "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] finished $TAG (log: $LOG)"
done

# Then the hybrid at the same seeds — error bars come from comparing the
# matched seed pairs.
for SEED in "${SEEDS[@]}"; do
  TAG="hybrid_${DATASET}_s${SEED}"
  LOG="logs/${TAG}.log"
  echo "[$(date +%H:%M:%S)] starting $TAG"
  PYTHONWARNINGS=ignore::FutureWarning,ignore::DeprecationWarning \
  PYTHONUNBUFFERED=1 \
    python -u scripts/train_stae_spectral_magma.py \
      --tag "$TAG" --seed "$SEED" "${COMMON_HYBRID[@]}" \
      > "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] finished $TAG (log: $LOG)"
done

echo "[$(date +%H:%M:%S)] multi-seed campaign complete on $DATASET"
