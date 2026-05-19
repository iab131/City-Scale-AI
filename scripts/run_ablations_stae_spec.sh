#!/bin/bash
# Ablation campaign for STAE-Spectral-Magma.
#
# Runs the full model + five single-piece-dropped variants on a chosen dataset.
# Each variant trains with the STAEformer-class LR schedule
# (lr_milestones=[20,30], lr_gamma=0.1) and gradient_clip=5.0 (required for
# the spectral sidechain — we explosion-tested this without clipping).
#
# Usage:
#   ./scripts/run_ablations_stae_spec.sh <dataset> <seed>
#   dataset ∈ {metr_la, pems_bay}
#
# Example:
#   ./scripts/run_ablations_stae_spec.sh pems_bay 42
#
set -euo pipefail

DATASET="${1:-pems_bay}"
SEED="${2:-42}"

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
  *)
    echo "unknown dataset: $DATASET (expected metr_la | pems_bay)" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source venv/bin/activate
mkdir -p logs results

COMMON_ARGS=(
  --data_path "$DATA"
  --adj_path "$ADJ"
  --cache_dir "$CACHE"
  --seed "$SEED"
  --epochs 60
  --patience 30
  --lr_milestones 20 30
  --lr_gamma 0.1
  --weight_decay 3e-4
  --gradient_clip 5.0
)

run_variant () {
  local tag="$1"; shift
  local log="logs/ablate_${DATASET}_${tag}_s${SEED}.log"
  echo "[$(date +%H:%M:%S)] [$DATASET][s$SEED] starting variant: $tag"
  PYTHONWARNINGS=ignore::FutureWarning,ignore::DeprecationWarning \
  PYTHONUNBUFFERED=1 \
    python -u scripts/train_stae_spectral_magma.py \
      --tag "ablate_${DATASET}_${tag}" \
      "${COMMON_ARGS[@]}" "$@" \
      > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] [$DATASET][s$SEED] finished: $tag  (log: $log)"
}

# Full model — the reference point for the ablation table.
run_variant "full"
# Each ablation strips ONE architectural piece and keeps everything else.
run_variant "no_sym"      --no-use_sym
run_variant "no_mag"      --no-use_mag
run_variant "no_sem"      --no-use_sem
run_variant "no_router"   --no-use_router
run_variant "no_modeaxis" --no-spec_mode_axis

echo "[$(date +%H:%M:%S)] ablation campaign complete on $DATASET seed $SEED"
echo "Results CSV: results/stae_spectral_magma/stae_spectral_magma_results.csv"
