#!/usr/bin/env bash
# Driver script for the full DiSR-Mamba campaign on the H200 / 4090.
#
# Order:
#   1. Train STAEformer trunk (4 seeds) if no usable ckpt exists.
#   2. Stage B (temporal residual) - control.
#   3. Stage C (symmetric spectral) - sweep over K.
#   4. Stage D (magnetic spectral) - sweep over q.
#   5. Stage E (horizon-cluster router) - sweep over n_clusters.
#   6. Ablations (no congestion, no mode axis, ...).
#   7. Multi-seed best candidate + ST-TTC eval.
#
# Usage:
#   bash scripts/disr/run_campaign.sh [--skip-stae]

set -euo pipefail
cd "$(dirname "$0")/../.."

LOGDIR=logs
mkdir -p "$LOGDIR"

SKIP_STAE=0
for arg in "$@"; do
  case "$arg" in
    --skip-stae) SKIP_STAE=1 ;;
  esac
done

# ----- 1. STAEformer trunk (4 seeds) ----------------------------------------
if [[ "$SKIP_STAE" -eq 0 ]]; then
  for s in 42 1 2 3; do
    tag="stae_trunk_s${s}"
    out="results/staeformer/$tag/best_stae_s$s.pth"
    if [[ -f "$out" ]]; then
      echo "[stae] skipping seed $s (ckpt exists)"
      continue
    fi
    echo "[stae] training seed $s"
    PYTHONPATH=. python3 -u scripts/train_staeformer.py \
        --tag "$tag" --seed "$s" --epochs 80 --patience 20 \
        --batch_size 16 --num_workers 4 \
        2>&1 | tee "$LOGDIR/stae_$s.log"
  done
fi

# ----- 2. Stage B: temporal residual ----------------------------------------
PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
    --config configs/disr/stage_b_temporal.yaml --seed 0 --no_compile \
    2>&1 | tee "$LOGDIR/disr_stageB_s0.log"

# ----- 3. Stage C: symmetric spectral ---------------------------------------
PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
    --config configs/disr/stage_c_symspec.yaml --seed 0 --no_compile \
    2>&1 | tee "$LOGDIR/disr_stageC_s0.log"

# ----- 4. Stage D: magnetic spectral (small sweep over q) -------------------
for q in 0.05 0.10 0.15 0.20 0.25; do
  cat > /tmp/disr_q${q}.yaml <<EOF
experiment: {tag: "stageD_q${q}"}
model: {q_charge: $q}
EOF
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --base_config configs/disr/stage_d_magspec.yaml \
      --config /tmp/disr_q${q}.yaml --seed 0 --no_compile \
      2>&1 | tee "$LOGDIR/disr_stageD_q${q}.log"
done

# ----- 5. Stage E: router ---------------------------------------------------
PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
    --config configs/disr/stage_e_router.yaml --seed 0 --no_compile \
    2>&1 | tee "$LOGDIR/disr_stageE_s0.log"

# ----- 6. Best candidate, multi-seed + ST-TTC eval --------------------------
for s in 0 1 2 3 4; do
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config configs/disr/stage_e_router.yaml --seed "$s" --no_compile \
      --tag_suffix "_final" \
      2>&1 | tee "$LOGDIR/disr_final_s${s}.log"
done

PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/stageE_router*_final_s*/best_disr.pth" \
    --use_ttc --ttc_groups 4 \
    --save results/disr/final_ensemble_ttc.npz \
    2>&1 | tee "$LOGDIR/disr_final_eval.log"

echo "[campaign] done."
