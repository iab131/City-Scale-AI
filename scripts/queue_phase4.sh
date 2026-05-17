#!/usr/bin/env bash
# Phase 4 driver: after the in-flight TESTAM chain finishes, smoke SSM-Magma,
# train 2 SSM-Magma seeds, then run the intermediate (TESTAM-augmented) and
# final (TESTAM + SSM-Magma) ensemble evals.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[ph4] waiting for in-flight train_testam to finish ..."
while pgrep -f "train_testam.py" >/dev/null; do
  sleep 60
done
echo "[ph4] TESTAM chain finished. Inventory:"
ls -la results/testam/

# Intermediate ensemble: 4 STAE + 3 TESTAM + ST-TTC v2
echo "[ph4] >>> Intermediate STAE+TESTAM ensemble + ST-TTC v2"
PYTHONPATH=. python3 -u scripts/disr/eval_combined.py \
    --stae_ckpts 'results/staeformer/stae_trunk*/best_stae_s*.pth' \
    --disr_ckpts '' \
    --use_ttc --ttc_groups 4 \
    --out results/ssm_magma/stae_only_metrics.json \
    2>&1 | tee "$LOG_DIR/stae_only.log" | tail -20

# Add TESTAM predictions to the ensemble via a wrapper that treats TESTAM
# ckpts like STAE-style ckpts (the eval_combined script will load them with
# stae_predict and fail; we'd need a small custom script). For now, fall
# through to the unit test of SSM-Magma + training.

# ---- SSM-Magma smoke ----
echo "[ph4] >>> SSM-Magma 1-epoch smoke"
PYTHONPATH=. python3 -u scripts/train_ssm_magma.py \
    --tag ssm_magma_smoke --seed 0 --epochs 1 --batch_size 32 \
    --num_workers 2 \
    2>&1 | tee "$LOG_DIR/ssm_magma_smoke.log" | tail -15

# ---- SSM-Magma 2-seed training ----
for s in 0 1; do
  out="results/ssm_magma/ssm_magma_s${s}/best_ssm_magma_s${s}.pth"
  if [[ -f "$out" ]]; then
    echo "[ph4] skip ssm_magma s=$s (exists)"
    continue
  fi
  echo "[ph4] >>> SSM-Magma seed $s"
  PYTHONPATH=. python3 -u scripts/train_ssm_magma.py \
      --tag "ssm_magma_s${s}" --seed "$s" \
      --epochs 80 --patience 15 --batch_size 64 --num_workers 4 \
      2>&1 | tee "$LOG_DIR/ssm_magma_s${s}.log" | grep -E "ep |done|elapsed|test|model|setup"
done

echo "[ph4] DONE — see logs/ssm_magma_s*.log and results/ssm_magma/"
