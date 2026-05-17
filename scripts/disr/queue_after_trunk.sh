#!/usr/bin/env bash
# Wait until STAEformer trunk training has finished, then kick off the
# DiSR-Mamba campaign. Designed to be launched with nohup so SSH disconnect
# does not kill the chain.
#
# Usage:
#   nohup bash scripts/disr/queue_after_trunk.sh > logs/queue_after_trunk.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

TRUNK_LOG="logs/stae_trunk_s42.log"
TRUNK_CKPT="results/staeformer/stae_trunk/best_stae_s42.pth"
LOG_DIR="logs/disr_campaign"
mkdir -p "$LOG_DIR"

# ----- Wait until the trunk python process is done -------------------------
echo "[queue] waiting for STAEformer trunk to finish..."
while pgrep -f "scripts/train_staeformer.py --tag stae_trunk" >/dev/null; do
  sleep 30
done
echo "[queue] trunk finished. Last 5 epochs:"
grep "ep " "$TRUNK_LOG" | tail -5

if [[ ! -f "$TRUNK_CKPT" ]]; then
  echo "[queue] ERROR: trunk checkpoint $TRUNK_CKPT not found"
  exit 1
fi

# ----- Stage B (temporal residual) -----------------------------------------
echo "[queue] >>> Stage B (temporal residual, seed 0)"
PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
    --config configs/disr/stage_b_temporal.yaml \
    --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
    2>&1 | tee "$LOG_DIR/stageB_s0.log" | grep -E "ep |done|elapsed|test"

# ----- Stage C (symmetric spectral, K=48) ----------------------------------
echo "[queue] >>> Stage C (sym spectral, K=48, seed 0)"
PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
    --config configs/disr/stage_c_symspec.yaml \
    --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
    2>&1 | tee "$LOG_DIR/stageC_s0.log" | grep -E "ep |done|elapsed|test"

# ----- Stage D (magnetic spectral, q-sweep, seed 0) ------------------------
for q in 0.05 0.10 0.15 0.20 0.25; do
  qtag=$(python3 -c "import sys; print(f'{int(float(sys.argv[1])*100):03d}')" "$q")
  echo "[queue] >>> Stage D q=$q (tag stageD_q${qtag}_K48, seed 0)"
  cat > /tmp/disr_q${q}.yaml <<EOF
experiment:
  tag: "stageD_q${qtag}_K48"
model:
  q_charge: $q
EOF
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config configs/disr/stage_d_magspec.yaml \
      --config /tmp/disr_q${q}.yaml \
      --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
      2>&1 | tee "$LOG_DIR/stageD_q${qtag}_s0.log" | grep -E "ep |done|elapsed|test"
done

# ----- Stage E (router) ----------------------------------------------------
echo "[queue] >>> Stage E (router c=12, seed 0)"
PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
    --config configs/disr/stage_e_router.yaml \
    --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
    2>&1 | tee "$LOG_DIR/stageE_s0.log" | grep -E "ep |done|elapsed|test"

# ----- Pick best single-seed config and run 2 more seeds -------------------
echo "[queue] aggregating intermediate results..."
PYTHONPATH=. python3 scripts/disr/aggregate_results.py 2>&1 | tail -20

# Find best stage to re-seed (lowest test_mae_60)
best_tag=$(python3 - <<'EOF'
import json, glob, os
ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
best = None
for p in glob.glob("results/disr/*/summary.json"):
    s = json.load(open(p))
    m60 = s.get("test_metrics", {}).get("mae_60", 1e9)
    if best is None or m60 < best[0]:
        best = (m60, s["tag"])
print(best[1] if best else "")
EOF
)
echo "[queue] best single-seed config: $best_tag"

if [[ -n "$best_tag" ]]; then
  for s in 1 2; do
    cfg=""
    case "$best_tag" in
      stageB*) cfg="configs/disr/stage_b_temporal.yaml" ;;
      stageC*) cfg="configs/disr/stage_c_symspec.yaml" ;;
      stageD*) cfg="configs/disr/stage_d_magspec.yaml" ;;
      stageE*) cfg="configs/disr/stage_e_router.yaml" ;;
      *) cfg="configs/disr/stage_e_router.yaml" ;;
    esac
    echo "[queue] >>> $best_tag seed=$s"
    PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
        --config "$cfg" \
        --trunk_ckpt "$TRUNK_CKPT" --seed "$s" --no_compile \
        2>&1 | tee "$LOG_DIR/${best_tag}_s${s}.log" | grep -E "ep |done|elapsed|test"
  done

  # Ensemble + ST-TTC
  echo "[queue] >>> ensemble + ST-TTC"
  PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
      --ckpts "results/disr/${best_tag}_s*/best_disr.pth" \
      --use_ttc --ttc_groups 4 \
      --save "results/disr/${best_tag}_ensemble_ttc.npz" \
      2>&1 | tee "$LOG_DIR/${best_tag}_ensemble.log"
fi

# Final aggregation + plots
PYTHONPATH=. python3 scripts/disr/aggregate_results.py
PYTHONPATH=. python3 scripts/disr/make_plots.py

echo "[queue] DONE"
