#!/usr/bin/env bash
# Continuation queue: assumes Stage B (and optionally Stage C) are done.
# Runs ONE magnetic q (q=0.10) + Stage E + multi-seed E + ensemble + ST-TTC.

set -euo pipefail
cd "$(dirname "$0")/../.."

TRUNK_CKPT="results/staeformer/stae_trunk/best_stae_s42.pth"
LOG_DIR="logs/disr_campaign"
mkdir -p "$LOG_DIR"

# ----- Stage C (skipped if already exists) ---------------------------------
if [[ ! -f "results/disr/stageC_symspec_s0/summary.json" ]]; then
  echo "[queue2] >>> Stage C (sym spectral, seed 0)"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config configs/disr/stage_c_symspec.yaml \
      --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
      2>&1 | tee "$LOG_DIR/stageC_s0.log" | grep -E "ep |done|elapsed|test" || true
fi

# ----- Stage D q=0.10 -------------------------------------------------------
if [[ ! -f "results/disr/stageD_magspec_s0/summary.json" ]]; then
  echo "[queue2] >>> Stage D (magnetic q=0.10, seed 0)"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config configs/disr/stage_d_magspec.yaml \
      --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
      2>&1 | tee "$LOG_DIR/stageD_q010_s0.log" | grep -E "ep |done|elapsed|test" || true
fi

# ----- Stage E (router, seed 0) --------------------------------------------
if [[ ! -f "results/disr/stageE_router_s0/summary.json" ]]; then
  echo "[queue2] >>> Stage E (router, seed 0)"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config configs/disr/stage_e_router.yaml \
      --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
      2>&1 | tee "$LOG_DIR/stageE_s0.log" | grep -E "ep |done|elapsed|test" || true
fi

# ----- Multi-seed the best so far ------------------------------------------
echo "[queue2] aggregating..."
PYTHONPATH=. python3 scripts/disr/aggregate_results.py 2>&1 | tail -15

best_tag=$(python3 - <<'EOF'
import json, glob
best = None
for p in glob.glob("results/disr/*/summary.json"):
    s = json.load(open(p))
    m60 = s.get("test_metrics", {}).get("mae_60", 1e9)
    if best is None or m60 < best[0]:
        best = (m60, s["tag"])
print(best[1] if best else "")
EOF
)
echo "[queue2] best single-seed config: $best_tag"

best_cfg=""
case "$best_tag" in
  stageB*) best_cfg="configs/disr/stage_b_temporal.yaml" ;;
  stageC*) best_cfg="configs/disr/stage_c_symspec.yaml" ;;
  stageD*) best_cfg="configs/disr/stage_d_magspec.yaml" ;;
  stageE*) best_cfg="configs/disr/stage_e_router.yaml" ;;
  *) best_cfg="configs/disr/stage_e_router.yaml" ;;
esac

# Multi-seed: 2 more seeds of the best config (so total 3 seeds)
for s in 1 2; do
  out="results/disr/${best_tag}_s${s}/summary.json"
  if [[ -f "$out" ]]; then
    echo "[queue2] skip $best_tag seed=$s (exists)"
    continue
  fi
  echo "[queue2] >>> $best_tag seed=$s"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config "$best_cfg" \
      --trunk_ckpt "$TRUNK_CKPT" --seed "$s" --no_compile \
      2>&1 | tee "$LOG_DIR/${best_tag}_s${s}.log" | grep -E "ep |done|elapsed|test" || true
done

# ----- Ensemble + ST-TTC ---------------------------------------------------
echo "[queue2] >>> ensemble + ST-TTC for $best_tag"
PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/${best_tag}_s*/best_disr.pth" \
    --use_ttc --ttc_groups 4 \
    --save "results/disr/${best_tag}_ensemble_ttc.npz" \
    2>&1 | tee "$LOG_DIR/${best_tag}_ensemble.log"

# ----- Final aggregation + plots -------------------------------------------
PYTHONPATH=. python3 scripts/disr/aggregate_results.py
PYTHONPATH=. python3 scripts/disr/make_plots.py

echo "[queue2] DONE"
