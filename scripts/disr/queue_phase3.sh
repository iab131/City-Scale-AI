#!/usr/bin/env bash
# Phase 3 driver: after the in-flight finish_campaign + trunk_s1 settle,
# train trunk seeds 2 and 3, then DiSR-on-best on s1/s2/s3, then mega-ensemble.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/disr_campaign"
mkdir -p "$LOG_DIR"

# Determine "best" DiSR config by 60-min test MAE (same logic as before).
best_tag=$(python3 - <<'EOF'
import json, glob
best = None
for p in glob.glob("results/disr/*/summary.json"):
    s = json.load(open(p))
    m60 = s.get("test_metrics", {}).get("mae_60", 1e9)
    if best is None or m60 < best[0]:
        best = (m60, s["tag"])
print(best[1] if best else "stageE_router")
EOF
)
case "$best_tag" in
  stageB*) best_cfg="configs/disr/stage_b_temporal.yaml" ;;
  stageC*) best_cfg="configs/disr/stage_c_symspec.yaml" ;;
  stageD*) best_cfg="configs/disr/stage_d_magspec.yaml" ;;
  stageE*) best_cfg="configs/disr/stage_e_router.yaml" ;;
  *) best_cfg="configs/disr/stage_e_router.yaml" ;;
esac
echo "[ph3] best_cfg=$best_cfg (tag $best_tag)"

# ----- Wait for any active training to settle --------------------------
echo "[ph3] waiting for active trainings to settle ..."
while pgrep -f "train_disr.py" >/dev/null || \
      pgrep -f "train_staeformer.py" >/dev/null; do
  sleep 30
done
echo "[ph3] free to launch."

# ----- 1. Trunk seeds 2, 3 (sequential) --------------------------------
for s in 2 3; do
  tag="stae_trunk_s${s}"
  out="results/staeformer/${tag}/best_stae_s${s}.pth"
  if [[ -f "$out" ]]; then
    echo "[ph3] skip trunk seed $s (exists)"
    continue
  fi
  echo "[ph3] >>> trunk seed $s"
  PYTHONPATH=. python3 -u scripts/train_staeformer.py \
      --tag "$tag" --seed "$s" --epochs 45 --patience 12 \
      --batch_size 16 --num_workers 4 \
      2>&1 | tee "$LOG_DIR/trunk_s${s}.log" | grep -E "ep |done|elapsed|test"
done

# ----- 2. DiSR (best config) on each available trunk -------------------
for s in 1 2 3; do
  trunk="results/staeformer/stae_trunk_s${s}/best_stae_s${s}.pth"
  if [[ ! -f "$trunk" ]]; then
    echo "[ph3] missing trunk seed $s — skip"
    continue
  fi
  out="results/disr/${best_tag}_trunk${s}_s0/summary.json"
  if [[ -f "$out" ]]; then
    echo "[ph3] skip DiSR-on-trunk-$s (exists)"
    continue
  fi
  echo "[ph3] >>> DiSR on trunk seed $s"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config "$best_cfg" \
      --trunk_ckpt "$trunk" --seed 0 --no_compile \
      --tag_suffix "_trunk${s}" \
      2>&1 | tee "$LOG_DIR/disr_trunk${s}.log" | grep -E "ep |done|elapsed|test"
done

# ----- 3. Trunk-only ensemble ------------------------------------------
echo "[ph3] >>> trunk-only ensemble + ST-TTC (4 STAEformer seeds)"
PYTHONPATH=. python3 -u scripts/eval_stae_ensemble.py --use_ttc --ttc_groups 4 \
    --stae_ckpts "results/staeformer/stae_trunk*/best_stae_s*.pth" \
    2>&1 | tee "$LOG_DIR/trunk_ensemble.log" | tail -40

# ----- 4. Mega-ensemble of all DiSR variants + ST-TTC ------------------
echo "[ph3] >>> mega-ensemble (all DiSR runs) + ST-TTC"
PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/*_s*/best_disr.pth" \
    --use_ttc --ttc_groups 4 \
    --save "results/disr/mega_ensemble_ttc.npz" \
    2>&1 | tee "$LOG_DIR/mega_ensemble.log" | tail -40

# ----- 5. Final aggregate + plots --------------------------------------
PYTHONPATH=. python3 scripts/disr/aggregate_results.py
PYTHONPATH=. python3 scripts/disr/make_plots.py

echo "[ph3] DONE"
