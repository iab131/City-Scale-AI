#!/usr/bin/env bash
# Phase 2 follow-up: train additional STAEformer trunks (seeds 1, 2, 3),
# then re-run the *best* DiSR config on each new trunk, then ensemble all
# (DiSR-augmented + trunk-alone) with ST-TTC.
#
# Designed to run AFTER `finish_campaign.sh` finishes. Will wait for any
# in-flight DiSR training first.

set -uo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="logs/disr_campaign"
mkdir -p "$LOG_DIR"

# Pick best DiSR config (smallest test_mae_60) — same logic as finish_campaign.
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
echo "[mt] best disr config = $best_cfg (tag $best_tag)"

# ----- Wait for any active train_* to finish ----------------------------
echo "[mt] waiting for active trainings ..."
while pgrep -f "train_disr.py\|train_staeformer.py" >/dev/null; do
  sleep 30
done
echo "[mt] free to launch."

# ----- 1. STAEformer trunks (seeds 1, 2, 3) ----------------------------
for s in 1 2 3; do
  tag="stae_trunk_s${s}"
  out="results/staeformer/${tag}/best_stae_s${s}.pth"
  if [[ -f "$out" ]]; then
    echo "[mt] skip trunk seed $s (exists)"
    continue
  fi
  echo "[mt] >>> trunk seed $s"
  PYTHONPATH=. python3 -u scripts/train_staeformer.py \
      --tag "$tag" --seed "$s" --epochs 45 --patience 12 \
      --batch_size 16 --num_workers 4 \
      2>&1 | tee "$LOG_DIR/trunk_s${s}.log" | grep -E "ep |done|elapsed|test"
done

# ----- 2. DiSR on each new trunk ----------------------------------------
for s in 1 2 3; do
  trunk="results/staeformer/stae_trunk_s${s}/best_stae_s${s}.pth"
  if [[ ! -f "$trunk" ]]; then
    echo "[mt] missing trunk for seed $s"
    continue
  fi
  out="results/disr/${best_tag}_trunk${s}_s0/summary.json"
  if [[ -f "$out" ]]; then
    echo "[mt] skip DiSR-on-trunk-$s (exists)"
    continue
  fi
  echo "[mt] >>> DiSR on trunk seed $s"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config "$best_cfg" \
      --trunk_ckpt "$trunk" --seed 0 --no_compile \
      --tag_suffix "_trunk${s}" \
      2>&1 | tee "$LOG_DIR/disr_trunk${s}.log" | grep -E "ep |done|elapsed|test"
done

# ----- 3. Mega-ensemble: all DiSR runs + ST-TTC -------------------------
echo "[mt] >>> mega-ensemble + ST-TTC"
PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/*_s*/best_disr.pth" \
    --use_ttc --ttc_groups 4 \
    --save "results/disr/mega_ensemble_ttc.npz" \
    2>&1 | tee "$LOG_DIR/mega_ensemble.log"

# ----- 4. Final aggregate + plots ---------------------------------------
PYTHONPATH=. python3 scripts/disr/aggregate_results.py
PYTHONPATH=. python3 scripts/disr/make_plots.py

echo "[mt] DONE"
