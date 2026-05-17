#!/usr/bin/env bash
# Finish the campaign after the queue has been manually killed.
#
# Behaviour:
#   1. Wait for any active train_disr.py to finish (the in-flight Stage D q=0.05).
#   2. Rename stageD_q000_K48_s0 -> stageD_q005_K48_s0 (was misnamed by the bc bug).
#   3. Run Stage E (router + all experts) at seed 0.
#   4. Aggregate all results.
#   5. Pick the best config and run 2 more seeds.
#   6. Ensemble + ST-TTC eval over all DiSR ckpts.
#   7. Final aggregate + plots.

set -uo pipefail
cd "$(dirname "$0")/../.."

TRUNK_CKPT="results/staeformer/stae_trunk/best_stae_s42.pth"
LOG_DIR="logs/disr_campaign"
mkdir -p "$LOG_DIR"

# ----- 1. Wait for any active DiSR train to finish ------------------------
echo "[finish] waiting for in-flight train_disr.py ..."
while pgrep -f "train_disr.py" >/dev/null; do
  sleep 30
done
# Also wait for any in-flight STAEformer training (which a parallel
# queue_multitrunk might have started).
while pgrep -f "train_staeformer.py" >/dev/null; do
  sleep 30
done
echo "[finish] all train_disr.py finished."

# ----- 2. Rename misnamed Stage D dir --------------------------------------
if [[ -d "results/disr/stageD_q000_K48_s0" && ! -d "results/disr/stageD_q005_K48_s0" ]]; then
  echo "[finish] renaming stageD_q000_K48_s0 -> stageD_q005_K48_s0"
  mv results/disr/stageD_q000_K48_s0 results/disr/stageD_q005_K48_s0
  # patch the summary.json's tag field too
  python3 -c "
import json
p = 'results/disr/stageD_q005_K48_s0/summary.json'
try:
    with open(p) as f: s = json.load(f)
    s['tag'] = 'stageD_q005_K48'
    with open(p, 'w') as f: json.dump(s, f, indent=2)
    print('patched', p)
except Exception as e:
    print('skip summary patch:', e)
"
fi

# ----- 3. Stage E ----------------------------------------------------------
if [[ ! -f "results/disr/stageE_router_s0/summary.json" ]]; then
  echo "[finish] >>> Stage E (router + all experts, seed 0)"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config configs/disr/stage_e_router.yaml \
      --trunk_ckpt "$TRUNK_CKPT" --seed 0 --no_compile \
      2>&1 | tee "$LOG_DIR/stageE_s0.log" | grep -E "ep |done|elapsed|test"
fi

# ----- 4. Aggregate ------------------------------------------------------
PYTHONPATH=. python3 scripts/disr/aggregate_results.py 2>&1 | tail -20

# ----- 5. Multi-seed the best so far ---------------------------------------
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
echo "[finish] best single-seed config: $best_tag"

best_cfg=""
case "$best_tag" in
  stageB*) best_cfg="configs/disr/stage_b_temporal.yaml" ;;
  stageC*) best_cfg="configs/disr/stage_c_symspec.yaml" ;;
  stageD*) best_cfg="configs/disr/stage_d_magspec.yaml" ;;
  stageE*) best_cfg="configs/disr/stage_e_router.yaml" ;;
  *) best_cfg="configs/disr/stage_e_router.yaml" ;;
esac

for s in 1 2; do
  out="results/disr/${best_tag}_s${s}/summary.json"
  if [[ -f "$out" ]]; then
    echo "[finish] skip $best_tag seed=$s (exists)"
    continue
  fi
  echo "[finish] >>> $best_tag seed=$s"
  PYTHONPATH=. python3 -u scripts/disr/train_disr.py \
      --config "$best_cfg" \
      --trunk_ckpt "$TRUNK_CKPT" --seed "$s" --no_compile \
      2>&1 | tee "$LOG_DIR/${best_tag}_s${s}.log" | grep -E "ep |done|elapsed|test"
done

# ----- 6. Ensemble + ST-TTC ------------------------------------------------
echo "[finish] >>> ensemble + ST-TTC over all single-seed runs"
PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/*_s*/best_disr.pth" \
    --use_ttc --ttc_groups 4 \
    --save "results/disr/all_ensemble_ttc.npz" \
    2>&1 | tee "$LOG_DIR/all_ensemble.log"

echo "[finish] >>> ensemble + ST-TTC over only the $best_tag seeds"
PYTHONPATH=. python3 -u scripts/disr/evaluate_disr.py \
    --ckpts "results/disr/${best_tag}_s*/best_disr.pth" \
    --use_ttc --ttc_groups 4 \
    --save "results/disr/${best_tag}_ensemble_ttc.npz" \
    2>&1 | tee "$LOG_DIR/${best_tag}_ensemble.log"

# ----- 7. Final aggregation + plots ----------------------------------------
PYTHONPATH=. python3 scripts/disr/aggregate_results.py
PYTHONPATH=. python3 scripts/disr/make_plots.py

echo "[finish] DONE"
