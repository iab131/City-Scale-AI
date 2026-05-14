#!/bin/bash
# Phase 3 R07a: STAEformer with Mixup augmentation, 2 seeds for ensemble.
set +e
cd /workspace/city-scale-ai
mkdir -p logs results/staeformer

echo "=== PHASE 3 R07a (mixup) START at $(date -u +%H:%M:%S) ==="

# 2 seeds with stronger mixup. Adding seed=42 with prob=0.3 as conservative variant.
for cfg in "42:0.3" "42:0.5" "100:0.5"; do
    SEED=${cfg%:*}; MIXUP_PROB=${cfg#*:}
    TAG="stae_R07a_mixup_p${MIXUP_PROB}_s${SEED}"
    echo "=== R07a $TAG ==="
    python3 -u scripts/train_staeformer_mixup.py \
        --tag "$TAG" --seed "$SEED" \
        --mixup_prob "$MIXUP_PROB" --mixup_alpha 0.5 \
        --batch_size 16 --epochs 100 --patience 25 \
        > "logs/${TAG}.log" 2>&1
    echo "=== R07a $TAG done at $(date -u +%H:%M:%S) ==="
done

# Final super-ensemble eval including mixup models
echo "=== final eval at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_R04_super_ensemble.py \
    --use_ttc --ttc_groups 4 \
    --stae_ckpts "results/staeformer/stae_R0*/best_stae_s*.pth" \
    --include_gwnet --include_hybrid \
    --out_json results/R04_phase3_ensemble.json \
    > logs/R04_phase3_ensemble.log 2>&1

# R08 residual stacking
echo "=== R08 residual stacking at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_R08_stacking.py \
    --stae_glob "results/staeformer/stae_*/best_stae_s*.pth" \
    --stack_hidden 32 --stack_epochs 50 \
    --out_json results/R08_stacking.json \
    > logs/R08_stacking.log 2>&1

# R09: 60-min specialist (strong horizon weighting)
echo "=== R09 60-min specialist STAEformer at $(date -u +%H:%M:%S) ==="
python3 -u scripts/train_staeformer.py \
    --tag stae_R09_h60_spec_s42 --seed 42 --batch_size 16 \
    --horizon_weighted \
    --horizon_weights 0.5 0.5 0.5 0.5 0.5 0.5 1.0 1.5 2.0 3.0 4.0 5.0 \
    --epochs 120 --patience 25 \
    > logs/R09_h60_spec.log 2>&1
echo "=== R09 done at $(date -u +%H:%M:%S) ==="

# Final ensemble with R09 added — picks per-horizon best blend
echo "=== R10 final ensemble with horizon specialist at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_R04_super_ensemble.py \
    --use_ttc --ttc_groups 4 --ttc_per_horizon \
    --stae_ckpts "results/staeformer/stae_R0*/best_stae_s*.pth" \
    --include_gwnet --include_hybrid \
    --out_json results/R10_final_ensemble.json \
    > logs/R10_final_ensemble.log 2>&1

# Phase 4 decision (conditional on best result so far)
echo "=== Phase 4 decision at $(date -u +%H:%M:%S) ==="
python3 -u scripts/decide_phase4.py > logs/phase4_decision.log 2>&1
if [ -x scripts/run_phase4.sh ]; then
    bash scripts/run_phase4.sh > logs/phase4_master.log 2>&1
fi
echo "=== Phase 4 done at $(date -u +%H:%M:%S) ==="

# Phase 5: more dropout=0.15 seeds (always runs — pure ensemble diversity)
echo "=== Phase 5 start at $(date -u +%H:%M:%S) ==="
bash scripts/run_phase5_more_seeds.sh > logs/phase5_master.log 2>&1
echo "=== Phase 5 done at $(date -u +%H:%M:%S) ==="

# Final leaderboard
echo "=== final leaderboard at $(date -u +%H:%M:%S) ==="
python3 -u scripts/show_leaderboard.py > logs/final_leaderboard.log 2>&1
echo "=== PHASE 3 DONE at $(date -u +%H:%M:%S) ==="
