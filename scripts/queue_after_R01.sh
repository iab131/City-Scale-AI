#!/bin/bash
# Wait for R01 (currently-running STAEformer training) to finish, then:
#   1. Interim 8-seed ensemble + ST-TTC eval (uses existing eval_stae_ensemble.py)
#   2. Run R02-R03 queue (big STAEformer, STMAE pretrain+finetune, super-ensemble eval)
set +e
cd /workspace/city-scale-ai

echo "=== queue_after_R01 START at $(date -u +%H:%M:%S) ==="

# Wait for any running train_staeformer.py to finish
while pgrep -f "scripts/train_staeformer.py" > /dev/null; do
    sleep 30
done
while pgrep -f "run_R01_seeds.sh" > /dev/null; do
    sleep 30
done

echo "=== R01 finished at $(date -u +%H:%M:%S) ==="

# Interim 8-seed ensemble + ST-TTC eval
echo "=== INTERIM 8-seed eval at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_stae_ensemble.py --use_ttc \
    --stae_ckpts "results/staeformer/stae_*/best_stae_s*.pth" \
    > logs/interim_8seed_eval.log 2>&1
echo "=== INTERIM 8-seed eval done at $(date -u +%H:%M:%S) ==="

# Run R02-R03 queue
bash scripts/run_R02_to_R03.sh > logs/R02_R03_queue.log 2>&1
echo "=== queue_after_R01 END at $(date -u +%H:%M:%S) ==="
