#!/bin/bash
# Queue: R02 (big STAEformer) → R03a (TMAE pretrain) → R03b (STAEformer + TMAE finetune)
#        → R04 (super-ensemble eval) → R05 (MoE gating eval)
# Each step's failure is logged but doesn't stop the queue.
set +e
cd /workspace/city-scale-ai
mkdir -p logs results

echo "=== QUEUE START at $(date -u +%H:%M:%S) ==="

# --- R02: big STAEformer ---
TAG="stae_R02_big_s42"
echo "=== R02 launch $TAG at $(date -u +%H:%M:%S) ==="
python3 -u scripts/train_staeformer.py \
    --tag "$TAG" --seed 42 --batch_size 16 \
    --input_embedding_dim 32 --tod_embedding_dim 32 --dow_embedding_dim 32 \
    --adaptive_embedding_dim 96 \
    --feed_forward_dim 384 --num_layers 4 \
    --dropout 0.15 --weight_decay 5e-4 \
    --lr_milestones 25 40 \
    --epochs 200 --patience 30 \
    > "logs/${TAG}.log" 2>&1
echo "=== R02 done $TAG at $(date -u +%H:%M:%S) ==="

# --- R03a: TMAE pretrain ---
echo "=== R03a TMAE pretrain at $(date -u +%H:%M:%S) ==="
python3 -u scripts/pretrain_stmae.py \
    --kind tmae --tag R03_pretrain --seed 42 \
    --batch_size 8 --epochs 50 --patience 10 \
    --T_long 2016 --patch_size 12 --max_patches 168 \
    --embed_dim 96 --num_heads 4 --ffn_mult 4 \
    --encoder_depth 4 --decoder_depth 1 \
    --mask_ratio 0.75 --learning_rate 5e-4 \
    > "logs/R03a_tmae.log" 2>&1
echo "=== R03a done at $(date -u +%H:%M:%S) ==="

# --- R03b: STAEformer + frozen TMAE finetune ---
if [ -f results/stmae/R03_pretrain/tmae_best.pth ]; then
    echo "=== R03b STAEformer-pretrained finetune at $(date -u +%H:%M:%S) ==="
    python3 -u scripts/finetune_stae_pretrained.py \
        --tag R03_stae_pretrained_s42 --seed 42 \
        --tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth \
        --batch_size 8 --epochs 80 --patience 15 \
        --T_long 2016 --d_pre 32 \
        > "logs/R03b_finetune.log" 2>&1
    echo "=== R03b done at $(date -u +%H:%M:%S) ==="
else
    echo "=== R03b SKIPPED — TMAE checkpoint missing at $(date -u +%H:%M:%S) ==="
fi

# --- R04: super-ensemble eval (with ST-TTC v2), include GWNet + Hybrid ---
echo "=== R04 super-ensemble eval at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_R04_super_ensemble.py \
    --use_ttc --ttc_groups 4 \
    --include_gwnet --include_hybrid \
    > "logs/R04_super_ensemble.log" 2>&1
echo "=== R04 done at $(date -u +%H:%M:%S) ==="

# Variant with per-horizon TTC + 8 groups
echo "=== R04b super-ensemble eval (TTC ph + 8 groups) at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_R04_super_ensemble.py \
    --use_ttc --ttc_groups 8 --ttc_per_horizon \
    --include_gwnet --include_hybrid \
    --out_json results/R04b_super_ensemble.json \
    > "logs/R04b_super_ensemble.log" 2>&1
echo "=== R04b done at $(date -u +%H:%M:%S) ==="

# --- R05: MoE gating eval (include all architectures) ---
echo "=== R05 MoE gating eval at $(date -u +%H:%M:%S) ==="
python3 -u scripts/eval_R05_moe_gating.py \
    --include_gwnet --include_hybrid \
    > "logs/R05_moe_gating.log" 2>&1
echo "=== R05 done at $(date -u +%H:%M:%S) ==="

# --- Phase 2: decide and run conditional variants ---
echo "=== Phase 2 decision at $(date -u +%H:%M:%S) ==="
python3 -u scripts/decide_phase2.py > logs/phase2_decision.log 2>&1
if [ -x scripts/run_phase2.sh ]; then
    bash scripts/run_phase2.sh > logs/phase2_master.log 2>&1
fi
echo "=== Phase 2 done at $(date -u +%H:%M:%S) ==="

# --- Phase 3 R07a: mixup-augmented STAEformer (always runs) ---
echo "=== Phase 3 R07a (mixup) at $(date -u +%H:%M:%S) ==="
bash scripts/run_phase3_mixup.sh > logs/phase3_mixup_master.log 2>&1
echo "=== Phase 3 R07a done at $(date -u +%H:%M:%S) ==="

echo "=== QUEUE END at $(date -u +%H:%M:%S) ==="
