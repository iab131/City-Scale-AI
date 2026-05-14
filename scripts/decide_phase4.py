"""
Final decision: scan all ensemble results, decide if we need Phase 4.

Reads all results/R0*.json + the leaderboard. If the best 60-min MAE is:
  ≤ 3.20  : done, beating SOTA cleanly.
  3.20-3.27: try Phase 4 R11 (more pretrained STAE seeds + longer T_long)
  > 3.27  : something's stuck; queue a "diagnostic" run for human review.

Writes scripts/run_phase4.sh (or "PHASE4_DONE" if no further action needed).
"""
import os
import sys
import glob
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT_SCRIPT = "scripts/run_phase4.sh"


def find_best_mae_60():
    """Find the minimum 60-min MAE across all ensemble JSON results."""
    best = float("inf")
    best_tag = None
    for path in glob.glob("results/R*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        for k in ("ensembles", "stacked", "moe_gated", "base"):
            v = d.get(k)
            if v is None:
                continue
            if k == "ensembles":
                for name, m in v.items():
                    v60 = m.get("mae_60")
                    if v60 and v60 < best:
                        best = v60; best_tag = f"{path}:{name}"
            elif isinstance(v, dict) and "mae_60" in v:
                v60 = v["mae_60"]
                if v60 and v60 < best:
                    best = v60; best_tag = f"{path}:{k}"
    return best, best_tag


def main():
    best, tag = find_best_mae_60()
    print(f"Best ensemble 60-min MAE so far: {best:.4f} ({tag})")

    commands = ["#!/bin/bash", "set +e", "cd /workspace/city-scale-ai", "mkdir -p logs"]
    if best <= 3.20:
        commands.append(f"echo 'PHASE 4 SKIP — best={best:.4f} already at/below 3.20'")
    elif best <= 3.27:
        # Try alternative pretraining variants
        commands.append(f"echo '=== PHASE 4 (best={best:.4f}): pretraining variants ==='")
        # R11: mask_ratio=0.5 (easier reconstruction)
        commands.append(
            "python3 -u scripts/pretrain_stmae.py --kind tmae --tag R11_pretrain_mask50 "
            "--seed 42 --batch_size 8 --epochs 50 --patience 10 --mask_ratio 0.5 "
            "--T_long 2016 --embed_dim 96 --encoder_depth 4 --decoder_depth 1 "
            "> logs/R11_pretrain.log 2>&1"
        )
        commands.append(
            "python3 -u scripts/finetune_stae_pretrained.py "
            "--tag R11_stae_mask50_s42 --seed 42 "
            "--tmae_ckpt results/stmae/R11_pretrain_mask50/tmae_best.pth "
            "--batch_size 8 --epochs 60 --patience 15 --T_long 2016 --d_pre 32 "
            "> logs/R11_finetune.log 2>&1"
        )
        # R12: bigger encoder
        commands.append(
            "python3 -u scripts/pretrain_stmae.py --kind tmae --tag R12_pretrain_big "
            "--seed 42 --batch_size 4 --epochs 40 --patience 8 "
            "--embed_dim 128 --encoder_depth 6 --decoder_depth 2 --num_heads 8 "
            "> logs/R12_pretrain.log 2>&1"
        )
        commands.append(
            "python3 -u scripts/finetune_stae_pretrained.py "
            "--tag R12_stae_big_pre_s42 --seed 42 "
            "--tmae_ckpt results/stmae/R12_pretrain_big/tmae_best.pth "
            "--batch_size 8 --epochs 60 --patience 15 --T_long 2016 --d_pre 48 "
            "> logs/R12_finetune.log 2>&1"
        )
        commands.append(
            "python3 -u scripts/eval_R04_super_ensemble.py "
            "--use_ttc --ttc_groups 4 --ttc_per_horizon "
            "--include_gwnet --include_hybrid "
            "--out_json results/R11_final.json "
            "> logs/R11_final_eval.log 2>&1"
        )
    else:
        commands.append(f"echo 'PHASE 4 DIAGNOSTIC — best={best:.4f} > 3.27, something stalled'")
        commands.append("python3 -u scripts/show_leaderboard.py > logs/diagnostic_leaderboard.log 2>&1")

    commands.append("python3 -u scripts/show_leaderboard.py > logs/final_leaderboard.log 2>&1")
    commands.append("echo '=== PHASE 4 DONE ==='")
    with open(OUT_SCRIPT, "w") as f:
        f.write("\n".join(commands) + "\n")
    os.chmod(OUT_SCRIPT, 0o755)
    print(f"Wrote {OUT_SCRIPT}")


if __name__ == "__main__":
    main()
