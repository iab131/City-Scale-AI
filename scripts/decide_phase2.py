"""
Decision script: read R03b test results, choose next experiment to queue.

Reads the latest row from `results/stae_pretrained/stae_pretrained_results.csv`
and writes a phase 2 queue script `scripts/run_phase2.sh` based on the decision
tree in `plans/R06_stmae_variants.txt`.
"""

import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CSV_PATH = "results/stae_pretrained/stae_pretrained_results.csv"
OUT_SCRIPT = "scripts/run_phase2.sh"


def read_latest_test_60():
    if not os.path.exists(CSV_PATH):
        return None
    try:
        with open(CSV_PATH) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None
    row = rows[-1]
    try:
        v = float(row["test_mae_60"])
        if v != v:  # NaN
            return None
        return v
    except (KeyError, ValueError):
        return None


def write_script(commands):
    script = "#!/bin/bash\nset +e\ncd /workspace/city-scale-ai\nmkdir -p logs\n"
    for cmd in commands:
        script += f"{cmd}\n"
    script += "echo '=== PHASE 2 DONE at $(date -u +%H:%M:%S) ==='\n"
    with open(OUT_SCRIPT, "w") as f:
        f.write(script)
    os.chmod(OUT_SCRIPT, 0o755)


def main():
    test60 = read_latest_test_60()
    print(f"R03b test_mae_60 = {test60}")
    commands = []
    if test60 is None:
        commands.append("echo 'R03 did not write a CSV row — skipping phase 2'")
    elif test60 <= 3.20:
        # Strong success — train 2 more pretrained STAE seeds first (cheap diversity)
        for s in (1, 2):
            commands.append(f"echo '=== R06_seed{s}: STAE+TMAE seed {s} ==='")
            commands.append(f"python3 -u scripts/finetune_stae_pretrained.py "
                            f"--tag R06_stae_pretrained_s{s} --seed {s} "
                            f"--tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth "
                            f"--batch_size 8 --epochs 80 --patience 15 --T_long 2016 --d_pre 32 "
                            f"> logs/R06_finetune_s{s}.log 2>&1")
        # Then SMAE+TMAE for orthogonal pretraining signal
        commands.append("echo '=== R06d: SMAE pretrain + TMAE+SMAE finetune ==='")
        commands.append("python3 -u scripts/pretrain_stmae.py --kind smae --tag R03_pretrain "
                        "--seed 42 --batch_size 8 --epochs 50 --patience 10 "
                        "> logs/R06d_smae_pretrain.log 2>&1")
        commands.append("python3 -u scripts/finetune_stae_pretrained.py "
                        "--tag R06d_stae_tmae_smae_s42 --seed 42 "
                        "--tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth "
                        "--smae_ckpt results/stmae/R03_pretrain/smae_best.pth "
                        "--batch_size 8 --epochs 80 --patience 15 --T_long 2016 --d_pre 48 "
                        "> logs/R06d_finetune.log 2>&1")
        commands.append("echo '=== R06a: STMAE unfrozen finetune ==='")
        commands.append("python3 -u scripts/finetune_stae_pretrained.py "
                        "--tag R06a_stae_unfrozen_s42 --seed 42 "
                        "--tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth "
                        "--smae_ckpt results/stmae/R03_pretrain/smae_best.pth "
                        "--no_freeze --batch_size 8 --epochs 60 --patience 15 --T_long 2016 --d_pre 48 "
                        "--learning_rate 5e-4 "
                        "> logs/R06a_unfrozen.log 2>&1")
    elif test60 <= 3.30:
        commands.append("echo '=== R06a: STMAE unfrozen finetune ==='")
        commands.append("python3 -u scripts/finetune_stae_pretrained.py "
                        "--tag R06a_stae_unfrozen_s42 --seed 42 "
                        "--tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth "
                        "--no_freeze --batch_size 8 --epochs 60 --patience 15 --T_long 2016 --d_pre 32 "
                        "--learning_rate 5e-4 "
                        "> logs/R06a_unfrozen.log 2>&1")
        commands.append("echo '=== R06b: bigger STMAE encoder ==='")
        commands.append("python3 -u scripts/pretrain_stmae.py --kind tmae --tag R06b_pretrain "
                        "--seed 42 --batch_size 8 --epochs 50 --patience 10 "
                        "--embed_dim 128 --encoder_depth 6 --decoder_depth 2 --num_heads 8 "
                        "> logs/R06b_pretrain.log 2>&1")
        commands.append("python3 -u scripts/finetune_stae_pretrained.py "
                        "--tag R06b_stae_big_pre_s42 --seed 42 "
                        "--tmae_ckpt results/stmae/R06b_pretrain/tmae_best.pth "
                        "--batch_size 8 --epochs 80 --patience 15 --T_long 2016 --d_pre 48 "
                        "> logs/R06b_finetune.log 2>&1")
    elif test60 <= 3.34:
        # Neutral STMAE — try unfrozen anyway. Often the frozen-vs-unfrozen gap is large
        # when the encoder was pretrained with masking but inferenced full-visibility.
        commands.append("echo '=== R06a: STMAE unfrozen finetune (neutral R03 fallback) ==='")
        commands.append("python3 -u scripts/finetune_stae_pretrained.py "
                        "--tag R06a_stae_unfrozen_s42 --seed 42 "
                        "--tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth "
                        "--no_freeze --batch_size 8 --epochs 50 --patience 12 --T_long 2016 --d_pre 32 "
                        "--learning_rate 5e-4 "
                        "> logs/R06a_unfrozen.log 2>&1")
    else:
        # R03 actively hurt — but unfrozen might still recover something. Last-shot attempt.
        commands.append(f"echo 'R03 hurt at {test60:.4f}. Trying UNFROZEN as last shot.'")
        commands.append("python3 -u scripts/finetune_stae_pretrained.py "
                        "--tag R06a_stae_unfrozen_s42 --seed 42 "
                        "--tmae_ckpt results/stmae/R03_pretrain/tmae_best.pth "
                        "--no_freeze --batch_size 8 --epochs 40 --patience 10 --T_long 2016 --d_pre 32 "
                        "--learning_rate 5e-4 "
                        "> logs/R06a_unfrozen.log 2>&1")

    # Always finish with a re-eval of all checkpoints (R04 super-ensemble)
    commands.append("echo '=== Phase 2 final eval ==='")
    commands.append("python3 -u scripts/eval_R04_super_ensemble.py --use_ttc --ttc_groups 4 "
                    "--out_json results/R04_phase2_ensemble.json "
                    "> logs/R04_phase2_ensemble.log 2>&1")
    commands.append("python3 -u scripts/eval_R05_moe_gating.py "
                    "--out_json results/R05_phase2_moe.json "
                    "> logs/R05_phase2_moe.log 2>&1")

    write_script(commands)
    print(f"Wrote {OUT_SCRIPT} with {len(commands)} commands")


if __name__ == "__main__":
    main()
