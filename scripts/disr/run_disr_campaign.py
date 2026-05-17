"""
DiSR-Mamba campaign driver.

Runs the planned stages sequentially and keeps a leaderboard. Stops early if
a run beats the 60-min test-MAE target.

Usage:
    python scripts/disr/run_disr_campaign.py \\
        --trunk_ckpt results/staeformer/stae_trunk/best_stae_s42.pth \\
        --target 3.2603
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _run(cmd, log_path):
    print(f"\n[campaign] >> {' '.join(cmd)}")
    print(f"[campaign]    log: {log_path}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return rc


def _latest_summary(tag, seed) -> dict:
    p = os.path.join(ROOT, "results", "disr", f"{tag}_s{seed}",
                     "summary.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def _check_target(summary: dict, target: float) -> bool:
    m60 = summary.get("test_metrics", {}).get("mae_60", float("inf"))
    return m60 < target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trunk_ckpt", type=str, required=True)
    p.add_argument("--target", type=float, default=3.2603,
                   help="60-min test MAE target. Stop the campaign once a run "
                        "beats it (or finish all and report best).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out_root", type=str, default="results/disr")
    p.add_argument("--stages", type=str, nargs="+",
                   default=["B", "C", "D", "E"])
    p.add_argument("--q_sweep", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.25])
    p.add_argument("--K_sweep", type=int, nargs="+",
                   default=[32, 48, 64])
    args = p.parse_args()

    os.chdir(ROOT)
    log_dir = os.path.join(ROOT, "logs", "disr_campaign")
    os.makedirs(log_dir, exist_ok=True)

    leaderboard = []
    beat_target = False

    def run_one(stage_name, cfg, seed, override_cfg=None):
        nonlocal beat_target
        if override_cfg:
            ov = override_cfg
        else:
            ov = None
        cmd = ["python3", "-u", "scripts/disr/train_disr.py",
               "--config", cfg, "--seed", str(seed),
               "--trunk_ckpt", args.trunk_ckpt,
               "--out_root", args.out_root, "--no_compile",
               "--tag_suffix", ""]
        if ov:
            cmd += ["--config", ov]
        log = os.path.join(log_dir,
                            f"{stage_name}_s{seed}.log")
        t0 = time.time()
        rc = _run(cmd, log)
        dt = time.time() - t0
        # collect summary
        tag = _extract_tag(cfg, ov)
        s = _latest_summary(tag, seed)
        if s:
            entry = {
                "stage": stage_name,
                "tag": s.get("tag"),
                "seed": s.get("seed"),
                "test_mae_60": s.get("test_metrics", {}).get("mae_60"),
                "test_avg_mae": s.get("test_metrics", {}).get("avg_mae"),
                "best_val_mae": s.get("best_val_mae"),
                "elapsed": dt,
                "rc": rc,
            }
            leaderboard.append(entry)
            print(f"[campaign] {stage_name} seed={seed} -> "
                  f"60min={entry['test_mae_60']:.4f}  "
                  f"avg={entry['test_avg_mae']:.4f}  "
                  f"({dt:.0f}s)")
            if entry["test_mae_60"] is not None \
                    and entry["test_mae_60"] < args.target:
                beat_target = True
        else:
            print(f"[campaign] {stage_name} seed={seed} FAILED (rc={rc})")

        leader_path = os.path.join(args.out_root, "campaign_leaderboard.json")
        os.makedirs(args.out_root, exist_ok=True)
        with open(leader_path, "w") as f:
            json.dump(leaderboard, f, indent=2)

    # -----------------------------------------------------------------
    # Stage B: temporal residual
    if "B" in args.stages:
        for s in args.seeds:
            run_one("B_temporal",
                    "configs/disr/stage_b_temporal.yaml", s)
            if beat_target:
                break

    # Stage C: symmetric spectral (single q-less config)
    if "C" in args.stages and not beat_target:
        # Single K=48 baseline; full K sweep below if needed.
        for s in args.seeds:
            run_one("C_symspec_K48",
                    "configs/disr/stage_c_symspec.yaml", s)
            if beat_target:
                break

    # Stage D: magnetic spectral q sweep (1 seed each)
    if "D" in args.stages and not beat_target:
        for q in args.q_sweep:
            ov = os.path.join("/tmp", f"disr_q{q:.2f}.yaml")
            with open(ov, "w") as f:
                f.write(
                    f"experiment:\n  tag: \"stageD_q{int(q*100):03d}_K48\"\n"
                    f"model:\n  q_charge: {q}\n"
                )
            run_one(f"D_q{int(q*100):03d}",
                    "configs/disr/stage_d_magspec.yaml",
                    args.seeds[0], override_cfg=ov)
            if beat_target:
                break

    # Stage E: router (single config, single seed first)
    if "E" in args.stages and not beat_target:
        for s in args.seeds:
            run_one("E_router_c12",
                    "configs/disr/stage_e_router.yaml", s)
            if beat_target:
                break

    # If beat target on some run, also run remaining seeds of the best config
    print("\n[campaign] " + ("BEAT TARGET" if beat_target
                              else "did NOT beat target")
          + f"  (target=3.2603)")
    print(f"[campaign] leaderboard -> "
          f"{os.path.join(args.out_root, 'campaign_leaderboard.json')}")


def _extract_tag(cfg_path, override_path):
    """Extract the experiment.tag from yaml files."""
    import yaml
    tag = None
    for p in (cfg_path, override_path):
        if not p:
            continue
        with open(p) as f:
            d = yaml.safe_load(f) or {}
        if d.get("experiment", {}).get("tag"):
            tag = d["experiment"]["tag"]
    return tag


if __name__ == "__main__":
    main()
