"""
Light-weight sweep driver. Reads sweep file YAML and launches `train_disr.py`
sequentially or with at most one job at a time (single-GPU).

Sweep file structure:
    seeds: [0, 1, 2]
    runs:
      - name: stageB
        config: configs/disr/stage_b_temporal.yaml
      - name: stageD_q010_k48
        config: configs/disr/stage_d_magspec.yaml
        overrides:
          model:
            q_charge: 0.10
            k_modes: 48
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _materialise_override(overrides, base_config, sweep_dir):
    """Write a temporary override YAML by merging base + sweep overrides."""
    base = {}
    with open(base_config) as f:
        base = yaml.safe_load(f) or {}
    merged = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    tmp_path = os.path.join(sweep_dir, "tmp_override.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return tmp_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=str, required=True,
                   help="Sweep YAML defining runs and seeds")
    p.add_argument("--out_root", type=str, default="results/disr")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    args = p.parse_args()

    os.chdir(ROOT)
    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    seeds = list(sweep.get("seeds", [0]))
    runs = list(sweep.get("runs", []))
    print(f"[sweep] {len(runs)} runs x {len(seeds)} seeds = "
          f"{len(runs) * len(seeds)} jobs")

    os.makedirs(args.out_root, exist_ok=True)
    sweep_log = os.path.join(args.out_root, "sweep_log.csv")
    new_log = not os.path.exists(sweep_log)
    log_f = open(sweep_log, "a")
    if new_log:
        log_f.write("ts,run,seed,exit,elapsed\n")

    for r in runs:
        name = r["name"]
        cfg_path = r["config"]
        overrides = r.get("overrides", {})
        if overrides:
            cfg_path = _materialise_override(overrides, cfg_path,
                                              sweep_dir=args.out_root)
        for seed in seeds:
            tag_suffix = f"_{name}" if not cfg_path.endswith(name + ".yaml") else ""
            cmd = [
                "python3", "scripts/disr/train_disr.py",
                "--config", cfg_path,
                "--seed", str(seed),
                "--out_root", args.out_root,
                "--tag_suffix", tag_suffix,
            ]
            if args.no_compile:
                cmd.append("--no_compile")
            print(f"\n[sweep] >>> {name} seed={seed}\n    {' '.join(cmd)}")
            t0 = time.time()
            if args.dry_run:
                rc = 0
            else:
                rc = subprocess.call(cmd)
            dt = time.time() - t0
            log_f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{name},"
                        f"{seed},{rc},{dt:.1f}\n")
            log_f.flush()
            if rc != 0:
                print(f"[sweep] run {name} seed={seed} FAILED (exit {rc})")
            else:
                print(f"[sweep] run {name} seed={seed} done in {dt:.1f}s")
    log_f.close()


if __name__ == "__main__":
    main()
