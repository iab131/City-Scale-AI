"""Shared helpers for DiSR-Mamba train / eval scripts."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# Make project root importable so we can `import models.disr ...`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess_v2 import get_cached_v2_data  # noqa: E402
from dataset_v2 import SSSMDataset, split_train_val_test  # noqa: E402


# ----------------------------------------------------------------------------
def load_config(base_path: str,
                override_path: Optional[Any] = None,
                cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Recursively merge: base yaml + optional override yaml(s) + CLI overrides.

    `override_path` can be a single path or a list of paths; later entries take
    precedence.
    """
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if override_path:
        paths = override_path if isinstance(override_path, (list, tuple)) else [override_path]
        for p in paths:
            if not p:
                continue
            with open(p, "r") as f:
                ov = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, ov)
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)
    return cfg


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit_hash() -> str:
    try:
        h = subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", ROOT, "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------------------
def get_amp_dtype() -> torch.dtype:
    """bf16 on supporting GPUs, otherwise fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ----------------------------------------------------------------------------
def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any, dict, Dict[str, Any]]:
    """Returns (train_loader, val_loader, test_loader, data_dict, splits_info)."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]

    data = get_cached_v2_data(
        data_cfg["data_path"], data_cfg["adj_path"],
        k=data_cfg.get("k_eig", 207),
        cache_dir=data_cfg["cache_dir"],
    )
    X = data["X"]; X_norm = data["X_norm"]
    tod = data["tod"]; dow = data["dow"]; mask = data["missing_mask"]

    arrs = split_train_val_test([X, X_norm, tod, dow, mask], 0.7, 0.1)
    (X_tr, X_va, X_te), (Xn_tr, Xn_va, Xn_te), \
        (tod_tr, tod_va, tod_te), (dow_tr, dow_va, dow_te), \
        (mk_tr, mk_va, mk_te) = arrs

    def mk(X_p, Xn_p, tod_p, dow_p, mk_p, shuffle, bs):
        ds = SSSMDataset(X_p, Xn_p, tod_p, dow_p, mk_p,
                         input_len=data_cfg["in_steps"],
                         pred_len=data_cfg["out_steps"])
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True, drop_last=False,
        )

    tr = mk(X_tr, Xn_tr, tod_tr, dow_tr, mk_tr, True, train_cfg["batch_size"])
    va = mk(X_va, Xn_va, tod_va, dow_va, mk_va, False, eval_cfg["batch_size"])
    te = mk(X_te, Xn_te, tod_te, dow_te, mk_te, False, eval_cfg["batch_size"])

    splits = {
        "n_train": len(tr.dataset),
        "n_val":   len(va.dataset),
        "n_test":  len(te.dataset),
        "train_X": X_tr,
    }
    return tr, va, te, data, splits


# ----------------------------------------------------------------------------
def save_run_metadata(out_dir: str, cfg: Dict[str, Any], extras: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {
        "git_commit": git_commit_hash(),
        "timestamp": now_iso(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "amp_dtype": str(get_amp_dtype()),
        **extras,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
