"""
Checkpoint management:
  save_checkpoint  — snapshot model + optimizers + scaler + metadata
  load_checkpoint  — restore from disk (strict or relaxed)
  prune_checkpoints — keep only the N most recent files
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    step: int,
    model: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    sched_g: Any,
    sched_d: Any,
    scaler_g: Optional[Any] = None,
    scaler_d: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    cfg_dict: Optional[dict] = None,
    keep_last_n: int = 5,
    is_best: bool = False,
) -> str:
    """Save a full training checkpoint and prune old ones."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    fname = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch:04d}_step{step:07d}.pth")

    payload = {
        "epoch":    epoch,
        "step":     step,
        "model":    model.state_dict(),
        "opt_g":    opt_g.state_dict(),
        "opt_d":    opt_d.state_dict(),
        "sched_g":  sched_g.state_dict() if sched_g else None,
        "sched_d":  sched_d.state_dict() if sched_d else None,
        "scaler_g": scaler_g.state_dict() if scaler_g else None,
        "scaler_d": scaler_d.state_dict() if scaler_d else None,
        "metrics":  metrics or {},
        "config":   cfg_dict or {},
    }
    torch.save(payload, fname)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(payload, best_path)

    prune_checkpoints(checkpoint_dir, keep_last_n)
    return fname


def load_checkpoint(
    path: str,
    model: nn.Module,
    opt_g: Optional[torch.optim.Optimizer] = None,
    opt_d: Optional[torch.optim.Optimizer] = None,
    sched_g: Optional[Any] = None,
    sched_d: Optional[Any] = None,
    scaler_g: Optional[Any] = None,
    scaler_d: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Restore a checkpoint.  Returns the metadata dict so callers can
    resume epoch / step counters.
    """
    ckpt = torch.load(path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if missing:
        print(f"[load_checkpoint] Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    def _restore(obj, key):
        if obj is not None and ckpt.get(key):
            obj.load_state_dict(ckpt[key])

    _restore(opt_g,    "opt_g")
    _restore(opt_d,    "opt_d")
    _restore(sched_g,  "sched_g")
    _restore(sched_d,  "sched_d")
    _restore(scaler_g, "scaler_g")
    _restore(scaler_d, "scaler_d")

    print(f"[load_checkpoint] Loaded epoch={ckpt['epoch']} step={ckpt['step']} ← {path}")
    return {"epoch": ckpt["epoch"], "step": ckpt["step"], "metrics": ckpt.get("metrics", {})}


def prune_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """Delete oldest checkpoints, keeping only the N most recent."""
    pattern = os.path.join(checkpoint_dir, "ckpt_epoch*.pth")
    files = sorted(glob.glob(pattern))
    for old in files[:-keep_last_n]:
        os.remove(old)
