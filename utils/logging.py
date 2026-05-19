"""
Logging utilities:
  • Python logger with rich formatting
  • TensorBoard SummaryWriter wrapper
  • Optional W&B integration (guarded import)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from omegaconf import DictConfig


# ─── Console Logger ──────────────────────────────────────────────────────────

def get_logger(name: str = "wavtokenizer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:                     # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ─── TensorBoard ─────────────────────────────────────────────────────────────

class TBWriter:
    """
    Thin wrapper around SummaryWriter.
    Falls back to a no-op if tensorboard is not installed.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self._writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[TBWriter] tensorboard not installed — skipping.")

    def scalar(self, tag: str, value: float, step: int):
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for k, v in metrics.items():
            self.scalar(f"{prefix}/{k}" if prefix else k, v, step)

    def audio(self, tag: str, wav, sample_rate: int, step: int):
        """wav: (1, T) or (T,) tensor"""
        if self._writer:
            self._writer.add_audio(tag, wav, step, sample_rate=sample_rate)

    def close(self):
        if self._writer:
            self._writer.close()


# ─── Weights & Biases (optional) ─────────────────────────────────────────────

class WandbLogger:
    """Optional W&B logger. Gracefully disabled if wandb not installed."""

    def __init__(self, cfg: DictConfig):
        self.enabled = cfg.project.wandb
        self._run = None
        if self.enabled:
            try:
                import wandb
                self._run = wandb.init(
                    project=cfg.project.wandb_project,
                    entity=cfg.project.wandb_entity,
                    name=cfg.project.run_name,
                    config=dict(cfg),
                )
            except Exception as e:
                print(f"[WandbLogger] Could not init: {e}")
                self.enabled = False

    def log(self, metrics: Dict[str, float], step: int):
        if self._run:
            self._run.log(metrics, step=step)

    def finish(self):
        if self._run:
            self._run.finish()
