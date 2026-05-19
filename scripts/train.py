#!/usr/bin/env python3
"""
WavTokenizer — Training Entry Point
─────────────────────────────────────

Usage examples:

  # Full training with default config
  python scripts/train.py

  # Use small/debug config
  python scripts/train.py --config small

  # Override any config key via dotlist
  python scripts/train.py --config default \
      training.batch_size=8         \
      training.epochs=500           \
      audio.token_rate=40           \
      project.run_name=my_run

  # Resume from checkpoint
  python scripts/train.py --resume checkpoints/ckpt_epoch0010_step0001234.pth

  # Multi-GPU with torchrun (DDP)
  torchrun --nproc_per_node=4 scripts/train.py --config default
"""

import argparse
import os
import sys

# Allow `import model`, `import training`, etc. from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from data      import build_dataloaders
from model     import (
    build_model,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiResolutionSTFTDiscriminator,
)
from training  import Trainer
from utils     import load_config, seed_everything, count_parameters, get_logger

logger = get_logger("train")


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train WavTokenizer")
    p.add_argument("--config",  default="default",
                   help="Config name (without .yaml) in configs/")
    p.add_argument("--resume",  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--run_name", default=None,
                   help="Override project.run_name")
    # Catch-all for dotlist overrides: e.g. training.batch_size=8
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


# ─── DDP helpers ─────────────────────────────────────────────────────────────

def _is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def _setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    overrides = list(args.overrides)
    if args.run_name:
        overrides.append(f"project.run_name={args.run_name}")

    cfg = load_config(args.config, overrides)

    # ── DDP / Single-GPU / CPU device setup ───────────────────────────────────
    ddp = _is_ddp()
    if ddp:
        local_rank = _setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        is_main = True
    else:
        device = torch.device("cpu")
        is_main = True

    if is_main:
        logger.info(f"Run: {cfg.project.run_name}  |  device: {device}")
        logger.info(f"Config:\n{cfg}")

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed_everything(cfg.project.seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    if is_main:
        logger.info("Building datasets …")
    train_loader, val_loader = build_dataloaders(cfg)
    if is_main:
        logger.info(
            f"  train={len(train_loader.dataset)} samples  "
            f"val={len(val_loader.dataset)} samples"
        )

    # ── Model + Discriminators ────────────────────────────────────────────────
    model   = build_model(cfg)
    mpd     = MultiPeriodDiscriminator.from_config(cfg)
    msd     = MultiScaleDiscriminator.from_config(cfg)
    mrstftd = MultiResolutionSTFTDiscriminator.from_config(cfg)

    if is_main:
        logger.info(
            f"Parameters  "
            f"WavTokenizer={count_parameters(model):,}  "
            f"MPD={count_parameters(mpd):,}  "
            f"MSD={count_parameters(msd):,}  "
            f"MRSTFTD={count_parameters(mrstftd):,}"
        )

    # ── Wrap in DDP ───────────────────────────────────────────────────────────
    if ddp:
        model   = torch.nn.parallel.DistributedDataParallel(model,   device_ids=[local_rank])
        mpd     = torch.nn.parallel.DistributedDataParallel(mpd,     device_ids=[local_rank])
        msd     = torch.nn.parallel.DistributedDataParallel(msd,     device_ids=[local_rank])
        mrstftd = torch.nn.parallel.DistributedDataParallel(mrstftd, device_ids=[local_rank])

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg          = cfg,
        model        = model,
        mpd          = mpd,
        msd          = msd,
        mrstftd      = mrstftd,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        resume_path  = args.resume,
    )

    trainer.fit()
    _cleanup_ddp()


if __name__ == "__main__":
    main()
