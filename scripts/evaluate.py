#!/usr/bin/env python3
"""
WavTokenizer — Evaluation Entry Point
───────────────────────────────────────

Usage:
  python scripts/evaluate.py \
      --checkpoint checkpoints/best.pth \
      --config default \
      --val_dir /path/to/val_wavs
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data        import build_dataloaders
from model       import build_model
from evaluation  import Evaluator
from utils       import load_config, load_checkpoint, get_logger

logger = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WavTokenizer")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config",     default="default")
    p.add_argument("overrides",    nargs="*")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config, list(args.overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, device=device, strict=True)
    model.to(device).eval()

    _, val_loader = build_dataloaders(cfg)

    ev = Evaluator(cfg, model, val_loader, device)
    metrics = ev.run()
    return metrics


if __name__ == "__main__":
    main()
