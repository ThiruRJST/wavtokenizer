#!/usr/bin/env python3
"""
WavTokenizer — Inference Entry Point
───────────────────────────────────────

Encode a WAV to tokens OR decode tokens back to WAV.

Usage:
  # Encode + reconstruct (round-trip)
  python scripts/inference.py \
      --checkpoint checkpoints/best.pth \
      --input  audio/input.wav \
      --output audio/reconstructed.wav

  # Save tokens to disk
  python scripts/inference.py \
      --checkpoint checkpoints/best.pth \
      --input  audio/input.wav \
      --save_tokens tokens.pt

  # Decode from saved tokens
  python scripts/inference.py \
      --checkpoint checkpoints/best.pth \
      --load_tokens tokens.pt \
      --output audio/from_tokens.wav
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio

from model import build_model
from utils import load_config, load_checkpoint, get_logger

logger = get_logger("inference")


def parse_args():
    p = argparse.ArgumentParser(description="WavTokenizer Inference")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--config",       default="default")
    p.add_argument("--input",        default=None, help="Input .wav path")
    p.add_argument("--output",       default=None, help="Output .wav path")
    p.add_argument("--save_tokens",  default=None, help="Save token tensor to .pt")
    p.add_argument("--load_tokens",  default=None, help="Load token tensor from .pt")
    p.add_argument("overrides",      nargs="*")
    return p.parse_args()


def load_wav(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.mean(0, keepdim=True).unsqueeze(0)   # (1, 1, T)


@torch.no_grad()
def main():
    args   = parse_args()
    cfg    = load_config(args.config, list(args.overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr     = cfg.audio.sample_rate

    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, device=device, strict=True)
    model.to(device).eval()

    os.makedirs(cfg.inference.output_dir, exist_ok=True)

    if args.load_tokens:
        # ── Decode mode ──────────────────────────────────────────────────────
        indices = torch.load(args.load_tokens, map_location=device)
        logger.info(f"Loaded tokens: {indices.shape}")
        wav = model.decode(indices)

    elif args.input:
        # ── Encode mode ──────────────────────────────────────────────────────
        wav_in = load_wav(args.input, sr).to(device)
        logger.info(f"Input: {args.input}  {wav_in.shape}  ({wav_in.shape[-1]/sr:.2f}s)")

        indices = model.encode(wav_in)
        logger.info(f"Tokens: {indices.shape}  ({indices.shape[-1]} @ {cfg.audio.token_rate} tok/s)")

        if args.save_tokens:
            torch.save(indices.cpu(), args.save_tokens)
            logger.info(f"Tokens saved → {args.save_tokens}")

        wav = model.decode(indices)

        # SNR
        min_len = min(wav_in.shape[-1], wav.shape[-1])
        noise = wav_in[..., :min_len] - wav[..., :min_len]
        snr   = 10 * torch.log10(
            wav_in[..., :min_len].pow(2).mean()
            / noise.pow(2).mean().clamp(min=1e-9)
        )
        logger.info(f"Round-trip SNR: {snr.item():.2f} dB")

    else:
        raise ValueError("Provide --input or --load_tokens")

    if args.output:
        torchaudio.save(args.output, wav.squeeze(0).cpu(), sr)
        logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
