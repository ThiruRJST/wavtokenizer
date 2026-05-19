"""
WavTokenizer Evaluator
───────────────────────
Computes objective metrics on a dataset:
  • Multi-scale Mel-spectrogram L1 (MelLoss)
  • Signal-to-Noise Ratio (SNR, dB)
  • Codebook utilization  (% of codes used)
  • PESQ (optional — requires pesq package)
  • STOI (optional — requires pystoi package)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from model import WavTokenizer
from losses import MelSpectrogramLoss
from utils import get_logger, AverageMeter


logger = get_logger("evaluator")


def _snr_db(real: torch.Tensor, fake: torch.Tensor, eps: float = 1e-9) -> float:
    min_len = min(real.shape[-1], fake.shape[-1])
    r = real[..., :min_len]
    f = fake[..., :min_len]
    noise = r - f
    snr = 10 * torch.log10(
        r.pow(2).mean() / noise.pow(2).mean().clamp(min=eps)
    )
    return snr.item()


class Evaluator:
    """
    Runs full evaluation pass over a DataLoader and reports metrics.

    Usage:
        ev = Evaluator(cfg, model, dataloader, device)
        metrics = ev.run()
    """

    def __init__(
        self,
        cfg:        DictConfig,
        model:      WavTokenizer,
        dataloader: DataLoader,
        device:     torch.device,
    ):
        self.cfg        = cfg
        self.model      = model.to(device)
        self.dataloader = dataloader
        self.device     = device
        self.mel_fn     = MelSpectrogramLoss.from_config(cfg).to(device)

        # Optional PESQ / STOI
        self._has_pesq  = self._check_import("pesq")
        self._has_stoi  = self._check_import("pystoi")

    @staticmethod
    def _check_import(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    @torch.no_grad()
    def run(self) -> Dict[str, float]:
        self.model.eval()

        meters: Dict[str, AverageMeter] = {
            "mel_loss":       AverageMeter(),
            "snr_db":         AverageMeter(),
            "codebook_util":  AverageMeter(),
        }
        if self._has_pesq:
            meters["pesq"] = AverageMeter()
        if self._has_stoi:
            meters["stoi"] = AverageMeter()

        for batch in self.dataloader:
            real = batch.to(self.device)                        # (B, 1, T)
            fake, _ = self.model(real)

            min_len = min(real.shape[-1], fake.shape[-1])
            r = real[..., :min_len]
            f = fake[..., :min_len]
            B = r.shape[0]

            # ── Mel loss ─────────────────────────────────────────────────────
            mel = self.mel_fn(r, f)
            meters["mel_loss"].update(mel.item(), B)

            # ── SNR ──────────────────────────────────────────────────────────
            snr = _snr_db(r, f)
            meters["snr_db"].update(snr, B)

            # ── Codebook utilization ─────────────────────────────────────────
            indices = self.model.encode(real)
            util = (
                torch.unique(indices).numel() / self.model.quantizer.K * 100
            )
            meters["codebook_util"].update(util, B)

            # ── PESQ (optional) ───────────────────────────────────────────────
            if self._has_pesq:
                from pesq import pesq as _pesq
                sr = self.cfg.audio.sample_rate
                for b in range(B):
                    try:
                        score = _pesq(
                            sr,
                            r[b, 0].cpu().numpy(),
                            f[b, 0].cpu().numpy(),
                            "wb",
                        )
                        meters["pesq"].update(score)
                    except Exception:
                        pass

            # ── STOI (optional) ───────────────────────────────────────────────
            if self._has_stoi:
                from pystoi import stoi as _stoi
                sr = self.cfg.audio.sample_rate
                for b in range(B):
                    try:
                        score = _stoi(
                            r[b, 0].cpu().numpy(),
                            f[b, 0].cpu().numpy(),
                            sr,
                            extended=False,
                        )
                        meters["stoi"].update(score)
                    except Exception:
                        pass

        results = {k: m.avg for k, m in meters.items()}
        self._print_results(results)
        return results

    @staticmethod
    def _print_results(metrics: Dict[str, float]):
        logger.info("═" * 50)
        logger.info("  Evaluation Results")
        logger.info("═" * 50)
        for k, v in metrics.items():
            logger.info(f"  {k:<20} {v:.4f}")
        logger.info("═" * 50)
