"""
WavTokenizer Loss Functions
────────────────────────────
  • hinge_loss_discriminator  — real/fake hinge objective for D
  • hinge_loss_generator      — adversarial hinge objective for G
  • feature_matching_loss     — L1 distance between D's internal activations
  • MelSpectrogramLoss        — multi-scale log-mel L1 reconstruction loss
  • TotalGeneratorLoss        — weighted sum of all G losses
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from omegaconf import DictConfig


# ─── Adversarial losses (hinge formulation) ──────────────────────────────────

def hinge_loss_discriminator(
    real_outs: List[torch.Tensor],
    fake_outs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Discriminator hinge loss:
        L_D = E[ReLU(1 - D(x))] + E[ReLU(1 + D(G(z)))]

    Averaged over all sub-discriminators.
    """
    loss = torch.tensor(0.0, device=real_outs[0].device)
    for r, f in zip(real_outs, fake_outs):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / len(real_outs)


def hinge_loss_generator(fake_outs: List[torch.Tensor]) -> torch.Tensor:
    """
    Generator adversarial hinge loss:
        L_G_adv = -E[D(G(z))]

    Averaged over all sub-discriminators.
    """
    loss = torch.tensor(0.0, device=fake_outs[0].device)
    for f in fake_outs:
        loss = loss + (-f.mean())
    return loss / len(fake_outs)


# ─── Feature-matching loss ───────────────────────────────────────────────────

def feature_matching_loss(
    real_fmaps: List[List[torch.Tensor]],
    fake_fmaps: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 distance between the discriminator's intermediate activations
    for real vs. fake audio.

    Provides a perceptually grounded gradient signal independent of whether
    the discriminator correctly classifies real vs. fake.
    """
    loss  = torch.tensor(0.0, device=real_fmaps[0][0].device)
    count = 0
    for rf_list, ff_list in zip(real_fmaps, fake_fmaps):
        for rf, ff in zip(rf_list, ff_list):
            loss  = loss + F.l1_loss(ff, rf.detach())
            count += 1
    return loss / max(count, 1)


# ─── Multi-scale Mel-spectrogram loss ────────────────────────────────────────

class MelSpectrogramLoss(nn.Module):
    """
    Log-mel L1 reconstruction loss computed at multiple FFT scales.
    Using multiple scales (32 → 2048) ensures both fine-grained temporal
    details and coarse spectral envelope are captured.
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        scales: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
        n_mels:     int = 80,
        log_offset: float = 1e-5,
    ):
        super().__init__()
        self.log_offset = log_offset
        self.transforms = nn.ModuleList([
            T.MelSpectrogram(
                sample_rate = sample_rate,
                n_fft       = n,
                hop_length  = max(n // 4, 1),
                n_mels      = n_mels,
            )
            for n in scales
        ])

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MelSpectrogramLoss":
        return cls(
            sample_rate = cfg.audio.sample_rate,
            scales      = tuple(cfg.loss.mel_scales),
        )

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # Trim to same length
        min_len = min(real.shape[-1], fake.shape[-1])
        real = real[..., :min_len].squeeze(1)   # (B, T)
        fake = fake[..., :min_len].squeeze(1)

        loss = torch.tensor(0.0, device=real.device)
        for mel in self.transforms:
            mel = mel.to(real.device)
            r = mel(real).clamp(min=self.log_offset).log()
            f = mel(fake).clamp(min=self.log_offset).log()
            loss = loss + F.l1_loss(f, r)
        return loss / len(self.transforms)


# ─── Total generator loss ────────────────────────────────────────────────────

class TotalGeneratorLoss(nn.Module):
    """
    Weighted combination of all generator-side losses:

        L_G = λ_adv · L_adv
            + λ_fm  · L_fm
            + λ_mel · L_mel
            + λ_vq  · L_vq

    Default weights are those recommended by the WavTokenizer paper.
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_fm:  float = 2.0,
        lambda_mel: float = 45.0,
        lambda_vq:  float = 1.0,
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm  = lambda_fm
        self.lambda_mel = lambda_mel
        self.lambda_vq  = lambda_vq

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TotalGeneratorLoss":
        return cls(
            lambda_adv = cfg.loss.lambda_adv,
            lambda_fm  = cfg.loss.lambda_fm,
            lambda_mel = cfg.loss.lambda_mel,
            lambda_vq  = cfg.loss.lambda_vq,
        )

    def forward(
        self,
        adv_loss: torch.Tensor,
        fm_loss:  torch.Tensor,
        mel_loss: torch.Tensor,
        vq_loss:  torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        total = (
            self.lambda_adv * adv_loss
            + self.lambda_fm  * fm_loss
            + self.lambda_mel * mel_loss
            + self.lambda_vq  * vq_loss
        )
        breakdown = {
            "adv": adv_loss.item(),
            "fm":  fm_loss.item(),
            "mel": mel_loss.item(),
            "vq":  vq_loss.item(),
            "total": total.item(),
        }
        return total, breakdown
