"""
WavTokenizer Discriminators
────────────────────────────
Three complementary discriminators as described in the paper:

  1. MultiPeriodDiscriminator  (MPD)    — from HiFi-GAN; evaluates periodic structure
  2. MultiScaleDiscriminator   (MSD)    — judges audio at multiple resolutions
  3. MultiResolutionSTFTDisc   (MRSTFTD)— spectral sub-band discrimination

Together they provide gradients across time, scale, and frequency to guide
the generator (encoder + VQ + decoder) toward high-fidelity reconstruction.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _wn(layer):
    return nn.utils.parametrizations.weight_norm(layer)


def _sn(layer):
    return nn.utils.parametrize.register_parametrization(
        layer, "weight", nn.utils.parametrizations.spectral_norm()
    ) if False else nn.utils.spectral_norm(layer)   # use nn.utils for compat


# ─── 1. Multi-Period Discriminator ───────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """
    HiFi-GAN period discriminator.
    Reshapes 1-D audio into a 2-D (T//p, p) view and applies 2-D convolutions,
    letting the discriminator evaluate periodic structure at period p.
    """

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        chs = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv2d(
                    chs[i], chs[i + 1],
                    kernel_size = (kernel_size, 1),
                    stride      = (stride, 1),
                    padding     = (2, 0),
                )
            ) for i in range(len(chs) - 1)
        ])
        self.post = nn.utils.weight_norm(
            nn.Conv2d(1024, 1, (3, 1), padding=(1, 0))
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (B, 1, T)
        B, C, T = x.shape
        pad = (self.period - T % self.period) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
        x = x.view(B, 1, -1, self.period)                      # (B, 1, T//p, p)

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MultiPeriodDiscriminator":
        return cls(periods=tuple(cfg.discriminator.mpd_periods))

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        r_outs, f_outs, r_fmaps, f_fmaps = [], [], [], []
        for d in self.discs:
            ro, rf = d(real)
            fo, ff = d(fake)
            r_outs.append(ro);  f_outs.append(fo)
            r_fmaps.append(rf); f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps


# ─── 2. Multi-Scale Discriminator ────────────────────────────────────────────

class ScaleDiscriminator(nn.Module):
    """HiFi-GAN scale discriminator with optional spectral norm."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,     128,  15, 1,  7)),
            norm(nn.Conv1d(128,   128,  41, 2, 20, groups=4)),
            norm(nn.Conv1d(128,   256,  41, 2, 20, groups=16)),
            norm(nn.Conv1d(256,   512,  41, 4, 20, groups=16)),
            norm(nn.Conv1d(512,  1024,  41, 4, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024,  41, 1, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024,   5, 1,  2)),
        ])
        self.post = norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3):
        super().__init__()
        # First sub-disc uses spectral norm; rest use weight norm
        self.discs = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=(i == 0))
            for i in range(n_scales)
        ])
        self.pools = nn.ModuleList(
            [nn.Identity()]
            + [nn.AvgPool1d(4, 2, padding=2) for _ in range(n_scales - 1)]
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MultiScaleDiscriminator":
        return cls(n_scales=cfg.discriminator.msd_scales)

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        r_outs, f_outs, r_fmaps, f_fmaps = [], [], [], []
        for d, pool in zip(self.discs, self.pools):
            ro, rf = d(pool(real))
            fo, ff = d(pool(fake))
            r_outs.append(ro);  f_outs.append(fo)
            r_fmaps.append(rf); f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps


# ─── 3. Multi-Resolution STFT Discriminator ──────────────────────────────────

class STFTDiscriminator(nn.Module):
    """
    Discriminates in the magnitude-spectrogram domain at one STFT resolution.
    Sub-band splitting (via strided 2-D convolutions) provides targeted
    frequency-domain gradient signals, improving high-frequency prediction.
    """

    def __init__(self, n_fft: int, hop: int, win: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop
        self.win   = win
        # 2-D conv stack over (freq_bins, time_frames)
        self.convs = nn.Sequential(
            nn.Conv2d(1,  32,  (3, 9), padding=(1, 4)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32,  (3, 8), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32,  (3, 8), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32,  (3, 6), stride=(1, 2), padding=(1, 3)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32,  1,  (3, 3), padding=(1, 1)),
        )

    def _get_fmaps(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmaps = []
        for layer in self.convs:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                fmaps.append(x)
        return x, fmaps

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (B, 1, T)
        B = x.shape[0]
        window = torch.hann_window(self.win, device=x.device, dtype=x.dtype)
        specs  = []
        for b in range(B):
            s = torch.stft(
                x[b, 0],
                n_fft      = self.n_fft,
                hop_length = self.hop,
                win_length = self.win,
                window     = window,
                return_complex = True,
            ).abs()
            specs.append(s)
        mag = torch.stack(specs).unsqueeze(1)                   # (B, 1, freq, time)
        return self._get_fmaps(mag)


class MultiResolutionSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int], ...] = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512,   50,  240),
        ),
    ):
        super().__init__()
        self.discs = nn.ModuleList([
            STFTDiscriminator(n, h, w) for n, h, w in resolutions
        ])

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MultiResolutionSTFTDiscriminator":
        return cls(resolutions=[tuple(r) for r in cfg.discriminator.mrstftd_resolutions])

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        r_outs, f_outs, r_fmaps, f_fmaps = [], [], [], []
        for d in self.discs:
            ro, rf = d(real)
            fo, ff = d(fake)
            r_outs.append(ro);  f_outs.append(fo)
            r_fmaps.append(rf); f_fmaps.append(ff)
        return r_outs, f_outs, r_fmaps, f_fmaps
