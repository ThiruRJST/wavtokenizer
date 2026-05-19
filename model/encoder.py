"""
WavTokenizer Encoder
─────────────────────
Raw waveform  →  latent frames

Architecture:
  Input Conv → [ResidualUnit × 3  +  Strided Conv] × n_stages → Projection
  Each residual block uses dilations [1, 3, 9] for a large effective receptive field.
  Strided convolutions provide temporal downsampling; the product of all strides
  must equal  sample_rate // token_rate  (e.g., 2×4×5×8 = 320 for 75 tok/s).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ResidualUnit(nn.Module):
    """Dilated causal-style residual block."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels,
                kernel_size=3, dilation=dilation,
                padding=dilation,
            ),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EncoderBlock(nn.Module):
    """
    One encoder stage:
      N residual units (with increasing dilation) → strided conv downsampler
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: List[int] = (1, 3, 9),
    ):
        super().__init__()
        self.res = nn.Sequential(
            *[ResidualUnit(in_channels, d) for d in dilations]
        )
        # Strided conv: kernel = 2×stride keeps the output length exact
        self.down = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
            ),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.res(x))


class Encoder(nn.Module):
    """
    Full encoder stack.

    Args (from config):
        in_channels   : 1 for mono audio
        base_channels : first-stage channel count; doubles each stage
        latent_dim    : output channel dimension (= VQ embedding_dim)
        strides       : downsampling factor per stage; product = hop_length
        dilations     : per-residual-block dilation list
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 512,
        strides: Tuple[int, ...] = (2, 4, 5, 8),
        dilations: Tuple[int, ...] = (1, 3, 9),
    ):
        super().__init__()
        ch = base_channels
        self.input_conv = nn.Conv1d(in_channels, ch, kernel_size=7, padding=3)

        blocks: List[EncoderBlock] = []
        for stride in strides:
            blocks.append(EncoderBlock(ch, ch * 2, stride, list(dilations)))
            ch *= 2
        self.blocks = nn.Sequential(*blocks)

        # Final projection to latent space
        self.proj = nn.Conv1d(ch, latent_dim, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Encoder":
        return cls(
            in_channels   = cfg.encoder.in_channels,
            base_channels = cfg.encoder.base_channels,
            latent_dim    = cfg.encoder.latent_dim,
            strides       = tuple(cfg.encoder.strides),
            dilations     = tuple(cfg.encoder.residual_dilations),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T_audio)
        Returns:
            z: (B, latent_dim, T_frames)  where T_frames = T_audio / prod(strides)
        """
        x = F.elu(self.input_conv(x))
        x = self.blocks(x)
        return self.proj(x)
