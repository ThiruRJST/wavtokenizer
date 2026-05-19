"""
WavTokenizer  —  Top-level codec model
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .encoder     import Encoder
from .quantizer   import VectorQuantizer
from .decoder     import Decoder


class WavTokenizer(nn.Module):
    """
    Full encode→quantize→decode pipeline.

    Public API
    ──────────
    encode(wav)         : (B,1,T) → token indices (B, T_tokens)
    decode(indices)     : (B, T_tokens) → waveform (B, 1, T_audio)
    forward(wav)        : (B,1,T) → reconstructed wav + vq_loss scalar
    """

    def __init__(
        self,
        encoder:   Encoder,
        quantizer: VectorQuantizer,
        decoder:   Decoder,
        sample_rate: int = 24_000,
        token_rate:  int = 75,
    ):
        super().__init__()
        self.encoder    = encoder
        self.quantizer  = quantizer
        self.decoder    = decoder
        self.sample_rate = sample_rate
        self.token_rate  = token_rate

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "WavTokenizer":
        enc = Encoder.from_config(cfg)
        vq  = VectorQuantizer.from_config(cfg)
        dec = Decoder.from_config(cfg)
        return cls(enc, vq, dec,
                   sample_rate=cfg.audio.sample_rate,
                   token_rate=cfg.audio.token_rate)

    # ── Core methods ─────────────────────────────────────────────────────────

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Tokenize a waveform.  Returns integer token indices."""
        z = self.encoder(wav)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform from token indices."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q)

    def forward(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode→quantize→decode forward pass used during training.

        Returns:
            wav_rec  : reconstructed waveform  (B, 1, T)  — may differ from
                       input length by a few samples due to iSTFT windowing
            vq_loss  : scalar VQ commitment + codebook loss
        """
        z          = self.encoder(wav)
        z_q, _, vq_loss = self.quantizer(z)
        wav_rec    = self.decoder(z_q)
        return wav_rec, vq_loss

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def codebook_utilization(self) -> float:
        return self.quantizer.codebook_utilization

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: DictConfig) -> WavTokenizer:
    """Factory: build WavTokenizer from OmegaConf config."""
    model = WavTokenizer.from_config(cfg)
    return model
