"""
WavTokenizer Decoder
─────────────────────
Quantized latent frames  →  waveform

Architecture (paper §3.3):
    Conv1D  →  SelfAttention  →  N × ConvNeXtBlock  →  iSTFT upsampling

Key design choices vs. standard mirror decoders:
  • ConvNeXt blocks instead of dilated-conv stacks
  • Multi-head self-attention to expand contextual window
  • Inverse STFT (iSTFT) for upsampling — avoids aliasing from transposed convs
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ─── ConvNeXt Block (1D) ─────────────────────────────────────────────────────

class ConvNeXtBlock(nn.Module):
    """
    1-D ConvNeXt block:
        depth-wise conv  →  LayerNorm  →  point-wise FFN  (GELU, expansion×)

    Using ConvNeXt instead of plain residual dilated convs gives the decoder
    a larger receptive field per parameter.
    """

    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        inner = channels * expansion
        self.dw   = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm  = nn.LayerNorm(channels)
        self.pw1   = nn.Linear(channels, inner)
        self.pw2   = nn.Linear(inner, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)                                          # (B, C, T)
        x = x.permute(0, 2, 1)                                  # (B, T, C)
        x = self.norm(x)
        x = F.gelu(self.pw1(x))
        x = self.pw2(x)
        return residual + x.permute(0, 2, 1)


# ─── Self-Attention (1D time axis) ───────────────────────────────────────────

class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.
    The paper found that adding an attention module in the decoder significantly
    improves semantic richness of the reconstructed audio.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        xt = x.permute(0, 2, 1)                                 # (B, T, C)
        out, _ = self.attn(xt, xt, xt)
        return x + self.norm(out).permute(0, 2, 1)


# ─── Inverse STFT Upsampler ──────────────────────────────────────────────────

class InverseSTFTUpsampler(nn.Module):
    """
    Replace transposed-convolution upsampling with iSTFT.

    The decoder head predicts a complex spectrogram (real + imaginary components
    via a 1×1 Conv1D) and reconstructs the waveform via iSTFT (overlap-add).

    This eliminates the periodic aliasing artifacts that dilated-transposed-conv
    decoders are prone to, as noted in the WavTokenizer paper (§3.3) and FACodec.
    """

    def __init__(self, channels: int, n_fft: int = 640, hop_length: int = 320):
        super().__init__()
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.win_length  = n_fft
        n_bins = n_fft // 2 + 1
        # Project feature map to 2 × n_bins  (real + imaginary)
        self.proj = nn.Conv1d(channels, n_bins * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_frames)
        B = x.shape[0]
        spec = self.proj(x)                                      # (B, 2*n_bins, T)
        n_bins = self.n_fft // 2 + 1
        real = spec[:, :n_bins, :]                               # (B, n_bins, T)
        imag = spec[:, n_bins:, :]

        stft_c = torch.complex(real, imag).permute(0, 2, 1)     # (B, T, n_bins)

        window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype)
        waveforms = []
        for b in range(B):
            wav = torch.istft(
                stft_c[b].T,                                     # (n_bins, T)
                n_fft      = self.n_fft,
                hop_length = self.hop_length,
                win_length = self.win_length,
                window     = window,
                return_complex = False,
            )
            waveforms.append(wav)

        return torch.stack(waveforms).unsqueeze(1)               # (B, 1, T_audio)


# ─── Full Decoder ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Args (from config):
        latent_dim    : must match encoder.latent_dim and quantizer.embedding_dim
        hidden        : internal channel width
        n_convnext    : number of ConvNeXt blocks
        n_heads       : attention heads
        istft_n_fft   : FFT size for iSTFT (≈ 2 × hop_length recommended)
        istft_hop_length : must match encoder's hop / audio.hop_length
    """

    def __init__(
        self,
        latent_dim:       int = 512,
        hidden:           int = 512,
        n_convnext:       int = 8,
        n_heads:          int = 8,
        convnext_kernel:  int = 7,
        convnext_expand:  int = 4,
        istft_n_fft:      int = 640,
        istft_hop_length: int = 320,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(latent_dim, hidden, kernel_size=7, padding=3)
        self.attn       = SelfAttention1D(hidden, num_heads=n_heads)
        self.convnext   = nn.Sequential(*[
            ConvNeXtBlock(hidden, convnext_kernel, convnext_expand)
            for _ in range(n_convnext)
        ])
        self.istft = InverseSTFTUpsampler(
            hidden, n_fft=istft_n_fft, hop_length=istft_hop_length
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Decoder":
        return cls(
            latent_dim       = cfg.quantizer.embedding_dim,
            hidden           = cfg.decoder.hidden,
            n_convnext       = cfg.decoder.n_convnext,
            n_heads          = cfg.decoder.n_heads,
            convnext_kernel  = cfg.decoder.convnext_kernel,
            convnext_expand  = cfg.decoder.convnext_expansion,
            istft_n_fft      = cfg.decoder.istft_n_fft,
            istft_hop_length = cfg.decoder.istft_hop_length,
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q : (B, D, T_frames)
        Returns:
            wav : (B, 1, T_audio)
        """
        x = F.elu(self.input_proj(z_q))
        x = self.attn(x)
        x = self.convnext(x)
        return self.istft(x)
