"""
WavTokenizer Vector Quantizer
──────────────────────────────
Single-layer VQ with:
  • Expanded codebook (up to 16 384 entries aligned with text vocab size)
  • K-means clustering initialization on first batch
  • Random awakening to recover dead (unused) codes
  • Straight-through estimator for end-to-end gradient flow

Key insight from the paper: compressing to a single quantizer (vs. the typical
4–8 used in EnCodec / SoundStream) is made viable by expanding the VQ space
rather than stacking multiple smaller codebooks.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        codebook_size: int = 4096,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
        dead_threshold: int = 100,
    ):
        """
        Args:
            codebook_size   : number of discrete codes  K
            embedding_dim   : code vector dimensionality D  (= latent_dim)
            commitment_cost : β in  L_commit = β · ‖z_e − sg[e]‖²
            dead_threshold  : codes used < this many times → re-init (random awakening)
        """
        super().__init__()
        self.K = codebook_size
        self.D = embedding_dim
        self.commitment_cost = commitment_cost
        self.dead_threshold  = dead_threshold

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA usage counters — NOT a learnable parameter
        self.register_buffer("code_usage",   torch.zeros(codebook_size))
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))

    # ── K-means initialization ────────────────────────────────────────────────

    @torch.no_grad()
    def _kmeans_init(self, flat: torch.Tensor):
        """
        Seed the codebook with K random vectors from the first batch.
        This drastically improves codebook utilization vs. random init.
        """
        n = flat.size(0)
        if n >= self.K:
            idx = torch.randperm(n, device=flat.device)[: self.K]
        else:
            # Tile and trim if batch is smaller than codebook
            repeat = math.ceil(self.K / n)
            idx = torch.randperm(n, device=flat.device).repeat(repeat)[: self.K]
        self.embedding.weight.data.copy_(flat[idx].detach())
        self._initialized.fill_(True)

    # ── Random awakening ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _random_awakening(self, flat: torch.Tensor):
        """
        Re-initialize codes that have not been selected in the current batch.
        Prevents codebook collapse where a few codes dominate and many go unused.
        """
        dead_mask = self.code_usage < 1.0
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return
        src_idx = torch.randint(0, flat.size(0), (int(n_dead),), device=flat.device)
        self.embedding.weight.data[dead_mask] = flat[src_idx].detach()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z : encoder output  (B, D, T)

        Returns:
            z_q     : quantized tensor  (B, D, T)  — straight-through gradient
            indices : token ids         (B, T)     — detached
            vq_loss : scalar            commitment + codebook loss
        """
        B, D, T = z.shape
        flat = z.permute(0, 2, 1).reshape(-1, D)               # (B·T, D)

        # ── Init ──────────────────────────────────────────────────────────────
        if self.training and not self._initialized.item():
            self._kmeans_init(flat)

        # ── Nearest-neighbour lookup ──────────────────────────────────────────
        # ||z - e||² = ||z||² - 2⟨z,e⟩ + ||e||²
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * (flat @ self.embedding.weight.T)
            + self.embedding.weight.pow(2).sum(1)
        )                                                        # (B·T, K)
        indices_flat = dist.argmin(dim=1)                       # (B·T,)

        # ── EMA usage + random awakening ─────────────────────────────────────
        if self.training:
            with torch.no_grad():
                self.code_usage.zero_()
                self.code_usage.scatter_add_(
                    0, indices_flat,
                    torch.ones_like(indices_flat, dtype=torch.float),
                )
                self._random_awakening(flat)

        # ── Quantized embedding ───────────────────────────────────────────────
        z_q_flat = self.embedding(indices_flat)                  # (B·T, D)

        # ── VQ losses ─────────────────────────────────────────────────────────
        codebook_loss   = F.mse_loss(z_q_flat,        flat.detach())
        commitment_loss = F.mse_loss(flat,             z_q_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # ── Straight-through estimator ────────────────────────────────────────
        # Pass gradients through as if quantization were identity
        z_q_st = flat + (z_q_flat - flat).detach()
        z_q    = z_q_st.reshape(B, T, D).permute(0, 2, 1)      # (B, D, T)
        indices = indices_flat.reshape(B, T)                    # (B, T)

        return z_q, indices, vq_loss

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids → (B, D, T) quantized latents."""
        return self.embedding(indices).permute(0, 2, 1)

    @property
    def codebook_utilization(self) -> float:
        """Fraction of codes that were used in the last training batch."""
        return (self.code_usage > 0).float().mean().item()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "VectorQuantizer":
        return cls(
            codebook_size   = cfg.quantizer.codebook_size,
            embedding_dim   = cfg.quantizer.embedding_dim,
            commitment_cost = cfg.quantizer.commitment_cost,
            dead_threshold  = cfg.quantizer.dead_threshold,
        )
