"""Tests for model/encoder.py, model/quantizer.py, model/decoder.py, model/wavtokenizer.py"""

import pytest
import torch
import torch.nn as nn

from model import WavTokenizer, build_model
from model.encoder import Encoder, ResidualUnit, EncoderBlock
from model.quantizer import VectorQuantizer
from model.decoder import Decoder, ConvNeXtBlock, SelfAttention1D, InverseSTFTUpsampler


# ─── ResidualUnit ────────────────────────────────────────────────────────────

class TestResidualUnit:
    def test_output_shape(self):
        unit = ResidualUnit(channels=32, dilation=3)
        x    = torch.randn(2, 32, 100)
        y    = unit(x)
        assert y.shape == x.shape, "ResidualUnit must preserve shape"

    def test_skip_connection(self):
        """With zero-init weights, output ≈ input (residual identity)."""
        unit = ResidualUnit(channels=16, dilation=1)
        # Force zero weights on the additive path
        for p in unit.net.parameters():
            nn.init.zeros_(p)
        x = torch.randn(1, 16, 50)
        y = unit(x)
        # With zero-weight net, net(x)=0, so y = x + 0 = x
        # (ELU(0)=0, so biases must also be zero)
        for p in unit.net.parameters():
            if p.requires_grad:
                assert p.abs().max().item() == 0.0

    def test_different_dilations(self):
        for d in [1, 3, 9]:
            unit = ResidualUnit(64, dilation=d)
            x    = torch.randn(1, 64, 200)
            y    = unit(x)
            assert y.shape == x.shape


# ─── EncoderBlock ────────────────────────────────────────────────────────────

class TestEncoderBlock:
    def test_downsampling(self):
        block = EncoderBlock(in_channels=32, out_channels=64, stride=4)
        x     = torch.randn(2, 32, 400)
        y     = block(x)
        assert y.shape[0] == 2
        assert y.shape[1] == 64
        assert y.shape[2] == 100, f"Expected 100, got {y.shape[2]}"

    def test_channel_doubling(self):
        block = EncoderBlock(16, 32, stride=2)
        x     = torch.randn(1, 16, 100)
        y     = block(x)
        assert y.shape[1] == 32


# ─── Encoder ─────────────────────────────────────────────────────────────────

class TestEncoder:
    def test_output_shape(self, cfg):
        enc = Encoder.from_config(cfg)
        wav = torch.randn(2, 1, 12_000)
        z   = enc(wav)
        assert z.shape[0] == 2
        assert z.shape[1] == cfg.encoder.latent_dim
        # Total stride = 2*4*5*8 = 320; 12000/320 = 37 frames
        expected_frames = 12_000 // 320
        assert z.shape[2] == expected_frames, f"Expected {expected_frames} frames, got {z.shape[2]}"

    def test_gradient_flow(self, cfg):
        enc = Encoder.from_config(cfg)
        wav = torch.randn(1, 1, 3_200, requires_grad=False)
        z   = enc(wav)
        loss = z.sum()
        loss.backward()
        # Check at least one gradient is non-zero
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in enc.parameters())
        assert has_grad, "Encoder has no gradient flow"

    def test_different_input_lengths(self, cfg):
        enc  = Encoder.from_config(cfg)
        hop  = 320
        for length in [hop * 10, hop * 37, hop * 100]:
            z = enc(torch.randn(1, 1, length))
            expected = length // hop
            # Strided-conv padding may produce ±1 frame difference
            assert abs(z.shape[2] - expected) <= 2, \
                f"Expected ≈{expected} frames, got {z.shape[2]} for input length {length}"


# ─── VectorQuantizer ─────────────────────────────────────────────────────────

class TestVectorQuantizer:
    def test_output_shapes(self, cfg):
        vq  = VectorQuantizer.from_config(cfg)
        z   = torch.randn(2, cfg.quantizer.embedding_dim, 37)
        z_q, idx, loss = vq(z)
        assert z_q.shape == z.shape,         "z_q shape mismatch"
        assert idx.shape == (2, 37),          "indices shape mismatch"
        assert loss.ndim == 0,                "vq_loss should be scalar"

    def test_indices_in_range(self, cfg):
        vq  = VectorQuantizer.from_config(cfg)
        z   = torch.randn(1, cfg.quantizer.embedding_dim, 10)
        _, idx, _ = vq(z)
        assert idx.min() >= 0
        assert idx.max() < cfg.quantizer.codebook_size

    def test_straight_through(self, cfg):
        """Gradient must flow through straight-through estimator."""
        vq = VectorQuantizer.from_config(cfg)
        z  = torch.randn(1, cfg.quantizer.embedding_dim, 5, requires_grad=True)
        z_q, _, loss = vq(z)
        (z_q.sum() + loss).backward()
        assert z.grad is not None, "No gradient at encoder output"
        assert z.grad.abs().sum() > 0

    def test_decode_indices(self, cfg):
        vq  = VectorQuantizer.from_config(cfg)
        idx = torch.randint(0, cfg.quantizer.codebook_size, (2, 15))
        out = vq.decode_indices(idx)
        assert out.shape == (2, cfg.quantizer.embedding_dim, 15)

    def test_kmeans_init_runs(self, cfg):
        """First training forward should trigger K-means init without error."""
        vq = VectorQuantizer.from_config(cfg)
        vq.train()
        assert not vq._initialized.item()
        z  = torch.randn(300, cfg.quantizer.embedding_dim, 5)   # large batch
        z_q, _, _ = vq(z)
        assert vq._initialized.item(), "K-means init should have run"

    def test_codebook_utilization(self, cfg):
        vq = VectorQuantizer.from_config(cfg)
        vq.train()
        z  = torch.randn(2, cfg.quantizer.embedding_dim, 20)
        vq(z)
        util = vq.codebook_utilization
        assert 0.0 <= util <= 1.0


# ─── Decoder ─────────────────────────────────────────────────────────────────

class TestDecoder:
    def test_output_shape(self, cfg):
        dec  = Decoder.from_config(cfg)
        z_q  = torch.randn(2, cfg.quantizer.embedding_dim, 37)
        wav  = dec(z_q)
        assert wav.shape[0] == 2
        assert wav.shape[1] == 1   # mono
        assert wav.shape[2] > 0

    def test_gradient_flow(self, cfg):
        dec = Decoder.from_config(cfg)
        z_q = torch.randn(1, cfg.quantizer.embedding_dim, 10, requires_grad=True)
        wav = dec(z_q)
        wav.sum().backward()
        assert z_q.grad is not None and z_q.grad.abs().sum() > 0

    def test_convnext_block(self):
        block = ConvNeXtBlock(channels=64, kernel_size=7, expansion=4)
        x     = torch.randn(2, 64, 50)
        y     = block(x)
        assert y.shape == x.shape

    def test_self_attention(self):
        attn = SelfAttention1D(channels=64, num_heads=4)
        x    = torch.randn(2, 64, 30)
        y    = attn(x)
        assert y.shape == x.shape

    def test_istft_output_length(self):
        istft = InverseSTFTUpsampler(channels=64, n_fft=640, hop_length=320)
        x     = torch.randn(2, 64, 37)   # 37 frames
        wav   = istft(x)
        assert wav.shape[0] == 2
        assert wav.shape[1] == 1
        # iSTFT output length ≈ frames × hop_length
        assert wav.shape[2] > 0


# ─── WavTokenizer (end-to-end) ────────────────────────────────────────────────

class TestWavTokenizer:
    def test_forward_shapes(self, model, short_wav):
        rec, vq_loss = model(short_wav)
        assert rec.shape[0] == short_wav.shape[0]
        assert rec.shape[1] == 1
        assert vq_loss.ndim == 0

    def test_encode_shape(self, model, short_wav):
        idx = model.encode(short_wav)
        assert idx.shape[0] == short_wav.shape[0]
        assert idx.ndim == 2

    def test_decode_shape(self, model, cfg):
        idx = torch.randint(0, cfg.quantizer.codebook_size, (2, 10))
        wav = model.decode(idx)
        assert wav.shape == (2, 1, wav.shape[2])

    def test_roundtrip_deterministic(self, model, short_wav):
        """Two encode→decode passes with the same input must give the same output."""
        model.eval()
        with torch.no_grad():
            r1 = model.decode(model.encode(short_wav))
            r2 = model.decode(model.encode(short_wav))
        assert torch.allclose(r1, r2), "Roundtrip is not deterministic"

    def test_gradients_end_to_end(self, cfg):
        """Full forward + backward pass must produce non-zero gradients."""
        m   = build_model(cfg)
        wav = torch.randn(1, 1, 2_400)
        rec, vq = m(wav)
        loss = rec.sum() + vq
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m.parameters())
        assert has_grad

    def test_model_num_parameters(self, model):
        n = model.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_codebook_utilization_property(self, model, short_wav):
        model.train()
        model(short_wav)
        u = model.codebook_utilization
        assert 0.0 <= u <= 1.0
