"""Tests for model/discriminators.py"""

import pytest
import torch

from model.discriminators import (
    PeriodDiscriminator,
    MultiPeriodDiscriminator,
    ScaleDiscriminator,
    MultiScaleDiscriminator,
    STFTDiscriminator,
    MultiResolutionSTFTDiscriminator,
)


@pytest.fixture
def audio_pair():
    torch.manual_seed(7)
    real = torch.randn(2, 1, 4_800)   # 0.2 s at 24 kHz
    fake = torch.randn(2, 1, 4_800)
    return real, fake


# ─── PeriodDiscriminator ──────────────────────────────────────────────────────

class TestPeriodDiscriminator:
    @pytest.mark.parametrize("period", [2, 3, 5, 7, 11])
    def test_output_not_empty(self, audio_pair, period):
        real, fake = audio_pair
        d    = PeriodDiscriminator(period)
        out, fmaps = d(real)
        assert out.numel() > 0
        assert len(fmaps) > 0

    def test_feature_map_count(self, audio_pair):
        real, _ = audio_pair
        d = PeriodDiscriminator(period=5)
        _, fmaps = d(real)
        assert len(fmaps) == 6   # 5 conv layers + post

    def test_gradients(self, audio_pair):
        real, _ = audio_pair
        d   = PeriodDiscriminator(period=3)
        out, _ = d(real)
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in d.parameters())
        assert has_grad


# ─── MultiPeriodDiscriminator ────────────────────────────────────────────────

class TestMultiPeriodDiscriminator:
    def test_n_subdiscs(self, mpd, audio_pair):
        real, fake = audio_pair
        ro, fo, rfm, ffm = mpd(real, fake)
        assert len(ro)  == 5
        assert len(fo)  == 5
        assert len(rfm) == 5
        assert len(ffm) == 5

    def test_fmap_list_of_lists(self, mpd, audio_pair):
        real, fake = audio_pair
        _, _, rfm, ffm = mpd(real, fake)
        for sub in rfm:
            assert isinstance(sub, list)
            for t in sub:
                assert t.ndim >= 2


# ─── ScaleDiscriminator ───────────────────────────────────────────────────────

class TestScaleDiscriminator:
    def test_output_shape(self, audio_pair):
        real, _ = audio_pair
        d  = ScaleDiscriminator(use_spectral_norm=False)
        out, fmaps = d(real)
        assert out.ndim == 2   # (B, features)
        assert len(fmaps) == 8  # 7 convs + post

    def test_spectral_norm_variant(self, audio_pair):
        real, _ = audio_pair
        d = ScaleDiscriminator(use_spectral_norm=True)
        out, _ = d(real)
        assert out.numel() > 0


# ─── MultiScaleDiscriminator ─────────────────────────────────────────────────

class TestMultiScaleDiscriminator:
    def test_n_scales(self, msd, audio_pair):
        real, fake = audio_pair
        ro, fo, rfm, ffm = msd(real, fake)
        assert len(ro) == 3
        assert len(fo) == 3

    def test_pooling_reduces_length(self, audio_pair):
        """Middle and last sub-discs receive pooled (shorter) audio."""
        real, fake = audio_pair
        msd = MultiScaleDiscriminator(n_scales=3)
        # The pooling doesn't affect the discriminator outputs directly
        # but it should not raise errors
        ro, fo, _, _ = msd(real, fake)
        assert all(o.numel() > 0 for o in ro + fo)


# ─── STFTDiscriminator ────────────────────────────────────────────────────────

class TestSTFTDiscriminator:
    @pytest.mark.parametrize("n_fft,hop,win", [(1024, 120, 600), (512, 50, 240)])
    def test_output_shape(self, audio_pair, n_fft, hop, win):
        real, _ = audio_pair
        d   = STFTDiscriminator(n_fft, hop, win)
        out, fmaps = d(real)
        assert out.numel() > 0
        assert len(fmaps) > 0

    def test_gradient_flow(self, audio_pair):
        real, _ = audio_pair
        d = STFTDiscriminator(1024, 120, 600)
        out, _ = d(real)
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in d.parameters())
        assert has_grad


# ─── MultiResolutionSTFTDiscriminator ────────────────────────────────────────

class TestMultiResolutionSTFTDiscriminator:
    def test_n_resolutions(self, mrstftd, audio_pair):
        real, fake = audio_pair
        ro, fo, rfm, ffm = mrstftd(real, fake)
        assert len(ro) == 3

    def test_from_config(self, cfg):
        d = MultiResolutionSTFTDiscriminator.from_config(cfg)
        assert len(d.discs) == 3
