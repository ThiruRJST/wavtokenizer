"""Tests for data/augmentation/transforms.py"""

import pytest
import torch

from data.augmentation import (
    GaussianNoise,
    RandomGain,
    RandomPolarityInvert,
    LowPassFilter,
    HighPassFilter,
    BandDropout,
    RoomImpulseResponse,
    PitchShift,
    TimeStretch,
    AudioAugmentationPipeline,
    build_augmentation_pipeline,
)


@pytest.fixture
def mono_wav():
    """Short mono batch for augmentation tests."""
    torch.manual_seed(42)
    return torch.randn(1, 1, 1_200)   # 0.2 s at 24 kHz


def _shape_preserved(transform, wav):
    """Helper: augmented output shape matches input."""
    out = transform(wav)
    return out.shape == wav.shape


# ─── Shape preservation ───────────────────────────────────────────────────────

class TestShapePreservation:
    def test_gaussian_noise(self, mono_wav):
        t = GaussianNoise(p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_random_gain(self, mono_wav):
        t = RandomGain(p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_polarity(self, mono_wav):
        t = RandomPolarityInvert(p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_lowpass(self, mono_wav):
        t = LowPassFilter(sample_rate=24_000, p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_highpass(self, mono_wav):
        t = HighPassFilter(sample_rate=24_000, p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_band_dropout(self, mono_wav):
        t = BandDropout(p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_rir(self, mono_wav):
        t = RoomImpulseResponse(sample_rate=24_000, p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_pitch_shift(self, mono_wav):
        t = PitchShift(sample_rate=24_000, p=1.0)
        assert _shape_preserved(t, mono_wav)

    def test_time_stretch(self, mono_wav):
        t = TimeStretch(sample_rate=24_000, p=1.0)
        assert _shape_preserved(t, mono_wav)


# ─── Semantic correctness ────────────────────────────────────────────────────

class TestSemantics:
    def test_noise_changes_waveform(self, mono_wav):
        t   = GaussianNoise(min_snr_db=5, max_snr_db=10, p=1.0)
        out = t(mono_wav)
        assert not torch.allclose(out, mono_wav)

    def test_polarity_inverts(self, mono_wav):
        t   = RandomPolarityInvert(p=1.0)
        out = t(mono_wav)
        assert torch.allclose(out, -mono_wav)

    def test_gain_changes_amplitude(self, mono_wav):
        # +12 dB should increase RMS energy, -12 dB should decrease it
        wav_low = torch.full_like(mono_wav, 0.05)   # small amplitude avoids clipping
        t_up  = RandomGain(min_db=12.0, max_db=12.0, p=1.0)
        t_dn  = RandomGain(min_db=-12.0, max_db=-12.0, p=1.0)
        assert t_up(wav_low).abs().mean() > wav_low.abs().mean(), "Positive gain should increase energy"
        assert t_dn(wav_low).abs().mean() < wav_low.abs().mean(), "Negative gain should decrease energy"

    def test_probability_zero_is_identity(self, mono_wav):
        for t in [GaussianNoise(p=0), RandomGain(p=0), RandomPolarityInvert(p=0)]:
            out = t(mono_wav)
            assert torch.allclose(out, mono_wav), f"{t} with p=0 should be identity"

    def test_lowpass_attenuates_high_freq(self, mono_wav):
        """After low-pass filtering, high-frequency energy should be reduced."""
        import torch.fft
        t   = LowPassFilter(min_cutoff_hz=500, max_cutoff_hz=500,
                            sample_rate=24_000, p=1.0)
        out = t(mono_wav)
        spec_in  = torch.fft.rfft(mono_wav.squeeze(), dim=-1).abs()
        spec_out = torch.fft.rfft(out.squeeze(),      dim=-1).abs()
        # High-frequency bins (top 10%) should be reduced
        n = spec_in.shape[-1]
        hi_slice = slice(int(n * 0.9), n)
        hi_in  = spec_in[..., hi_slice].mean().item()
        hi_out = spec_out[..., hi_slice].mean().item()
        assert hi_out < hi_in, "LowPassFilter did not attenuate high frequencies"


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class TestAugmentationPipeline:
    def test_empty_pipeline(self, mono_wav):
        pipeline = AudioAugmentationPipeline([])
        out = pipeline(mono_wav)
        assert torch.allclose(out, mono_wav)

    def test_full_pipeline_shape(self, mono_wav):
        pipeline = AudioAugmentationPipeline([
            GaussianNoise(p=1.0),
            RandomGain(p=1.0),
            RandomPolarityInvert(p=1.0),
        ])
        out = pipeline(mono_wav)
        assert out.shape == mono_wav.shape

    def test_repr(self):
        pipeline = AudioAugmentationPipeline([GaussianNoise(p=0.5)])
        r = repr(pipeline)
        assert "AudioAugmentationPipeline" in r
        assert "GaussianNoise" in r

    def test_build_from_config_disabled(self, cfg):
        result = build_augmentation_pipeline(cfg)
        assert result is None   # small config has augmentation disabled

    def test_build_from_config_enabled(self, cfg):
        from omegaconf import OmegaConf
        cfg2 = OmegaConf.merge(cfg, OmegaConf.create({
            "data": {
                "augmentation": {
                    "enabled": True,
                    "noise": {"p": 0.5, "min_snr_db": 10, "max_snr_db": 40},
                    "gain":  {"p": 0.5, "min_db": -6, "max_db": 6},
                }
            }
        }))
        pipeline = build_augmentation_pipeline(cfg2)
        assert pipeline is not None
        assert len(pipeline.transforms) == 2
        out = pipeline(torch.randn(1, 1, 2_400))
        assert out.shape == (1, 1, 2_400)
