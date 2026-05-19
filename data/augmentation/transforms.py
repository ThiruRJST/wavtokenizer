"""
Audio Augmentation Pipeline for WavTokenizer Training
───────────────────────────────────────────────────────
All transforms operate on float32 waveform tensors (B, 1, T) or (1, T).
Every transform is individually togglable and probability-gated.

Transforms:
  GaussianNoise         — additive white noise (SNR-controlled)
  RandomGain            — volume scaling ∈ [min_db, max_db]
  RandomPolarityInvert  — random sign flip (phase-invariant codecs)
  LowPassFilter         — sinc low-pass via FFT convolution
  HighPassFilter        — sinc high-pass via FFT convolution
  BandDropout           — zero out random STFT frequency bands
  RoomImpulseResponse   — synthetic RIR convolution (simple exponential decay)
  PitchShift            — resampling-based pitch shift  ±n semitones
  TimeStretch           — phase-vocoder-lite stretch via resampling

AudioAugmentationPipeline — composable, config-driven pipeline
build_augmentation_pipeline — factory from OmegaConf config
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor


# ─── Base class ──────────────────────────────────────────────────────────────

class AudioTransform:
    """
    Base class for waveform transforms.
    All transforms accept / return (B, 1, T) or (1, T) float tensors.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.p = p

    def apply(self, wav: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, wav: Tensor) -> Tensor:
        if random.random() < self.p:
            return self.apply(wav)
        return wav

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ─── 1. Gaussian Noise ────────────────────────────────────────────────────────

class GaussianNoise(AudioTransform):
    """
    Add white Gaussian noise at a random SNR between [min_snr_db, max_snr_db].
    SNR-controlled: σ_noise = σ_signal / 10^(SNR/20)
    """

    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 40.0, p: float = 0.5):
        super().__init__(p)
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db

    def apply(self, wav: Tensor) -> Tensor:
        snr_db  = random.uniform(self.min_snr, self.max_snr)
        sig_rms = wav.pow(2).mean().sqrt().clamp(min=1e-9)
        noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        noise = torch.randn_like(wav) * noise_rms
        return (wav + noise).clamp(-1.0, 1.0)


# ─── 2. Random Gain ───────────────────────────────────────────────────────────

class RandomGain(AudioTransform):
    """
    Scale the waveform by a random gain drawn from [min_db, max_db] dB.
    Applies loudness variation without introducing distortion.
    """

    def __init__(self, min_db: float = -6.0, max_db: float = 6.0, p: float = 0.5):
        super().__init__(p)
        self.min_db = min_db
        self.max_db = max_db

    def apply(self, wav: Tensor) -> Tensor:
        gain_db    = random.uniform(self.min_db, self.max_db)
        gain_linear = 10 ** (gain_db / 20.0)
        return (wav * gain_linear).clamp(-1.0, 1.0)


# ─── 3. Random Polarity Invert ────────────────────────────────────────────────

class RandomPolarityInvert(AudioTransform):
    """
    Randomly flip the sign of the waveform.
    Human hearing is insensitive to polarity — this forces codec robustness.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, wav: Tensor) -> Tensor:
        return -wav


# ─── 4. Low-Pass Filter ───────────────────────────────────────────────────────

class LowPassFilter(AudioTransform):
    """
    Sinc low-pass filter with a randomly sampled cutoff frequency.
    Simulates telephone, radio, or lossy compression bandlimiting.
    """

    def __init__(
        self,
        min_cutoff_hz: float = 2000.0,
        max_cutoff_hz: float = 8000.0,
        sample_rate:   int   = 24_000,
        num_taps:      int   = 127,
        p:             float = 0.3,
    ):
        super().__init__(p)
        self.min_fc    = min_cutoff_hz
        self.max_fc    = max_cutoff_hz
        self.sr        = sample_rate
        self.num_taps  = num_taps | 1   # ensure odd

    @staticmethod
    def _sinc_kernel(fc_norm: float, num_taps: int, device: torch.device) -> Tensor:
        """Build a windowed-sinc low-pass kernel, normalised cutoff fc_norm ∈ (0, 0.5)."""
        half = num_taps // 2
        n    = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        with torch.no_grad():
            h         = 2.0 * fc_norm * torch.sinc(2.0 * fc_norm * n)
            window    = torch.hann_window(num_taps, device=device)
            h         = h * window
            h         = h / h.sum()
        return h

    def apply(self, wav: Tensor) -> Tensor:
        fc      = random.uniform(self.min_fc, self.max_fc)
        fc_norm = fc / self.sr
        kernel  = self._sinc_kernel(fc_norm, self.num_taps, wav.device)
        pad     = self.num_taps // 2
        # wav: (..., T)  →  treat as (B*C, 1, T) for F.conv1d
        shape   = wav.shape
        x       = wav.reshape(-1, 1, shape[-1])
        k       = kernel.view(1, 1, -1)
        x       = F.conv1d(F.pad(x, (pad, pad), mode="reflect"), k)
        return x.reshape(shape)


# ─── 5. High-Pass Filter ─────────────────────────────────────────────────────

class HighPassFilter(AudioTransform):
    """
    Sinc high-pass filter with random cutoff.
    Simulates rumble removal or upsampled content.
    """

    def __init__(
        self,
        min_cutoff_hz: float = 80.0,
        max_cutoff_hz: float = 300.0,
        sample_rate:   int   = 24_000,
        num_taps:      int   = 127,
        p:             float = 0.3,
    ):
        super().__init__(p)
        self.min_fc   = min_cutoff_hz
        self.max_fc   = max_cutoff_hz
        self.sr       = sample_rate
        self.num_taps = num_taps | 1

    def apply(self, wav: Tensor) -> Tensor:
        fc      = random.uniform(self.min_fc, self.max_fc)
        fc_norm = fc / self.sr
        half    = self.num_taps // 2
        n       = torch.arange(-half, half + 1, dtype=torch.float32, device=wav.device)
        with torch.no_grad():
            h_lp   = 2.0 * fc_norm * torch.sinc(2.0 * fc_norm * n)
            window = torch.hann_window(self.num_taps, device=wav.device)
            h_lp   = h_lp * window
            h_lp   = h_lp / h_lp.sum()
            # high-pass = all-pass - low-pass
            all_pass        = torch.zeros_like(h_lp)
            all_pass[half]  = 1.0
            h_hp = all_pass - h_lp

        shape = wav.shape
        x     = wav.reshape(-1, 1, shape[-1])
        k     = h_hp.view(1, 1, -1)
        pad   = self.num_taps // 2
        x     = F.conv1d(F.pad(x, (pad, pad), mode="reflect"), k)
        return x.reshape(shape)


# ─── 6. Band Dropout ─────────────────────────────────────────────────────────

class BandDropout(AudioTransform):
    """
    Zero out N random frequency bands in the STFT domain.
    Forces the codec to learn frequency-independent representations.
    """

    def __init__(
        self,
        n_fft:     int   = 512,
        hop:       int   = 128,
        n_bands:   int   = 2,
        max_width: float = 0.05,   # fraction of total bins per zeroed band
        p:         float = 0.3,
    ):
        super().__init__(p)
        self.n_fft     = n_fft
        self.hop       = hop
        self.n_bands   = n_bands
        self.max_width = max_width

    def apply(self, wav: Tensor) -> Tensor:
        n_bins  = self.n_fft // 2 + 1
        shape   = wav.shape
        x       = wav.reshape(-1, shape[-1])          # (B, T)
        window  = torch.hann_window(self.n_fft, device=wav.device)
        results = []
        for b in range(x.shape[0]):
            spec = torch.stft(x[b], self.n_fft, self.hop, self.n_fft,
                              window, return_complex=True)           # (n_bins, frames)
            mask = torch.ones(n_bins, 1, device=wav.device)
            for _ in range(self.n_bands):
                w     = random.randint(1, max(1, int(n_bins * self.max_width)))
                start = random.randint(0, n_bins - w)
                mask[start : start + w] = 0.0
            spec = spec * mask
            rec  = torch.istft(spec, self.n_fft, self.hop, self.n_fft,
                               window, return_complex=False,
                               length=x.shape[-1])
            results.append(rec)
        return torch.stack(results).reshape(shape)


# ─── 7. Synthetic Room Impulse Response ──────────────────────────────────────

class RoomImpulseResponse(AudioTransform):
    """
    Convolve audio with a synthetic exponentially-decaying RIR.
    RT60 (reverberation time to -60 dB) is drawn randomly.
    No external IR dataset required.
    """

    def __init__(
        self,
        sample_rate:  int   = 24_000,
        min_rt60_ms:  float = 50.0,
        max_rt60_ms:  float = 800.0,
        p:            float = 0.3,
    ):
        super().__init__(p)
        self.sr         = sample_rate
        self.min_rt60   = min_rt60_ms / 1000.0
        self.max_rt60   = max_rt60_ms / 1000.0

    def _make_rir(self, rt60: float, device: torch.device) -> Tensor:
        """Simple exponential-decay noise RIR of length rt60 samples."""
        n_samples = int(rt60 * self.sr)
        n_samples = max(n_samples, 16)
        t         = torch.arange(n_samples, dtype=torch.float32, device=device)
        decay     = torch.exp(-6.9078 * t / n_samples)   # -60 dB at t=rt60
        noise     = torch.randn(n_samples, device=device)
        rir       = noise * decay
        rir       = rir / rir.abs().max().clamp(min=1e-9)
        return rir

    def apply(self, wav: Tensor) -> Tensor:
        rt60  = random.uniform(self.min_rt60, self.max_rt60)
        rir   = self._make_rir(rt60, wav.device)
        shape = wav.shape
        x     = wav.reshape(-1, 1, shape[-1])
        k     = rir.view(1, 1, -1)
        pad   = rir.shape[-1] - 1
        conv  = F.conv1d(F.pad(x, (pad, 0)), k)
        # Trim back to original length
        conv  = conv[..., :shape[-1]]
        # Normalise peak to avoid clipping
        peak  = conv.abs().max().clamp(min=1e-9)
        return (conv / peak * wav.abs().max().clamp(min=1e-9)).reshape(shape)


# ─── 8. Pitch Shift ──────────────────────────────────────────────────────────

class PitchShift(AudioTransform):
    """
    Resampling-based pitch shift (±n_semitones).
    Cheap approximation: resample then crop/pad back to original length.
    No phase vocoder complexity required for codec training augmentation.
    """

    def __init__(
        self,
        sample_rate:     int   = 24_000,
        min_semitones:   float = -2.0,
        max_semitones:   float = 2.0,
        p:               float = 0.3,
    ):
        super().__init__(p)
        self.sr            = sample_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def apply(self, wav: Tensor) -> Tensor:
        import torchaudio.functional as AF
        n_st    = random.uniform(self.min_semitones, self.max_semitones)
        ratio   = 2 ** (n_st / 12.0)                  # frequency multiplier
        new_sr  = int(self.sr * ratio)
        shape   = wav.shape
        x       = wav.reshape(-1, shape[-1])            # (B, T)
        shifted = AF.resample(x, new_sr, self.sr)       # change speed + pitch
        T       = shape[-1]
        if shifted.shape[-1] >= T:
            shifted = shifted[..., :T]
        else:
            shifted = F.pad(shifted, (0, T - shifted.shape[-1]))
        return shifted.reshape(shape)


# ─── 9. Time Stretch ─────────────────────────────────────────────────────────

class TimeStretch(AudioTransform):
    """
    Speed-change time stretch via resampling (no pitch change).
    Rate ∈ [1/max_factor, max_factor]; output is cropped/padded to input length.
    """

    def __init__(
        self,
        sample_rate: int   = 24_000,
        max_factor:  float = 1.1,   # max speed-up / slow-down ratio
        p:           float = 0.3,
    ):
        super().__init__(p)
        self.sr         = sample_rate
        self.max_factor = max_factor

    def apply(self, wav: Tensor) -> Tensor:
        import torchaudio.functional as AF
        rate    = random.uniform(1.0 / self.max_factor, self.max_factor)
        new_sr  = int(self.sr * rate)
        shape   = wav.shape
        x       = wav.reshape(-1, shape[-1])
        x       = AF.resample(x, self.sr, new_sr)     # resample to change tempo
        T       = shape[-1]
        if x.shape[-1] >= T:
            start = (x.shape[-1] - T) // 2
            x     = x[..., start : start + T]
        else:
            x = F.pad(x, (0, T - x.shape[-1]))
        return x.reshape(shape)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class AudioAugmentationPipeline:
    """
    Composable ordered pipeline of AudioTransforms.

    Usage:
        pipeline = AudioAugmentationPipeline([
            GaussianNoise(min_snr_db=15, max_snr_db=35, p=0.4),
            RandomGain(min_db=-6, max_db=6, p=0.5),
            RoomImpulseResponse(p=0.3),
        ])
        wav_aug = pipeline(wav)
    """

    def __init__(self, transforms: List[AudioTransform]):
        self.transforms = transforms

    def __call__(self, wav: Tensor) -> Tensor:
        for t in self.transforms:
            wav = t(wav)
        return wav

    def __repr__(self) -> str:
        lines = ["AudioAugmentationPipeline(["]
        for t in self.transforms:
            lines.append(f"  {t},")
        lines.append("])")
        return "\n".join(lines)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_augmentation_pipeline(cfg: DictConfig) -> Optional[AudioAugmentationPipeline]:
    """
    Build the augmentation pipeline from OmegaConf config.
    Returns None if cfg.data.augmentation.enabled is False or key is absent.

    Expected config shape (under data.augmentation):
        enabled: true
        noise:        { p: 0.4, min_snr_db: 10, max_snr_db: 40 }
        gain:         { p: 0.5, min_db: -6, max_db: 6 }
        polarity:     { p: 0.5 }
        lowpass:      { p: 0.2, min_cutoff_hz: 3000, max_cutoff_hz: 8000 }
        highpass:     { p: 0.2, min_cutoff_hz: 60,   max_cutoff_hz: 200  }
        band_dropout: { p: 0.2, n_bands: 2 }
        rir:          { p: 0.3, min_rt60_ms: 50, max_rt60_ms: 600 }
        pitch_shift:  { p: 0.2, min_semitones: -2, max_semitones: 2 }
        time_stretch: { p: 0.2, max_factor: 1.1 }
    """
    aug_cfg = getattr(cfg.data, "augmentation", None)
    if aug_cfg is None or not getattr(aug_cfg, "enabled", False):
        return None

    sr = cfg.audio.sample_rate
    transforms: List[AudioTransform] = []

    def _get(key):
        return getattr(aug_cfg, key, None)

    if c := _get("noise"):
        transforms.append(GaussianNoise(
            min_snr_db=getattr(c, "min_snr_db", 10),
            max_snr_db=getattr(c, "max_snr_db", 40),
            p=getattr(c, "p", 0.4),
        ))

    if c := _get("gain"):
        transforms.append(RandomGain(
            min_db=getattr(c, "min_db", -6),
            max_db=getattr(c, "max_db", 6),
            p=getattr(c, "p", 0.5),
        ))

    if c := _get("polarity"):
        transforms.append(RandomPolarityInvert(p=getattr(c, "p", 0.5)))

    if c := _get("lowpass"):
        transforms.append(LowPassFilter(
            min_cutoff_hz=getattr(c, "min_cutoff_hz", 3000),
            max_cutoff_hz=getattr(c, "max_cutoff_hz", 8000),
            sample_rate=sr,
            p=getattr(c, "p", 0.2),
        ))

    if c := _get("highpass"):
        transforms.append(HighPassFilter(
            min_cutoff_hz=getattr(c, "min_cutoff_hz", 60),
            max_cutoff_hz=getattr(c, "max_cutoff_hz", 200),
            sample_rate=sr,
            p=getattr(c, "p", 0.2),
        ))

    if c := _get("band_dropout"):
        transforms.append(BandDropout(
            n_bands=getattr(c, "n_bands", 2),
            p=getattr(c, "p", 0.2),
        ))

    if c := _get("rir"):
        transforms.append(RoomImpulseResponse(
            sample_rate=sr,
            min_rt60_ms=getattr(c, "min_rt60_ms", 50),
            max_rt60_ms=getattr(c, "max_rt60_ms", 600),
            p=getattr(c, "p", 0.3),
        ))

    if c := _get("pitch_shift"):
        transforms.append(PitchShift(
            sample_rate=sr,
            min_semitones=getattr(c, "min_semitones", -2),
            max_semitones=getattr(c, "max_semitones", 2),
            p=getattr(c, "p", 0.2),
        ))

    if c := _get("time_stretch"):
        transforms.append(TimeStretch(
            sample_rate=sr,
            max_factor=getattr(c, "max_factor", 1.1),
            p=getattr(c, "p", 0.2),
        ))

    return AudioAugmentationPipeline(transforms) if transforms else None
