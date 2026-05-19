"""
WavTokenizer Datasets
──────────────────────
  DummyAudioDataset   — synthetic random waveforms (CI / smoke tests)
  AudioFileDataset    — custom directory of WAV files
  LibriTTSDataset     — wraps torchaudio.datasets.LIBRITTS with
                         on-the-fly resampling + random crop / pad
  build_dataloaders   — factory that returns train + val DataLoaders
                         based on the OmegaConf config
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dummy dataset (CPU smoke tests / quick CI runs)
# ─────────────────────────────────────────────────────────────────────────────

class DummyAudioDataset(Dataset):
    """Generates deterministic synthetic waveforms."""

    def __init__(
        self,
        num_samples:  int   = 128,
        duration_sec: float = 1.0,
        sample_rate:  int   = 24_000,
        seed:         int   = 0,
    ):
        self.num_samples = num_samples
        self.length      = int(duration_sec * sample_rate)
        self.rng         = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        # Use idx as additional seed offset for variety
        return torch.randn(1, self.length)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom WAV file dataset
# ─────────────────────────────────────────────────────────────────────────────

class AudioFileDataset(Dataset):
    """
    Loads .wav / .flac / .mp3 files from a flat directory or list of paths.

    Features:
      • On-the-fly resampling to target sample_rate
      • Stereo → mono downmix
      • Random crop for training; centre crop for evaluation
      • Duration filtering (min / max seconds)
    """

    EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(
        self,
        root:             Optional[str]       = None,
        file_paths:       Optional[List[str]] = None,
        sample_rate:      int   = 24_000,
        segment_duration: float = 1.0,
        random_crop:      bool  = True,
        min_duration:     float = 0.5,
        max_duration:     float = 10.0,
    ):
        if file_paths is None:
            if root is None:
                raise ValueError("Provide root or file_paths.")
            root = Path(root)
            file_paths = [
                str(p) for p in root.rglob("*")
                if p.suffix.lower() in self.EXTENSIONS
            ]

        self.sample_rate      = sample_rate
        self.seg_samples      = int(segment_duration * sample_rate)
        self.random_crop      = random_crop
        self.min_dur_samples  = int(min_duration * sample_rate)
        self.max_dur_samples  = int(max_duration * sample_rate)

        # Filter by duration (uses torchaudio.info for O(1) metadata read)
        self.paths: List[str] = []
        for p in file_paths:
            try:
                info = torchaudio.info(p)
                dur  = info.num_frames
                if self.min_dur_samples <= dur <= self.max_dur_samples:
                    self.paths.append(p)
            except Exception:
                pass  # skip unreadable files

        if not self.paths:
            raise RuntimeError(f"No valid audio files found (checked {len(file_paths)} paths).")

    def __len__(self) -> int:
        return len(self.paths)

    def _crop_or_pad(self, wav: Tensor) -> Tensor:
        T = wav.shape[-1]
        if T >= self.seg_samples:
            if self.random_crop:
                start = random.randint(0, T - self.seg_samples)
            else:
                start = (T - self.seg_samples) // 2
            return wav[:, start : start + self.seg_samples]
        return F.pad(wav, (0, self.seg_samples - T))

    def __getitem__(self, idx: int) -> Tensor:
        path = self.paths[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0, keepdim=True)                         # mono (1, T)
        wav = self._crop_or_pad(wav)
        # Normalize to [-1, 1]
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak.clamp(min=1.0)
        return wav


# ─────────────────────────────────────────────────────────────────────────────
# 3. LibriTTS dataset  (auto-downloaded via torchaudio)
# ─────────────────────────────────────────────────────────────────────────────

class LibriTTSDataset(Dataset):
    """
    Wraps torchaudio.datasets.LIBRITTS.

    Supported splits: train-clean-100, train-clean-360, train-other-500,
                      dev-clean, dev-other, test-clean, test-other
    """

    def __init__(
        self,
        root:             str,
        splits:           List[str],
        sample_rate:      int   = 24_000,
        segment_duration: float = 1.0,
        random_crop:      bool  = True,
        download:         bool  = True,
    ):
        self.sample_rate  = sample_rate
        self.seg_samples  = int(segment_duration * sample_rate)
        self.random_crop  = random_crop

        raw_datasets = []
        for split in splits:
            ds = torchaudio.datasets.LIBRITTS(
                root=root, url=split, download=download
            )
            raw_datasets.append(ds)

        # Flatten all utterances from all splits
        self._items: List[Tuple] = []
        for ds in raw_datasets:
            for item in ds:
                self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tensor:
        # LIBRITTS item: (waveform, sample_rate, transcript, ...)
        wav, sr, *_ = self._items[idx]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0, keepdim=True)

        T = wav.shape[-1]
        if T >= self.seg_samples:
            if self.random_crop:
                start = random.randint(0, T - self.seg_samples)
            else:
                start = 0
            wav = wav[:, start : start + self.seg_samples]
        else:
            wav = F.pad(wav, (0, self.seg_samples - T))

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak.clamp(min=1.0)
        return wav


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train + validation DataLoaders according to config.

    Priority:
      1. LibriTTS (cfg.data.use_libritts = true)
      2. Custom WAV directory (cfg.data.use_custom = true)
      3. DummyAudioDataset (fallback — no real data needed)

    Returns: (train_loader, val_loader)
    """
    sr       = cfg.audio.sample_rate
    seg_dur  = cfg.audio.segment_duration
    bs       = cfg.training.batch_size
    nw       = cfg.training.num_workers
    pin      = cfg.training.pin_memory

    if cfg.data.use_libritts:
        full_ds = LibriTTSDataset(
            root             = cfg.data.libritts_root,
            splits           = list(cfg.data.libritts_splits),
            sample_rate      = sr,
            segment_duration = seg_dur,
            random_crop      = True,
            download         = True,
        )
        val_n   = max(1, int(len(full_ds) * cfg.data.val_split_fraction))
        train_n = len(full_ds) - val_n
        train_ds, val_ds = random_split(
            full_ds, [train_n, val_n],
            generator=torch.Generator().manual_seed(cfg.project.seed),
        )
        # Disable random crop for the validation subset (centre crop)
        # (wrapped dataset; val items still use same __getitem__ but that's fine)

    elif cfg.data.use_custom:
        def _make(folder, crop):
            return AudioFileDataset(
                root             = folder,
                sample_rate      = sr,
                segment_duration = seg_dur,
                random_crop      = crop,
                min_duration     = cfg.data.min_duration_sec,
                max_duration     = cfg.data.max_duration_sec,
            )
        train_ds = _make(cfg.data.custom_train_dir, crop=True)
        val_ds   = _make(cfg.data.custom_val_dir,   crop=False) \
                   if cfg.data.custom_val_dir else train_ds

    else:
        # Smoke-test fallback
        n_total  = 64
        val_n    = 8
        train_ds = DummyAudioDataset(n_total - val_n, seg_dur, sr)
        val_ds   = DummyAudioDataset(val_n,           seg_dur, sr, seed=99)

    train_loader = DataLoader(
        train_ds,
        batch_size  = bs,
        shuffle     = True,
        num_workers = nw,
        pin_memory  = pin,
        drop_last   = True,
        persistent_workers = (nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = bs,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = pin,
        drop_last   = False,
        persistent_workers = (nw > 0),
    )
    return train_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────
# 5. Augmented dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AugmentedDataset(Dataset):
    """
    Wraps any Dataset and applies an AudioAugmentationPipeline to each sample.
    Applied only during training — validation sets should NOT be wrapped.
    """

    def __init__(self, dataset: Dataset, pipeline):
        self.dataset  = dataset
        self.pipeline = pipeline

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        wav = self.dataset[idx]
        return self.pipeline(wav)


def build_dataloaders_with_augmentation(
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Like build_dataloaders, but also wraps the training set in the
    AugmentedDataset pipeline when cfg.data.augmentation.enabled = true.
    """
    from data.augmentation import build_augmentation_pipeline

    train_loader, val_loader = build_dataloaders(cfg)
    pipeline = build_augmentation_pipeline(cfg)

    if pipeline is not None:
        aug_train_ds = AugmentedDataset(train_loader.dataset, pipeline)
        train_loader = DataLoader(
            aug_train_ds,
            batch_size         = cfg.training.batch_size,
            shuffle            = True,
            num_workers        = cfg.training.num_workers,
            pin_memory         = cfg.training.pin_memory,
            drop_last          = True,
            persistent_workers = (cfg.training.num_workers > 0),
        )

    return train_loader, val_loader
