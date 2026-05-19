"""Tests for data/dataset.py"""

import pytest
import torch
from torch.utils.data import DataLoader

from data import (
    DummyAudioDataset,
    AudioFileDataset,
    AugmentedDataset,
    build_dataloaders,
)
from data.augmentation import GaussianNoise, AudioAugmentationPipeline


# ─── DummyAudioDataset ────────────────────────────────────────────────────────

class TestDummyAudioDataset:
    def test_len(self):
        ds = DummyAudioDataset(num_samples=10)
        assert len(ds) == 10

    def test_item_shape(self):
        ds  = DummyAudioDataset(num_samples=4, duration_sec=0.5, sample_rate=24_000)
        wav = ds[0]
        assert wav.shape == (1, 12_000)

    def test_custom_duration(self):
        ds  = DummyAudioDataset(num_samples=2, duration_sec=2.0, sample_rate=24_000)
        assert ds[0].shape == (1, 48_000)

    def test_dataloader_batch_shape(self):
        ds = DummyAudioDataset(num_samples=8, duration_sec=0.2, sample_rate=24_000)
        dl = DataLoader(ds, batch_size=4)
        batch = next(iter(dl))
        assert batch.shape == (4, 1, 4_800)

    def test_values_finite(self):
        ds = DummyAudioDataset(num_samples=5)
        for i in range(5):
            assert torch.isfinite(ds[i]).all()


# ─── AugmentedDataset ─────────────────────────────────────────────────────────

class TestAugmentedDataset:
    def test_len_matches_base(self):
        base     = DummyAudioDataset(num_samples=10)
        pipeline = AudioAugmentationPipeline([GaussianNoise(p=1.0)])
        aug_ds   = AugmentedDataset(base, pipeline)
        assert len(aug_ds) == len(base)

    def test_item_shape_preserved(self):
        base     = DummyAudioDataset(num_samples=4, duration_sec=0.5, sample_rate=24_000)
        pipeline = AudioAugmentationPipeline([GaussianNoise(p=1.0)])
        aug_ds   = AugmentedDataset(base, pipeline)
        assert aug_ds[0].shape == base[0].shape

    def test_augmentation_applied(self):
        """With p=1, augmented output must differ from base output."""
        base     = DummyAudioDataset(num_samples=4, duration_sec=0.2, sample_rate=24_000)
        pipeline = AudioAugmentationPipeline([GaussianNoise(min_snr_db=5, max_snr_db=5, p=1.0)])
        aug_ds   = AugmentedDataset(base, pipeline)
        assert not torch.allclose(aug_ds[0], base[0])


# ─── build_dataloaders ────────────────────────────────────────────────────────

class TestBuildDataloaders:
    def test_returns_two_loaders(self, cfg):
        train_dl, val_dl = build_dataloaders(cfg)
        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl,   DataLoader)

    def test_batch_shape(self, cfg):
        train_dl, _ = build_dataloaders(cfg)
        batch = next(iter(train_dl))
        sr      = cfg.audio.sample_rate
        seg_len = int(cfg.audio.segment_duration * sr)
        assert batch.shape == (cfg.training.batch_size, 1, seg_len)

    def test_train_has_more_samples_than_val(self, cfg):
        train_dl, val_dl = build_dataloaders(cfg)
        assert len(train_dl.dataset) > len(val_dl.dataset)

    def test_float32_dtype(self, cfg):
        train_dl, _ = build_dataloaders(cfg)
        batch = next(iter(train_dl))
        assert batch.dtype == torch.float32

    def test_no_nan_in_batches(self, cfg):
        train_dl, val_dl = build_dataloaders(cfg)
        for loader in [train_dl, val_dl]:
            for batch in loader:
                assert torch.isfinite(batch).all(), "NaN/Inf in batch"
                break   # one batch each is enough
