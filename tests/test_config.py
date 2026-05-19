"""Tests for utils/config.py"""

import pytest
from omegaconf import OmegaConf

from utils import load_config


class TestLoadConfig:
    def test_default_loads(self):
        cfg = load_config("default")
        assert cfg.audio.sample_rate == 24_000
        assert cfg.audio.token_rate  == 75
        assert cfg.quantizer.codebook_size == 4096

    def test_small_inherits_from_default(self):
        """Small config must have all keys from default."""
        default = load_config("default")
        small   = load_config("small")
        for key in ["audio", "encoder", "quantizer", "decoder", "training", "loss", "data"]:
            assert hasattr(small, key), f"Key '{key}' missing in small config"

    def test_small_overrides_encoder(self):
        small = load_config("small")
        assert small.encoder.base_channels == 8
        assert small.encoder.latent_dim    == 64

    def test_small_inherits_sample_rate(self):
        small = load_config("small")
        assert small.audio.sample_rate == 24_000

    def test_hop_length_auto_filled(self):
        cfg = load_config("default")
        expected = cfg.audio.sample_rate // cfg.audio.token_rate
        assert cfg.audio.hop_length == expected

    def test_cli_override_batch_size(self):
        cfg = load_config("small", overrides=["training.batch_size=8"])
        assert cfg.training.batch_size == 8

    def test_cli_override_nested(self):
        cfg = load_config("small", overrides=["quantizer.codebook_size=512"])
        assert cfg.quantizer.codebook_size == 512

    def test_multiple_overrides(self):
        cfg = load_config("small", overrides=[
            "training.batch_size=2",
            "audio.token_rate=40",
            "project.run_name=test_run",
        ])
        assert cfg.training.batch_size  == 2
        assert cfg.audio.token_rate     == 40
        assert cfg.project.run_name     == "test_run"

    def test_invalid_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config_xyz")

    def test_cfg_to_dict(self):
        from utils import cfg_to_dict
        cfg  = load_config("small")
        d    = cfg_to_dict(cfg)
        assert isinstance(d, dict)
        assert "audio" in d
        assert isinstance(d["audio"]["sample_rate"], int)
