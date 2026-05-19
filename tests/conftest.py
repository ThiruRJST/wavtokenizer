"""
Shared pytest fixtures for WavTokenizer test suite.
All tests run on CPU with the 'small' config to keep CI fast.
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from utils import load_config, seed_everything
from model import (
    build_model,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiResolutionSTFTDiscriminator,
)
from losses import MelSpectrogramLoss, TotalGeneratorLoss
from data import build_dataloaders


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """Small config — no GPU required."""
    seed_everything(42)
    return load_config("small", ["project.run_name=pytest"])


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(cfg):
    return build_model(cfg)


@pytest.fixture(scope="session")
def mpd(cfg):
    return MultiPeriodDiscriminator.from_config(cfg)


@pytest.fixture(scope="session")
def msd(cfg):
    return MultiScaleDiscriminator.from_config(cfg)


@pytest.fixture(scope="session")
def mrstftd(cfg):
    return MultiResolutionSTFTDiscriminator.from_config(cfg)


@pytest.fixture(scope="session")
def mel_loss(cfg):
    return MelSpectrogramLoss.from_config(cfg)


@pytest.fixture(scope="session")
def gen_loss(cfg):
    return TotalGeneratorLoss.from_config(cfg)


@pytest.fixture(scope="session")
def dummy_wav():
    """1-second mono 24 kHz waveform, batch size 2."""
    torch.manual_seed(0)
    return torch.randn(2, 1, 24_000)


@pytest.fixture(scope="session")
def short_wav():
    """0.1-second mono waveform for fast tests."""
    torch.manual_seed(1)
    return torch.randn(2, 1, 2_400)


@pytest.fixture(scope="session")
def dataloaders(cfg):
    return build_dataloaders(cfg)
