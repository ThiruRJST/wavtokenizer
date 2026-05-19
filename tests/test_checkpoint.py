"""Tests for utils/checkpoint.py"""

import os
import glob
import pytest
import torch

from utils import save_checkpoint, load_checkpoint, prune_checkpoints
from model import build_model


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def opt_pair(model):
    opt_g = torch.optim.AdamW(model.parameters(), lr=2e-4)
    opt_d = torch.optim.AdamW(model.parameters(), lr=2e-4)
    return opt_g, opt_d


class TestSaveLoadCheckpoint:
    def test_save_creates_file(self, model, opt_pair, tmp_ckpt_dir):
        opt_g, opt_d = opt_pair
        path = save_checkpoint(
            tmp_ckpt_dir, epoch=1, step=100,
            model=model, opt_g=opt_g, opt_d=opt_d,
            sched_g=None, sched_d=None,
        )
        assert os.path.isfile(path)

    def test_saved_file_contains_keys(self, model, opt_pair, tmp_ckpt_dir):
        opt_g, opt_d = opt_pair
        path = save_checkpoint(
            tmp_ckpt_dir, epoch=3, step=500,
            model=model, opt_g=opt_g, opt_d=opt_d,
            sched_g=None, sched_d=None,
            metrics={"mel_loss": 0.42},
        )
        ckpt = torch.load(path, map_location="cpu")
        for key in ["epoch", "step", "model", "opt_g", "opt_d", "metrics"]:
            assert key in ckpt, f"Key '{key}' missing from checkpoint"
        assert ckpt["epoch"] == 3
        assert ckpt["step"]  == 500
        assert ckpt["metrics"]["mel_loss"] == pytest.approx(0.42)

    def test_load_restores_epoch_step(self, cfg, model, opt_pair, tmp_ckpt_dir):
        opt_g, opt_d = opt_pair
        path = save_checkpoint(
            tmp_ckpt_dir, epoch=7, step=1234,
            model=model, opt_g=opt_g, opt_d=opt_d,
            sched_g=None, sched_d=None,
        )
        model2 = build_model(cfg)
        meta   = load_checkpoint(path, model2, device=torch.device("cpu"))
        assert meta["epoch"] == 7
        assert meta["step"]  == 1234

    def test_load_restores_model_weights(self, cfg, model, opt_pair, tmp_ckpt_dir, short_wav):
        """Model reloaded from checkpoint must give bit-identical output."""
        opt_g, opt_d = opt_pair
        model.eval()
        with torch.no_grad():
            ref_out, _ = model(short_wav)

        path   = save_checkpoint(
            tmp_ckpt_dir, epoch=1, step=10,
            model=model, opt_g=opt_g, opt_d=opt_d,
            sched_g=None, sched_d=None,
        )
        model2 = build_model(cfg)
        load_checkpoint(path, model2, device=torch.device("cpu"))
        model2.eval()
        with torch.no_grad():
            new_out, _ = model2(short_wav)

        assert torch.allclose(ref_out, new_out), "Reloaded model output differs"

    def test_best_checkpoint_saved(self, model, opt_pair, tmp_ckpt_dir):
        opt_g, opt_d = opt_pair
        save_checkpoint(
            tmp_ckpt_dir, epoch=1, step=100,
            model=model, opt_g=opt_g, opt_d=opt_d,
            sched_g=None, sched_d=None,
            is_best=True,
        )
        best_path = os.path.join(tmp_ckpt_dir, "best.pth")
        assert os.path.isfile(best_path)


class TestPruneCheckpoints:
    def test_keeps_last_n(self, model, opt_pair, tmp_ckpt_dir):
        opt_g, opt_d = opt_pair
        for i in range(6):
            save_checkpoint(
                tmp_ckpt_dir, epoch=i, step=i * 10,
                model=model, opt_g=opt_g, opt_d=opt_d,
                sched_g=None, sched_d=None,
                keep_last_n=3,
            )
        remaining = glob.glob(os.path.join(tmp_ckpt_dir, "ckpt_epoch*.pth"))
        assert len(remaining) == 3, f"Expected 3 checkpoints, got {len(remaining)}"

    def test_most_recent_are_kept(self, model, opt_pair, tmp_ckpt_dir):
        opt_g, opt_d = opt_pair
        for i in range(5):
            save_checkpoint(
                tmp_ckpt_dir, epoch=i, step=i,
                model=model, opt_g=opt_g, opt_d=opt_d,
                sched_g=None, sched_d=None,
                keep_last_n=2,
            )
        remaining = sorted(glob.glob(os.path.join(tmp_ckpt_dir, "ckpt_epoch*.pth")))
        # Should keep epoch 3 and 4
        assert "epoch0003" in remaining[-2]
        assert "epoch0004" in remaining[-1]
