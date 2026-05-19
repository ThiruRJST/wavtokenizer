"""Tests for losses/losses.py"""

import pytest
import torch

from losses import (
    hinge_loss_discriminator,
    hinge_loss_generator,
    feature_matching_loss,
    MelSpectrogramLoss,
    TotalGeneratorLoss,
)


@pytest.fixture
def fake_disc_outputs():
    """Simulated discriminator output lists."""
    torch.manual_seed(0)
    real_outs = [torch.randn(2, 100) for _ in range(3)]
    fake_outs = [torch.randn(2, 100) for _ in range(3)]
    return real_outs, fake_outs


@pytest.fixture
def fake_fmaps():
    """Simulated feature map lists (list of lists)."""
    real_fmaps = [[torch.randn(2, 32, 50), torch.randn(2, 64, 25)] for _ in range(3)]
    fake_fmaps = [[torch.randn(2, 32, 50), torch.randn(2, 64, 25)] for _ in range(3)]
    return real_fmaps, fake_fmaps


# ─── Hinge losses ────────────────────────────────────────────────────────────

class TestHingeLoss:
    def test_disc_loss_non_negative(self, fake_disc_outputs):
        real_outs, fake_outs = fake_disc_outputs
        loss = hinge_loss_discriminator(real_outs, fake_outs)
        assert loss.item() >= 0.0

    def test_disc_loss_scalar(self, fake_disc_outputs):
        real_outs, fake_outs = fake_disc_outputs
        loss = hinge_loss_discriminator(real_outs, fake_outs)
        assert loss.ndim == 0

    def test_gen_loss_scalar(self, fake_disc_outputs):
        _, fake_outs = fake_disc_outputs
        loss = hinge_loss_generator(fake_outs)
        assert loss.ndim == 0

    def test_perfect_disc_gives_zero_loss(self):
        """Discriminator perfectly separating real=+∞, fake=-∞ → loss ≈ 0."""
        real_outs = [torch.full((2, 10),  100.0)]
        fake_outs = [torch.full((2, 10), -100.0)]
        loss = hinge_loss_discriminator(real_outs, fake_outs)
        assert loss.item() < 1e-3

    def test_disc_loss_gradient(self, fake_disc_outputs):
        real_outs = [o.requires_grad_(True) for o in fake_disc_outputs[0]]
        fake_outs = [o.requires_grad_(True) for o in fake_disc_outputs[1]]
        loss = hinge_loss_discriminator(real_outs, fake_outs)
        loss.backward()
        assert any(o.grad is not None for o in real_outs + fake_outs)

    def test_averaged_over_subdiscs(self):
        """Loss should be the same whether we pass 1 or N sub-discriminators."""
        out_single  = [torch.tensor([[0.5, -0.5]])]
        out_doubled = [torch.tensor([[0.5, -0.5]]), torch.tensor([[0.5, -0.5]])]
        l1 = hinge_loss_discriminator(out_single,  out_single)
        l2 = hinge_loss_discriminator(out_doubled, out_doubled)
        assert abs(l1.item() - l2.item()) < 1e-5


# ─── Feature matching loss ───────────────────────────────────────────────────

class TestFeatureMatchingLoss:
    def test_zero_when_identical(self, fake_fmaps):
        real_fmaps, _ = fake_fmaps
        loss = feature_matching_loss(real_fmaps, real_fmaps)
        assert loss.item() < 1e-6

    def test_positive_for_different(self, fake_fmaps):
        real_fmaps, fake_fmaps_ = fake_fmaps
        loss = feature_matching_loss(real_fmaps, fake_fmaps_)
        assert loss.item() > 0.0

    def test_gradient_on_fake(self, fake_fmaps):
        real_fmaps, fake_fmaps_ = fake_fmaps
        fake_fmaps_grad = [[t.requires_grad_(True) for t in sub] for sub in fake_fmaps_]
        loss = feature_matching_loss(real_fmaps, fake_fmaps_grad)
        loss.backward()
        has_grad = any(t.grad is not None and t.grad.abs().sum() > 0
                       for sub in fake_fmaps_grad for t in sub)
        assert has_grad


# ─── MelSpectrogramLoss ───────────────────────────────────────────────────────

class TestMelSpectrogramLoss:
    def test_zero_for_identical(self, mel_loss):
        wav = torch.randn(2, 1, 4_800)
        loss = mel_loss(wav, wav.clone())
        assert loss.item() < 1e-4

    def test_positive_for_different(self, mel_loss):
        real = torch.randn(2, 1, 4_800)
        fake = torch.randn(2, 1, 4_800)
        loss = mel_loss(real, fake)
        assert loss.item() > 0.0

    def test_handles_length_mismatch(self, mel_loss):
        """Mel loss should trim to the shorter of the two without error."""
        real = torch.randn(2, 1, 5_000)
        fake = torch.randn(2, 1, 4_800)
        loss = mel_loss(real, fake)
        assert loss.item() > 0.0

    def test_gradient_flows_to_fake(self, mel_loss):
        real = torch.randn(2, 1, 4_800)
        fake = torch.randn(2, 1, 4_800, requires_grad=True)
        loss = mel_loss(real, fake)
        loss.backward()
        assert fake.grad is not None and fake.grad.abs().sum() > 0

    def test_multi_scale_from_config(self, cfg):
        fn = MelSpectrogramLoss.from_config(cfg)
        wav = torch.randn(1, 1, 4_800)
        loss = fn(wav, wav * 0.9)
        assert loss.ndim == 0


# ─── TotalGeneratorLoss ───────────────────────────────────────────────────────

class TestTotalGeneratorLoss:
    def test_breakdown_keys(self, gen_loss):
        adv = torch.tensor(1.0)
        fm  = torch.tensor(2.0)
        mel = torch.tensor(3.0)
        vq  = torch.tensor(0.5)
        total, bk = gen_loss(adv, fm, mel, vq)
        assert set(bk.keys()) == {"adv", "fm", "mel", "vq", "total"}

    def test_total_matches_weighted_sum(self, gen_loss):
        adv, fm, mel, vq = (torch.tensor(float(x)) for x in [1.0, 2.0, 3.0, 0.5])
        total, bk = gen_loss(adv, fm, mel, vq)
        expected = (
            gen_loss.lambda_adv * 1.0
            + gen_loss.lambda_fm  * 2.0
            + gen_loss.lambda_mel * 3.0
            + gen_loss.lambda_vq  * 0.5
        )
        assert abs(bk["total"] - expected) < 1e-4

    def test_gradient_flows(self, gen_loss):
        for i, name in enumerate(["adv", "fm", "mel", "vq"]):
            t = torch.tensor(1.0, requires_grad=True)
            vals = [torch.tensor(0.0)] * 4
            vals[i] = t
            total, _ = gen_loss(*vals)
            total.backward()
            assert t.grad is not None and t.grad.abs().sum() > 0, \
                f"No gradient for {name}"
