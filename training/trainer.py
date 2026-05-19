"""
WavTokenizer Trainer
─────────────────────
Full training loop with:
  • Mixed-precision (torch.amp — bf16 on Ampere/RTX 5080, fp16 fallback)
  • Separate GradScalers for generator and discriminator
  • Linear LR warm-up then exponential decay
  • Delayed adversarial start (disc_start_epoch)
  • Per-step TensorBoard / W&B logging
  • Periodic checkpoint saving with best-model tracking
  • Graceful resume from checkpoint
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from model import (
    WavTokenizer,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiResolutionSTFTDiscriminator,
)
from losses import (
    hinge_loss_discriminator,
    hinge_loss_generator,
    feature_matching_loss,
    MelSpectrogramLoss,
    TotalGeneratorLoss,
)
from utils import (
    get_logger,
    TBWriter,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    cfg_to_dict,
)


logger = get_logger("trainer")


# ─── Optimizer factory ───────────────────────────────────────────────────────

def _make_optimizer(
    params, cfg: DictConfig
) -> torch.optim.Optimizer:
    kwargs = dict(
        lr           = cfg.optimizer.lr_g,
        betas        = tuple(cfg.optimizer.betas),
        weight_decay = cfg.optimizer.weight_decay,
        eps          = cfg.optimizer.eps,
    )
    if cfg.optimizer.type == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if cfg.optimizer.type == "adam":
        return torch.optim.Adam(params, **kwargs)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer.type}")


# ─── LR scheduler with linear warm-up ───────────────────────────────────────

class WarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_steps:  int,
        gamma:         float,
        decay_type:    str = "exponential",
    ):
        self.warmup_steps = warmup_steps
        self.gamma        = gamma
        self.decay_type   = decay_type

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            if decay_type == "exponential":
                return gamma ** (step - warmup_steps)
            return 1.0

        super().__init__(optimizer, lr_lambda)


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full WavTokenizer training procedure."""

    def __init__(
        self,
        cfg:          DictConfig,
        model:        WavTokenizer,
        mpd:          MultiPeriodDiscriminator,
        msd:          MultiScaleDiscriminator,
        mrstftd:      MultiResolutionSTFTDiscriminator,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
        resume_path:  Optional[str] = None,
    ):
        self.cfg    = cfg
        self.device = device

        # ── Move all modules to device ────────────────────────────────────────
        self.model   = model.to(device)
        self.mpd     = mpd.to(device)
        self.msd     = msd.to(device)
        self.mrstftd = mrstftd.to(device)

        # ── torch.compile (PyTorch ≥ 2.0, optional) ──────────────────────────
        if cfg.training.compile and hasattr(torch, "compile"):
            logger.info("torch.compile enabled — compiling model …")
            self.model = torch.compile(self.model)

        # ── Loss functions ────────────────────────────────────────────────────
        self.mel_loss_fn = MelSpectrogramLoss.from_config(cfg).to(device)
        self.gen_loss_fn = TotalGeneratorLoss.from_config(cfg)

        # ── Optimizers ────────────────────────────────────────────────────────
        self.opt_g = _make_optimizer(self.model.parameters(), cfg)
        disc_params = (
            list(self.mpd.parameters())
            + list(self.msd.parameters())
            + list(self.mrstftd.parameters())
        )
        disc_cfg       = DictConfig({**cfg_to_dict(cfg), "optimizer": {**cfg_to_dict(cfg.optimizer), "lr_g": cfg.optimizer.lr_d}})
        self.opt_d     = _make_optimizer(disc_params, disc_cfg)

        # ── LR schedulers ────────────────────────────────────────────────────
        self.sched_g = WarmupScheduler(
            self.opt_g,
            cfg.scheduler.warmup_steps,
            cfg.scheduler.gamma,
            cfg.scheduler.type,
        )
        self.sched_d = WarmupScheduler(
            self.opt_d,
            cfg.scheduler.warmup_steps,
            cfg.scheduler.gamma,
            cfg.scheduler.type,
        )

        # ── Mixed precision ───────────────────────────────────────────────────
        self.use_amp  = cfg.training.mixed_precision and device.type == "cuda"
        amp_dtype_str = cfg.training.amp_dtype
        self.amp_dtype = (
            torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
        )
        self.scaler_g = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.scaler_d = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # ── State ─────────────────────────────────────────────────────────────
        self.epoch      = 0
        self.global_step = 0
        self.best_mel   = float("inf")

        # ── Logging ───────────────────────────────────────────────────────────
        run_dir = Path(cfg.project.log_dir) / cfg.project.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.tb = TBWriter(str(run_dir), enabled=cfg.project.tensorboard)

        try:
            from utils.logging import WandbLogger
            self.wb = WandbLogger(cfg)
        except Exception:
            self.wb = None

        # ── Resume ────────────────────────────────────────────────────────────
        if resume_path:
            meta = load_checkpoint(
                resume_path,
                self.model, self.opt_g, self.opt_d,
                self.sched_g, self.sched_d,
                self.scaler_g, self.scaler_d,
                device=device,
            )
            self.epoch       = meta["epoch"]
            self.global_step = meta["step"]
            self.best_mel    = meta["metrics"].get("mel_loss", float("inf"))
            logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")

    # ── Context manager for AMP ──────────────────────────────────────────────

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
        return torch.amp.autocast(device_type="cpu", enabled=False)

    # ── Single training step ─────────────────────────────────────────────────

    def _train_step(
        self,
        real: torch.Tensor,
        disc_active: bool,
    ) -> Dict[str, float]:
        real = real.to(self.device, non_blocking=True)
        B, C, T = real.shape

        # ════ Discriminator update ════════════════════════════════════════════
        with self._autocast():
            with torch.no_grad():
                fake, _ = self.model(real)

        min_len = min(real.shape[-1], fake.shape[-1])
        r = real[..., :min_len]
        f = fake[..., :min_len].detach()

        if disc_active:
            self.opt_d.zero_grad(set_to_none=True)

            with self._autocast():
                ro_mpd, fo_mpd, _, _ = self.mpd(r, f)
                ro_msd, fo_msd, _, _ = self.msd(r, f)
                ro_mr,  fo_mr,  _, _ = self.mrstftd(r, f)
                d_loss = (
                    hinge_loss_discriminator(ro_mpd, fo_mpd)
                    + hinge_loss_discriminator(ro_msd, fo_msd)
                    + hinge_loss_discriminator(ro_mr,  fo_mr)
                ) / 3.0

            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.unscale_(self.opt_d)
            nn.utils.clip_grad_norm_(
                list(self.mpd.parameters())
                + list(self.msd.parameters())
                + list(self.mrstftd.parameters()),
                self.cfg.training.grad_clip,
            )
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
        else:
            d_loss = torch.tensor(0.0, device=self.device)

        # ════ Generator update ════════════════════════════════════════════════
        self.opt_g.zero_grad(set_to_none=True)

        with self._autocast():
            fake, vq_loss = self.model(real)
            min_len = min(real.shape[-1], fake.shape[-1])
            r = real[..., :min_len]
            f = fake[..., :min_len]

            if disc_active:
                _, fo_mpd, rfm_mpd, ffm_mpd = self.mpd(r, f)
                _, fo_msd, rfm_msd, ffm_msd = self.msd(r, f)
                _, fo_mr,  rfm_mr,  ffm_mr  = self.mrstftd(r, f)

                adv_loss = (
                    hinge_loss_generator(fo_mpd)
                    + hinge_loss_generator(fo_msd)
                    + hinge_loss_generator(fo_mr)
                ) / 3.0
                fm_loss = (
                    feature_matching_loss(rfm_mpd, ffm_mpd)
                    + feature_matching_loss(rfm_msd, ffm_msd)
                    + feature_matching_loss(rfm_mr,  ffm_mr)
                ) / 3.0
            else:
                adv_loss = torch.tensor(0.0, device=self.device)
                fm_loss  = torch.tensor(0.0, device=self.device)

            mel_loss = self.mel_loss_fn(r, f)
            g_loss, breakdown = self.gen_loss_fn(adv_loss, fm_loss, mel_loss, vq_loss)

        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.unscale_(self.opt_g)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        # Step schedulers every step
        self.sched_g.step()
        self.sched_d.step()

        return {
            "g_total": breakdown["total"],
            "g_adv":   breakdown["adv"],
            "g_fm":    breakdown["fm"],
            "g_mel":   breakdown["mel"],
            "g_vq":    breakdown["vq"],
            "d_loss":  d_loss.item(),
            "codebook_util": self.model.quantizer.codebook_utilization * 100,
        }

    # ── Training epoch ───────────────────────────────────────────────────────

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.mpd.train(); self.msd.train(); self.mrstftd.train()

        meters = {k: AverageMeter() for k in
                  ["g_total", "g_mel", "g_vq", "d_loss", "codebook_util"]}
        disc_active = (self.epoch >= self.cfg.training.disc_start_epoch)
        t0 = time.time()

        for step, real in enumerate(self.train_loader):
            metrics = self._train_step(real, disc_active)
            self.global_step += 1

            for k, v in metrics.items():
                if k in meters:
                    meters[k].update(v)

            # ── Per-step logging ─────────────────────────────────────────────
            if self.global_step % self.cfg.training.log_interval == 0:
                elapsed = time.time() - t0
                lr_g = self.opt_g.param_groups[0]["lr"]
                logger.info(
                    f"Ep {self.epoch:04d} | step {self.global_step:07d} "
                    f"| G={metrics['g_total']:.4f} "
                    f"mel={metrics['g_mel']:.4f} "
                    f"vq={metrics['g_vq']:.4f} "
                    f"D={metrics['d_loss']:.4f} "
                    f"cb={metrics['codebook_util']:.1f}% "
                    f"lr={lr_g:.2e} "
                    f"({elapsed:.1f}s)"
                )
                self.tb.scalars(metrics, self.global_step, prefix="train")
                self.tb.scalar("lr/generator",     lr_g, self.global_step)
                self.tb.scalar("lr/discriminator", self.opt_d.param_groups[0]["lr"], self.global_step)

                if self.wb:
                    self.wb.log({f"train/{k}": v for k, v in metrics.items()}, self.global_step)

                t0 = time.time()

        return {k: m.avg for k, m in meters.items()}

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        meters = {k: AverageMeter() for k in ["mel_loss", "codebook_util"]}

        for real in self.val_loader:
            real = real.to(self.device, non_blocking=True)
            with self._autocast():
                fake, _ = self.model(real)
            min_len = min(real.shape[-1], fake.shape[-1])
            mel = self.mel_loss_fn(real[..., :min_len], fake[..., :min_len])
            meters["mel_loss"].update(mel.item())
            meters["codebook_util"].update(
                self.model.quantizer.codebook_utilization * 100
            )

        metrics = {k: m.avg for k, m in meters.items()}
        logger.info(
            f"[VAL] Ep {self.epoch:04d} | "
            f"mel={metrics['mel_loss']:.4f} "
            f"cb_util={metrics['codebook_util']:.1f}%"
        )
        self.tb.scalars(metrics, self.global_step, prefix="val")
        if self.wb:
            self.wb.log({f"val/{k}": v for k, v in metrics.items()}, self.global_step)
        return metrics

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(self):
        """Run the full training procedure."""
        cfg = self.cfg
        start_epoch = self.epoch + 1

        logger.info(
            f"Starting training  epochs={cfg.training.epochs}  "
            f"device={self.device}  "
            f"amp={self.use_amp} ({cfg.training.amp_dtype})"
        )

        for epoch in range(start_epoch, cfg.training.epochs + 1):
            self.epoch = epoch
            logger.info(f"──── Epoch {epoch}/{cfg.training.epochs} ────")

            train_metrics = self.train_epoch()

            # ── Evaluation ──────────────────────────────────────────────────
            if epoch % cfg.training.eval_interval == 0:
                val_metrics = self.validate()
                is_best = val_metrics["mel_loss"] < self.best_mel
                if is_best:
                    self.best_mel = val_metrics["mel_loss"]
                    logger.info(f"  ★ New best mel={self.best_mel:.4f}")
            else:
                is_best = False
                val_metrics = {}

            # ── Checkpoint ──────────────────────────────────────────────────
            if epoch % cfg.training.save_interval == 0 or is_best:
                ckpt_path = save_checkpoint(
                    checkpoint_dir = cfg.project.checkpoint_dir,
                    epoch          = epoch,
                    step           = self.global_step,
                    model          = self.model,
                    opt_g          = self.opt_g,
                    opt_d          = self.opt_d,
                    sched_g        = self.sched_g,
                    sched_d        = self.sched_d,
                    scaler_g       = self.scaler_g,
                    scaler_d       = self.scaler_d,
                    metrics        = {**train_metrics, **val_metrics},
                    cfg_dict       = cfg_to_dict(cfg),
                    keep_last_n    = cfg.training.keep_last_n,
                    is_best        = is_best,
                )
                logger.info(f"  Saved checkpoint → {ckpt_path}")

        self.tb.close()
        if self.wb:
            self.wb.finish()
        logger.info("Training complete.")
