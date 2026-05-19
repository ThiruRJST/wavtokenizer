#!/usr/bin/env python3
"""
WavTokenizer — Profiler & Throughput Benchmark
────────────────────────────────────────────────
Measures:
  • Encode / decode / full-forward throughput  (samples/s, real-time factor)
  • Peak GPU / CPU memory usage
  • Per-module time breakdown  (torch.profiler)
  • Mixed-precision vs full-precision comparison

Usage:
  # Quick benchmark with small config
  python scripts/profile.py --config small

  # Full model on GPU with AMP
  python scripts/profile.py --config default \
      training.mixed_precision=true

  # Export Chrome trace for visualization in chrome://tracing
  python scripts/profile.py --trace profile_trace.json
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import DictConfig

from model import build_model
from utils import load_config, count_parameters, get_logger

logger = get_logger("profile")


# ─── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def _timer(label: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    logger.info(f"  {label:<30} {elapsed * 1000:8.2f} ms")


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0


def _reset_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _autocast(use_amp: bool, dtype: torch.dtype):
    if use_amp and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.amp.autocast("cpu", enabled=False)


# ─── Benchmark: throughput ───────────────────────────────────────────────────

def benchmark_throughput(
    model:         torch.nn.Module,
    cfg:           DictConfig,
    device:        torch.device,
    batch_size:    int   = 4,
    n_warmup:      int   = 5,
    n_runs:        int   = 20,
) -> Dict[str, float]:
    """
    Measure tokens/s and real-time factor for encode, decode, and roundtrip.
    """
    sr       = cfg.audio.sample_rate
    seg_dur  = cfg.audio.segment_duration
    n_samples = int(seg_dur * sr)
    tok_rate = cfg.audio.token_rate
    n_tokens = int(seg_dur * tok_rate)

    wav = torch.randn(batch_size, 1, n_samples, device=device)
    idx = torch.randint(0, cfg.quantizer.codebook_size,
                        (batch_size, n_tokens), device=device)

    use_amp  = cfg.training.mixed_precision and device.type == "cuda"
    amp_type = torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16

    model.eval()
    results: Dict[str, float] = {}

    def _run_timed(fn, label, n_audio_sec: float):
        # Warm-up
        for _ in range(n_warmup):
            with _autocast(use_amp, amp_type):
                fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(n_runs):
            with _autocast(use_amp, amp_type):
                fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed   = time.perf_counter() - t0
        avg_ms    = elapsed / n_runs * 1000
        total_sec = batch_size * seg_dur          # audio-seconds per batch
        throughput = total_sec / (elapsed / n_runs)  # audio-s / wall-s
        rtf        = 1.0 / throughput             # wall-s per audio-s

        results[f"{label}_ms"]   = avg_ms
        results[f"{label}_rtf"]  = rtf
        logger.info(f"  {label:<18}  {avg_ms:8.2f} ms/batch  "
                    f"RTF={rtf:.4f}  ({throughput:.1f}× real-time)")

    logger.info(f"\nThroughput  [bs={batch_size}, {seg_dur}s, "
                f"{sr}Hz, {'AMP' if use_amp else 'FP32'}]")
    logger.info("─" * 65)

    with torch.no_grad():
        _run_timed(lambda: model.encode(wav),    "encode",    seg_dur)
        _run_timed(lambda: model.decode(idx),    "decode",    seg_dur)
        _run_timed(lambda: model(wav),           "roundtrip", seg_dur)

    return results


# ─── Benchmark: memory ────────────────────────────────────────────────────────

def benchmark_memory(
    model:      torch.nn.Module,
    cfg:        DictConfig,
    device:     torch.device,
    batch_size: int = 4,
) -> Dict[str, float]:
    if device.type != "cuda":
        logger.info("  (Memory profiling only available on CUDA)")
        return {}

    sr       = cfg.audio.sample_rate
    seg_dur  = cfg.audio.segment_duration
    n_samples = int(seg_dur * sr)
    tok_rate  = cfg.audio.token_rate
    n_tokens  = int(seg_dur * tok_rate)
    wav = torch.randn(batch_size, 1, n_samples, device=device)

    use_amp  = cfg.training.mixed_precision
    amp_type = torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16

    results = {}
    model.eval()

    logger.info(f"\nPeak Memory  [bs={batch_size}]")
    logger.info("─" * 65)

    for label, fn in [
        ("encode",    lambda: model.encode(wav)),
        ("decode",    lambda: model.decode(
                          torch.randint(0, cfg.quantizer.codebook_size,
                                        (batch_size, n_tokens), device=device))),
        ("roundtrip", lambda: model(wav)),
    ]:
        _reset_memory(device)
        with torch.no_grad(), _autocast(use_amp, amp_type):
            fn()
        mb = _peak_memory_mb(device)
        results[f"{label}_peak_mb"] = mb
        logger.info(f"  {label:<18}  {mb:.1f} MB peak")

    return results


# ─── Benchmark: per-module timing ─────────────────────────────────────────────

def benchmark_module_timing(
    model:      torch.nn.Module,
    cfg:        DictConfig,
    device:     torch.device,
    batch_size: int   = 2,
    trace_path: str   = None,
):
    """Use torch.profiler to get per-operator timing."""
    sr        = cfg.audio.sample_rate
    seg_dur   = cfg.audio.segment_duration
    n_samples = int(seg_dur * sr)
    wav       = torch.randn(batch_size, 1, n_samples, device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    logger.info("\nPer-module timing (torch.profiler)")
    logger.info("─" * 65)

    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(
            activities       = activities,
            record_shapes    = True,
            profile_memory   = True,
            with_stack       = False,
        ) as prof:
            for _ in range(3):
                model(wav)

    # Print top-20 operators by self CUDA / CPU time
    key_avg = prof.key_averages()
    print(key_avg.table(sort_by="self_cpu_time_total", row_limit=20))

    if trace_path:
        prof.export_chrome_trace(trace_path)
        logger.info(f"Chrome trace exported → {trace_path}")


# ─── Parameter count summary ─────────────────────────────────────────────────

def print_parameter_summary(cfg: DictConfig, device: torch.device):
    from model import (
        MultiPeriodDiscriminator,
        MultiScaleDiscriminator,
        MultiResolutionSTFTDiscriminator,
    )
    model   = build_model(cfg).to(device)
    mpd     = MultiPeriodDiscriminator.from_config(cfg)
    msd     = MultiScaleDiscriminator.from_config(cfg)
    mrstftd = MultiResolutionSTFTDiscriminator.from_config(cfg)

    g_total = count_parameters(model)
    d_total = (count_parameters(mpd) + count_parameters(msd)
               + count_parameters(mrstftd))

    logger.info("\nParameter counts")
    logger.info("─" * 65)
    logger.info(f"  {'WavTokenizer':<30} {g_total:>12,}")
    logger.info(f"  {'  encoder':<30} {count_parameters(model.encoder):>12,}")
    logger.info(f"  {'  quantizer (codebook)':<30} {count_parameters(model.quantizer):>12,}")
    logger.info(f"  {'  decoder':<30} {count_parameters(model.decoder):>12,}")
    logger.info(f"  {'MultiPeriodDisc':<30} {count_parameters(mpd):>12,}")
    logger.info(f"  {'MultiScaleDisc':<30} {count_parameters(msd):>12,}")
    logger.info(f"  {'MultiResSTFTDisc':<30} {count_parameters(mrstftd):>12,}")
    logger.info(f"  {'Total (G + D)':<30} {g_total + d_total:>12,}")
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="WavTokenizer Profiler")
    p.add_argument("--config",     default="default")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_runs",     type=int, default=20)
    p.add_argument("--trace",      default=None, help="Export chrome trace to path")
    p.add_argument("--skip_profiler", action="store_true",
                   help="Skip per-operator torch.profiler (saves time)")
    p.add_argument("overrides",    nargs="*")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config, list(args.overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Profile  config={args.config}  device={device}")

    model = print_parameter_summary(cfg, device)
    model.to(device)

    benchmark_throughput(model, cfg, device, args.batch_size, n_runs=args.n_runs)
    benchmark_memory(model, cfg, device, args.batch_size)

    if not args.skip_profiler:
        benchmark_module_timing(model, cfg, device,
                                batch_size=min(args.batch_size, 2),
                                trace_path=args.trace)


if __name__ == "__main__":
    main()
