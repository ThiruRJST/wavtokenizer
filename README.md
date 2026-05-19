# WavTokenizer — PyTorch Implementation

> **ICLR 2025 paper:** [WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](https://arxiv.org/abs/2408.16532)

A clean, modular, production-ready PyTorch implementation of WavTokenizer — a neural audio codec that compresses 24 kHz audio into **40 or 75 discrete tokens per second** using a single vector quantizer.

---

## Project Structure

```
wavtokenizer/
├── configs/
│   ├── default.yaml        ← all hyperparameters
│   └── small.yaml          ← tiny model for fast iteration / CI
│
├── model/
│   ├── encoder.py          ← Conv1D + ResidualUnit + strided downsampling
│   ├── quantizer.py        ← VQ with K-means init + random awakening
│   ├── decoder.py          ← ConvNeXt + SelfAttention + iSTFT upsampling
│   ├── discriminators.py   ← MPD, MSD, Multi-resolution STFT disc
│   └── wavtokenizer.py     ← top-level model + factory
│
├── losses/
│   └── losses.py           ← hinge adv, feature-matching, mel, total-G
│
├── data/
│   └── dataset.py          ← DummyAudio, AudioFile, LibriTTS, dataloader factory
│
├── training/
│   └── trainer.py          ← full training loop, AMP, grad scaler, LR warm-up
│
├── evaluation/
│   └── evaluator.py        ← mel loss, SNR, codebook utilization, PESQ, STOI
│
├── utils/
│   ├── config.py           ← OmegaConf loader + CLI merge
│   ├── logging.py          ← logger, TensorBoard, optional W&B
│   ├── checkpoint.py       ← save / load / prune checkpoints
│   └── misc.py             ← seed, param count, AverageMeter
│
└── scripts/
    ├── train.py            ← training entry point (single-GPU + DDP)
    ├── evaluate.py         ← evaluation entry point
    └── inference.py        ← encode / decode a WAV file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke test (no real data needed)

```bash
# Uses DummyAudioDataset + small config (CPU-friendly)
python scripts/train.py --config small
```

### 3. Train on LibriTTS

```bash
python scripts/train.py --config default \
    data.use_libritts=true              \
    data.libritts_root=/data/libritts   \
    training.batch_size=16              \
    project.run_name=libritts_75tok
```

### 4. Train on your own WAV files

```bash
python scripts/train.py --config default \
    data.use_custom=true                  \
    data.custom_train_dir=/data/train_wavs \
    data.custom_val_dir=/data/val_wavs
```

### 5. Multi-GPU training (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train.py --config default \
    training.batch_size=4           # per-GPU batch size
```

### 6. Resume from checkpoint

```bash
python scripts/train.py --resume checkpoints/ckpt_epoch0050_step0123456.pth
```

### 7. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --config default
```

### 8. Round-trip encode/decode

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best.pth \
    --input  audio/speech.wav         \
    --output audio/reconstructed.wav  \
    --save_tokens tokens.pt
```

---

## Key Config Options

| Key | Default | Description |
|-----|---------|-------------|
| `audio.token_rate` | `75` | 40 or 75 tokens/s |
| `quantizer.codebook_size` | `4096` | VQ codebook size (expand for better quality) |
| `encoder.base_channels` | `32` | First-stage channel width (doubles per stage) |
| `decoder.n_convnext` | `8` | ConvNeXt blocks in decoder |
| `training.mixed_precision` | `true` | Enable bf16/fp16 AMP |
| `training.amp_dtype` | `bfloat16` | `bfloat16` (RTX 3090+) or `float16` |
| `training.batch_size` | `16` | Per-GPU batch size |
| `loss.lambda_mel` | `45.0` | Multi-scale mel reconstruction weight |
| `training.disc_start_epoch` | `0` | Delay adversarial training by N epochs |

---

## Architecture Summary

```
Raw Audio (B, 1, T)
        │
        ▼
   ┌──────────┐    Conv1D + [ResUnit(d=1,3,9) + StridedConv] × 4
   │ Encoder  │    Strides: 2×4×5×8=320 → 75 tok/s
   └────┬─────┘
        │  z  (B, 512, T/320)
        ▼
   ┌──────────┐    Single VQ, codebook_size=4096
   │    VQ    │    K-means init + random awakening
   └────┬─────┘    Straight-through gradient
        │  z_q  (B, 512, T/320)
        ▼
   ┌──────────┐    Conv1D → SelfAttention → 8×ConvNeXt → iSTFT
   │ Decoder  │    iSTFT eliminates transposed-conv aliasing
   └────┬─────┘
        │
        ▼
Reconstructed Audio (B, 1, T)
```

## Loss Functions

| Loss | Symbol | Weight |
|------|--------|--------|
| Multi-scale Mel L1 | L_mel | 45.0 |
| Feature Matching | L_fm | 2.0 |
| Adversarial (hinge) | L_adv | 1.0 |
| VQ commitment | L_vq | 1.0 |

---

## Citation

```bibtex
@inproceedings{ji2025wavtokenizer,
  title     = {WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer
               for Audio Language Modeling},
  author    = {Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and others},
  booktitle = {ICLR},
  year      = {2025},
}
```
