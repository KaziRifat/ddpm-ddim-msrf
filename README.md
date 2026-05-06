# DDPM/DDIM from Scratch with Multi-Scale Residual Fusion (MSRF)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, research-grade PyTorch implementation of **DDPM** and **DDIM** diffusion models from scratch, featuring:

- 🏗️ **Full UNet with multi-head self-attention** — Residual blocks, AdaGN time conditioning, sinusoidal embeddings  
- ⚡ **DDPM + DDIM sampling** — Full 1000-step ancestral + deterministic/stochastic fast sampling with configurable steps  
- 🎯 **Classifier-Free Guidance (CFG)** — Class-conditional generation with guidance scale sweep  
- ⭐ **Architectural twist: Multi-Scale Residual Fusion (MSRF)** — Each decoder level fuses ALL encoder scales via learned, time-conditioned gates  
- 📊 **Complete training pipeline** — EMA, gradient clipping, warmup+cosine LR, W&B/TensorBoard logging, checkpointing  
- 📐 **Evaluation** — FID computation via `clean-fid`, sample grids, eta sweep visualization

---

## ⭐ Architectural Innovation: Multi-Scale Residual Fusion (MSRF)

Standard UNets use a single skip connection per decoder level (from the *matching* encoder level). **MSRF** instead aggregates skip connections from **all** encoder levels into each decoder block, fusing them via learned time-conditioned gate weights:

```
Encoder Level 1 ─┐
Encoder Level 2 ─┤─► MSRFGate(t) ─► Weighted Fusion ─► Decoder Level i
Encoder Level 3 ─┤       ↑
Encoder Level 4 ─┘   gates from t
```

**Why it works:**
- At **high noise levels** (early denoising), the model learns to rely on deep, semantically-rich encoder features
- At **low noise levels** (fine-detail refinement), it shifts weight toward shallow, high-resolution features  
- The gating is differentiable and conditioned on the timestep embedding, so the model discovers this behavior automatically

Toggle with `use_msrf: true/false` in any config to ablate.

---

## Project Structure

```
ddpm-ddim-msrf/
├── models/
│   ├── unet.py          # UNet architecture (MSRF, attention, time conditioning)
│   └── diffusion.py     # DDPM/DDIM schedules, forward/reverse processes, CFG
├── data/
│   └── datasets.py      # CIFAR-10/100, CelebA, Tiny ImageNet, custom folder
├── utils/
│   └── training.py      # EMA, checkpointing, logging, FID, LR schedule
├── configs/
│   ├── cifar10_uncond.yaml   # CIFAR-10 unconditional
│   ├── cifar10_cond.yaml     # CIFAR-10 class-conditional + CFG
│   ├── celeba64.yaml         # CelebA 64×64 faces
│   └── debug.yaml            # Tiny config for quick smoke test
├── notebooks/
│   └── explore.ipynb    # Visualizations, CFG sweep, DDPM vs DDIM
├── train.py             # Main training script
├── sample.py            # Inference / sampling script
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/ddpm-ddim-msrf
cd ddpm-ddim-msrf
pip install -r requirements.txt
```

**Supported platforms:** CUDA GPU, Apple Silicon (MPS), CPU  
**Python:** 3.8+ | **PyTorch:** 2.0+

### 2. Smoke test (CPU/MPS, ~minutes)

```bash
python train.py --config configs/debug.yaml
```

Verify the forward pass, data loading, sampling, and checkpointing all work before committing to a full run.

### 3. Full training

**Unconditional CIFAR-10:**
```bash
python train.py --config configs/cifar10_uncond.yaml
```

**Class-conditional CIFAR-10 with CFG:**
```bash
python train.py --config configs/cifar10_cond.yaml
```

**Resume from checkpoint:**
```bash
python train.py --config configs/cifar10_cond.yaml \
    --resume outputs/cifar10_cond/ckpts/ckpt_epoch0050_step0200000.pt
```

**Override config on CLI:**
```bash
python train.py --config configs/cifar10_cond.yaml \
    training.batch_size=64 training.lr=1e-4
```

### 4. Generate samples

```bash
# Fast DDIM sampling (50 steps)
python sample.py \
    --checkpoint outputs/cifar10_cond/ckpts/best.pt \
    --config configs/cifar10_cond.yaml \
    --sampler ddim --steps 50 --cfg_scale 3.0 --n_samples 64

# Generate a specific class (0=airplane, 1=car, ...)
python sample.py \
    --checkpoint outputs/cifar10_cond/ckpts/best.pt \
    --config configs/cifar10_cond.yaml \
    --class_label 5 --n_samples 16

# Visualize eta sweep (deterministic → stochastic)
python sample.py \
    --checkpoint outputs/cifar10_cond/ckpts/best.pt \
    --config configs/cifar10_cond.yaml \
    --eta_sweep

# Generate 50k samples for FID evaluation
python sample.py \
    --checkpoint outputs/cifar10_cond/ckpts/best.pt \
    --config configs/cifar10_cond.yaml \
    --n_samples 50000 --fid_mode \
    --output_dir outputs/fid_samples/
```

---

## Configuration Reference

All hyperparameters live in YAML config files. Key settings:

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `use_msrf` | Enable/disable MSRF twist |
| `model` | `base_channels` | Base channel width (128 = standard) |
| `model` | `channel_mults` | Channel multipliers per level |
| `model` | `attn_resolutions` | Spatial resolutions to apply attention |
| `diffusion` | `schedule` | `cosine` (recommended) / `linear` / `quadratic` |
| `diffusion` | `timesteps` | Diffusion steps (default: 1000) |
| `training` | `cfg_dropout_prob` | Prob. of dropping class label (CFG training) |
| `training` | `ema_decay` | EMA decay (0.9999 for stable samples) |
| `sampling` | `cfg_scale` | Guidance weight w (1.0 = no guidance) |
| `sampling` | `eta` | DDIM stochasticity: 0=deterministic, 1=DDPM |

---

## Implementation Details

### Forward Process
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t} x_0,\ (1-\bar\alpha_t)\mathbf{I})$$

### Cosine Schedule
$$\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

### Training Objective (L_simple)
$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### DDIM Update Rule
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}{\sqrt{\bar\alpha_t}}}_{\text{predicted }x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1} - \sigma_t^2}\,\hat\epsilon_\theta}_{\text{direction}} + \underbrace{\sigma_t\epsilon_t}_{\text{noise}}$$

### Classifier-Free Guidance
$$\hat\epsilon = \epsilon_\theta(x_t, \emptyset) + w\cdot\bigl(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\bigr)$$

---

## Expected Results

| Dataset | Model | Steps | FID ↓ |
|---------|-------|-------|-------|
| CIFAR-10 | UNet (128ch, MSRF) | DDPM 1000 | ~8–12 |
| CIFAR-10 | UNet (128ch, MSRF) | DDIM 50 | ~10–15 |
| CIFAR-10 + CFG | UNet (128ch, MSRF) | DDIM 50, w=3 | ~5–9 |

*Results depend on hardware, training duration, and batch size. CIFAR-10 typically shows clear image structure after 50–100 epochs.*

---

## Hardware Notes

| Hardware | Batch Size | Expected Time (500 epochs) |
|----------|------------|---------------------------|
| RTX 3090 / A100 | 128 | ~12–24 hours |
| M1/M2/M3 Pro (MPS) | 32–64 | ~2–4 days |
| CPU only | 8–16 | Not recommended |

For MPS (Apple Silicon), the code auto-selects `mps` backend. Some operations fall back to CPU — this is expected.

---

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)
- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
