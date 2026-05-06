"""
Main Training Script for DDPM/DDIM with MSRF UNet

Usage:
  # CIFAR-10 unconditional (quick test):
  python train.py --config configs/cifar10_uncond.yaml

  # CIFAR-10 class-conditional with CFG:
  python train.py --config configs/cifar10_cond.yaml

  # Resume from checkpoint:
  python train.py --config configs/cifar10_cond.yaml --resume outputs/ckpts/ckpt_epoch0010_step0050000.pt

  # Override config values via CLI:
  python train.py --config configs/cifar10_cond.yaml training.batch_size=64 training.lr=1e-4
"""

import os
import sys
import time
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.unet import UNet
from models.diffusion import GaussianDiffusion
from data.datasets import build_dataloaders
from utils.training import (
    EMA, Logger, WarmupCosineScheduler,
    save_checkpoint, load_checkpoint,
    save_sample_grid, save_samples_for_fid,
    grad_norm, count_parameters,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply key=value overrides to nested config dict."""
    for override in overrides:
        if '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to infer type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
        d[keys[-1]] = value
    return cfg


def dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert dict to SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device detection (CUDA > MPS > CPU)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Build model from config
# ---------------------------------------------------------------------------

def build_model(cfg, num_classes: int) -> UNet:
    m = cfg.model
    return UNet(
        image_size=m.image_size,
        in_channels=m.in_channels,
        base_channels=m.base_channels,
        channel_mults=tuple(m.channel_mults),
        num_res_blocks=m.num_res_blocks,
        attn_resolutions=tuple(m.attn_resolutions),
        dropout=m.dropout,
        num_classes=num_classes if num_classes > 0 else None,
        use_msrf=m.use_msrf,
    )


# ---------------------------------------------------------------------------
# Sampling during training
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model, diffusion, cfg, device,
    num_samples: int = 16,
    class_labels=None,
    use_ddim: bool = True,
):
    model.eval()
    shape = (num_samples, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)

    if use_ddim:
        samples = diffusion.ddim_sample_loop(
            model, shape, device,
            ddim_steps=cfg.sampling.ddim_steps,
            class_labels=class_labels,
            cfg_scale=cfg.sampling.cfg_scale,
            null_class=cfg.sampling.null_class if hasattr(cfg.sampling, 'null_class') else None,
            eta=cfg.sampling.eta,
            progress=False,
        )
    else:
        samples = diffusion.p_sample_loop(
            model, shape, device,
            class_labels=class_labels,
            cfg_scale=cfg.sampling.cfg_scale,
            null_class=cfg.sampling.null_class if hasattr(cfg.sampling, 'null_class') else None,
            progress=False,
        )

    model.train()
    return samples


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg, resume_path: str = None):
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"Training on: {device}")

    # ------ Data ------
    train_loader, val_loader, num_classes = build_dataloaders(
        dataset_name=cfg.data.dataset,
        data_root=cfg.data.root,
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        augment=cfg.data.augment,
    )
    print(f"Dataset: {cfg.data.dataset} | Classes: {num_classes} | "
          f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ------ Model ------
    model = build_model(cfg, num_classes).to(device)
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M  (MSRF: {cfg.model.use_msrf})")

    # ------ Diffusion ------
    null_class = num_classes if num_classes > 0 else None
    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion.timesteps,
        schedule=cfg.diffusion.schedule,
        loss_type=cfg.diffusion.loss_type,
    )

    # ------ Optimizer & Scheduler ------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = cfg.training.epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)
    lr_schedule = WarmupCosineScheduler(warmup_steps, total_steps, min_lr_ratio=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    # ------ EMA ------
    ema = EMA(model, decay=cfg.training.ema_decay)

    # ------ Logger ------
    logger = Logger(
        config=vars(cfg) if hasattr(cfg, '__dict__') else {},
        project='ddpm-ddim-msrf',
        use_wandb=cfg.logging.use_wandb,
        use_tb=cfg.logging.use_tensorboard,
    )

    # ------ Directories ------
    ckpt_dir = os.path.join(cfg.training.output_dir, 'ckpts')
    sample_dir = os.path.join(cfg.training.output_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # ------ Resume ------
    start_epoch = 0
    global_step = 0
    if resume_path:
        meta = load_checkpoint(resume_path, model, optimizer, ema, scheduler, str(device))
        start_epoch = meta.get('epoch', 0) + 1
        global_step = meta.get('step', 0)

    # ------ Fixed class labels for samples ------
    sample_labels = None
    if num_classes > 0:
        n_per_class = max(1, 16 // num_classes)
        sample_labels = torch.arange(num_classes).repeat_interleave(n_per_class)[:16].to(device)

    # ------ Training ------
    model.train()
    best_loss = float('inf')

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            t = torch.randint(0, diffusion.timesteps, (imgs.shape[0],), device=device)

            class_labels_batch = None
            if num_classes > 0:
                class_labels_batch = labels.to(device)

            loss = diffusion.compute_loss(
                model, imgs, t,
                class_labels=class_labels_batch,
                cfg_dropout_prob=cfg.training.cfg_dropout_prob,
                null_class=null_class,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            global_step += 1

            # ------ Logging ------
            if global_step % cfg.logging.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                gn = grad_norm(model)
                logger.log({
                    'train/loss': loss.item(),
                    'train/lr': lr,
                    'train/grad_norm': gn,
                    'train/epoch': epoch,
                }, step=global_step)

            # ------ Sample generation ------
            if global_step % cfg.logging.sample_every == 0:
                with ema.average_parameters(ema, model):
                    samples = generate_samples(
                        model, diffusion, cfg, device,
                        num_samples=16,
                        class_labels=sample_labels,
                        use_ddim=True,
                    )
                grid_path = os.path.join(sample_dir, f'step_{global_step:07d}.png')
                save_sample_grid(samples, grid_path)
                logger.log_images('samples/ddim', samples[:16], step=global_step)
                print(f"  → Saved samples: {grid_path}")

        # ------ End of epoch ------
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"\nEpoch [{epoch+1}/{cfg.training.epochs}] "
              f"loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # Save checkpoint every N epochs
        if (epoch + 1) % cfg.logging.save_every_epochs == 0:
            save_checkpoint(ckpt_dir, epoch, global_step, model, optimizer, ema, scheduler,
                            config=cfg.__dict__ if hasattr(cfg, '__dict__') else {})

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(ckpt_dir, epoch, global_step, model, optimizer, ema, scheduler,
                            best=True, config=cfg.__dict__ if hasattr(cfg, '__dict__') else {})

    logger.finish()
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('overrides', nargs='*', help='Config overrides: key.subkey=value')
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg_dict = apply_overrides(cfg_dict, args.overrides)
    cfg = dict_to_ns(cfg_dict)

    os.makedirs(cfg.training.output_dir, exist_ok=True)

    train(cfg, resume_path=args.resume)
