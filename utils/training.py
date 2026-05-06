"""
Training utilities:
  - ExponentialMovingAverage (EMA)
  - Checkpointing
  - WandB / TensorBoard logging
  - FID score computation
  - Sample grid saving
"""

import os
import math
import copy
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from torchvision.utils import make_grid, save_image


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains a shadow copy of model parameters updated with EMA.
    EMA models are significantly more stable for generating high-quality samples.
    Usage:
        ema = EMA(model, decay=0.9999)
        # after each optimizer.step():
        ema.update(model)
        # for sampling:
        with ema.average_parameters():
            samples = diffusion.ddim_sample_loop(model, ...)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: torch.nn.Module):
        """Copy EMA weights into model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name])
        self.original.clear()

    class average_parameters:
        """Context manager: temporarily apply EMA weights."""
        def __init__(self, ema_instance, model):
            self.ema = ema_instance
            self.model = model

        def __enter__(self):
            self.ema.apply_shadow(self.model)
            return self.model

        def __exit__(self, *args):
            self.ema.restore(self.model)

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, d):
        self.shadow = d['shadow']
        self.decay = d.get('decay', self.decay)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: str,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    scheduler=None,
    best: bool = False,
    config: Optional[Dict] = None,
):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    fname = 'best.pt' if best else f'ckpt_epoch{epoch:04d}_step{step:07d}.pt'
    path = os.path.join(output_dir, fname)

    payload = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'config': config,
    }
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMA] = None,
    scheduler=None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Load training checkpoint. Returns metadata dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if ema and 'ema_state_dict' in ckpt:
        ema.load_state_dict(ckpt['ema_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    print(f"Loaded checkpoint: {path} (epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')})")
    return ckpt


# ---------------------------------------------------------------------------
# Sample Grid
# ---------------------------------------------------------------------------

def save_sample_grid(
    samples: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
):
    """Save a grid of generated samples as PNG."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # samples expected in [-1,1]; normalize to [0,1] for saving
    grid = make_grid(samples, nrow=nrow, normalize=normalize, value_range=(-1, 1))
    save_image(grid, path)
    return path


# ---------------------------------------------------------------------------
# Learning Rate Warmup + Cosine Decay Schedule
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine decay.
    Used as a callable for torch LambdaLR.
    """
    def __init__(self, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay


# ---------------------------------------------------------------------------
# Logging (WandB + TensorBoard with graceful fallback)
# ---------------------------------------------------------------------------

class Logger:
    """
    Unified logger that tries wandb first, falls back to TensorBoard, then stdout.
    """

    def __init__(self, config: Dict, project: str = 'ddpm-ddim', use_wandb: bool = True, use_tb: bool = True):
        self.config = config
        self.wandb_run = None
        self.tb_writer = None
        self._step = 0

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project, config=config, resume='allow')
                print(f"WandB run: {self.wandb_run.url}")
            except Exception as e:
                print(f"WandB init failed ({e}), falling back to TensorBoard/stdout.")

        if use_tb and self.wandb_run is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(config.get('output_dir', 'outputs'), 'tb_logs')
                self.tb_writer = SummaryWriter(tb_dir)
                print(f"TensorBoard logs: {tb_dir}")
            except Exception as e:
                print(f"TensorBoard unavailable ({e}), using stdout only.")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        step = step or self._step
        self._step = step + 1

        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        # Always print
        metric_str = '  '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        print(f"[step {step:7d}] {metric_str}")

    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None):
        step = step or self._step
        if self.wandb_run:
            try:
                import wandb
                grid = make_grid(images, normalize=True, value_range=(-1, 1))
                self.wandb_run.log({tag: wandb.Image(grid.permute(1, 2, 0).cpu().numpy())}, step=step)
            except Exception:
                pass
        if self.tb_writer:
            grid = make_grid(images, normalize=True, value_range=(-1, 1))
            self.tb_writer.add_image(tag, grid, step)

    def finish(self):
        if self.wandb_run:
            self.wandb_run.finish()
        if self.tb_writer:
            self.tb_writer.close()


# ---------------------------------------------------------------------------
# FID Score (requires clean-fid or torch-fidelity)
# ---------------------------------------------------------------------------

def compute_fid(
    real_dir: str,
    fake_dir: str,
    device: str = 'cuda',
    method: str = 'clean-fid',
) -> float:
    """
    Compute FID between two image directories.
    Install: pip install clean-fid   OR   pip install torch-fidelity
    """
    try:
        if method == 'clean-fid':
            from cleanfid import fid
            score = fid.compute_fid(real_dir, fake_dir, device=device)
        else:
            import torch_fidelity
            metrics = torch_fidelity.calculate_metrics(
                input1=real_dir, input2=fake_dir,
                fid=True, cuda=(device == 'cuda')
            )
            score = metrics['frechet_inception_distance']
        print(f"FID: {score:.2f}")
        return score
    except ImportError:
        print("FID computation requires: pip install clean-fid")
        return -1.0


def save_samples_for_fid(
    samples: torch.Tensor,
    out_dir: str,
    start_idx: int = 0,
):
    """Save individual sample images for FID computation."""
    os.makedirs(out_dir, exist_ok=True)
    samples_01 = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
    for i, img in enumerate(samples_01):
        path = os.path.join(out_dir, f'{start_idx + i:06d}.png')
        save_image(img, path)


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def grad_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
