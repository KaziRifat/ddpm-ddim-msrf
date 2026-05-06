"""
Inference Script: Sample from a trained DDPM/DDIM model.

Usage examples:
  # DDIM sampling (fast, 50 steps):
  python sample.py --checkpoint outputs/ckpts/best.pt --config configs/cifar10_cond.yaml \
                   --sampler ddim --steps 50 --n_samples 64 --cfg_scale 3.0

  # DDPM sampling (full 1000 steps):
  python sample.py --checkpoint outputs/ckpts/best.pt --config configs/cifar10_cond.yaml \
                   --sampler ddpm --n_samples 16

  # Generate a specific class:
  python sample.py --checkpoint outputs/ckpts/best.pt --config configs/cifar10_cond.yaml \
                   --sampler ddim --class_label 3 --n_samples 16

  # Generate for FID evaluation (saves individual files):
  python sample.py --checkpoint outputs/ckpts/best.pt --config configs/cifar10_cond.yaml \
                   --sampler ddim --n_samples 50000 --fid_mode --output_dir fid_samples/

  # Sweep eta (DDIM stochasticity):
  python sample.py --checkpoint outputs/ckpts/best.pt --config configs/cifar10_cond.yaml \
                   --eta_sweep
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent))

from models.unet import UNet
from models.diffusion import GaussianDiffusion
from utils.training import (
    EMA, load_checkpoint, save_sample_grid,
    save_samples_for_fid, count_parameters,
)


# ---------------------------------------------------------------------------
# Helpers (duplicated from train.py so this script is self-contained)
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dict_to_ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model_from_cfg(cfg, num_classes):
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
# Core sampling function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(
    model,
    diffusion,
    device,
    image_size,
    in_channels,
    n_samples,
    batch_size,
    sampler,
    ddim_steps,
    eta,
    cfg_scale,
    class_label,
    num_classes,
    null_class,
):
    """Run batch-wise sampling and collect all results."""
    all_samples = []

    batches = [batch_size] * (n_samples // batch_size)
    remainder = n_samples % batch_size
    if remainder:
        batches.append(remainder)

    for i, bs in enumerate(batches):
        shape = (bs, in_channels, image_size, image_size)

        # Build class labels
        labels = None
        if num_classes > 0 and class_label is not None:
            labels = torch.full((bs,), class_label, dtype=torch.long, device=device)
        elif num_classes > 0:
            # Cycle through all classes uniformly
            labels = torch.arange(bs, device=device) % num_classes

        if sampler == 'ddim':
            samples = diffusion.ddim_sample_loop(
                model, shape, device,
                ddim_steps=ddim_steps,
                class_labels=labels,
                cfg_scale=cfg_scale,
                null_class=null_class,
                eta=eta,
                progress=(i == 0),
            )
        else:
            samples = diffusion.p_sample_loop(
                model, shape, device,
                class_labels=labels,
                cfg_scale=cfg_scale,
                null_class=null_class,
                progress=(i == 0),
            )

        all_samples.append(samples.cpu())
        n_done = sum(len(x) for x in all_samples)
        if len(batches) > 1:
            print(f"  Generated {n_done}/{n_samples}", end='\r')

    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# Eta sweep: visualize effect of stochasticity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eta_sweep(model, diffusion, cfg, device, output_dir):
    """Generate grids for eta in [0.0, 0.25, 0.5, 0.75, 1.0]."""
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    os.makedirs(output_dir, exist_ok=True)
    m = cfg.model
    num_classes = getattr(cfg.data, 'num_classes', 0)
    null_class = num_classes if num_classes > 0 else None

    labels = None
    if num_classes > 0:
        labels = torch.arange(16, device=device) % num_classes

    # Fix noise seed for fair comparison
    torch.manual_seed(42)
    x_noise = torch.randn(16, m.in_channels, m.image_size, m.image_size, device=device)

    grids = []
    for eta in etas:
        # Reset to same starting noise
        torch.manual_seed(42)
        samples = diffusion.ddim_sample_loop(
            model, x_noise.shape, device,
            ddim_steps=50, class_labels=labels,
            cfg_scale=cfg.sampling.cfg_scale,
            null_class=null_class,
            eta=eta, progress=False,
        )
        grids.append(samples)
        path = os.path.join(output_dir, f'eta_{eta:.2f}.png')
        save_sample_grid(samples, path)
        print(f"  eta={eta:.2f} → {path}")

    print("Eta sweep complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DDPM/DDIM Sampling')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--output_dir', default='outputs/generated', help='Output directory')
    parser.add_argument('--sampler', choices=['ddim', 'ddpm'], default='ddim')
    parser.add_argument('--steps', type=int, default=50, help='DDIM steps (ignored for DDPM)')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM stochasticity (0=det, 1=DDPM)')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='CFG guidance scale')
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--class_label', type=int, default=None, help='Specific class to generate')
    parser.add_argument('--fid_mode', action='store_true', help='Save individual files for FID')
    parser.add_argument('--eta_sweep', action='store_true', help='Generate eta sweep visualization')
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use EMA weights')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load config
    cfg_dict = load_config(args.config)
    cfg = dict_to_ns(cfg_dict)
    num_classes = getattr(cfg.data, 'num_classes', 0)
    null_class = num_classes if num_classes > 0 else None

    # Build model
    model = build_model_from_cfg(cfg, num_classes).to(device)
    ema = EMA(model, decay=cfg.training.ema_decay)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, ema=ema, device=str(device))
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")

    # Apply EMA weights for sampling
    if args.use_ema:
        print("Using EMA weights for sampling.")
        ema.apply_shadow(model)

    model.eval()

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion.timesteps,
        schedule=cfg.diffusion.schedule,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Eta sweep
    if args.eta_sweep:
        eta_sweep(model, diffusion, cfg, device, args.output_dir)
        return

    # Regular sampling
    print(f"\nSampling {args.n_samples} images with {args.sampler.upper()} "
          f"(steps={args.steps if args.sampler=='ddim' else cfg.diffusion.timesteps}, "
          f"eta={args.eta}, cfg={args.cfg_scale})")

    samples = run_sampling(
        model, diffusion, device,
        image_size=cfg.model.image_size,
        in_channels=cfg.model.in_channels,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        sampler=args.sampler,
        ddim_steps=args.steps,
        eta=args.eta,
        cfg_scale=args.cfg_scale,
        class_label=args.class_label,
        num_classes=num_classes,
        null_class=null_class,
    )

    if args.fid_mode:
        save_samples_for_fid(samples, args.output_dir)
        print(f"\nSaved {len(samples)} individual images to: {args.output_dir}")
    else:
        grid_path = os.path.join(args.output_dir, 'generated_grid.png')
        save_sample_grid(samples, grid_path, nrow=min(8, args.n_samples))
        print(f"\nSaved sample grid: {grid_path}")

    if args.use_ema:
        ema.restore(model)


if __name__ == '__main__':
    main()
