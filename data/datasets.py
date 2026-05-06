"""
Dataset utilities for DDPM/DDIM training.

Supports:
  - CIFAR-10 / CIFAR-100
  - CelebA (faces)
  - ImageFolder (custom datasets)
  - Tiny ImageNet
  - Custom paths via config

All datasets return images normalized to [-1, 1] as required by diffusion models.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from typing import Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# For diffusion: map [0,1] → [-1,1]
DIFFUSION_MEAN = (0.5, 0.5, 0.5)
DIFFUSION_STD  = (0.5, 0.5, 0.5)


def get_diffusion_transform(
    image_size: int,
    augment: bool = True,
    grayscale: bool = False,
) -> transforms.Compose:
    """
    Standard transform pipeline for diffusion model training.
    Output: float32 tensor in [-1, 1].
    """
    t_list = []

    if image_size is not None:
        t_list.append(transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC))
        t_list.append(transforms.CenterCrop(image_size))

    if augment:
        t_list.append(transforms.RandomHorizontalFlip())

    if grayscale:
        t_list.append(transforms.Grayscale(num_output_channels=3))

    t_list += [
        transforms.ToTensor(),                               # [0, 1]
        transforms.Normalize(DIFFUSION_MEAN, DIFFUSION_STD), # [-1, 1]
    ]

    return transforms.Compose(t_list)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] tensor back to [0,1] for visualization/saving."""
    return (x.clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# Built-in Dataset Loaders
# ---------------------------------------------------------------------------

def get_cifar10(
    root: str,
    image_size: int = 32,
    augment: bool = True,
) -> Tuple[Dataset, Dataset, int]:
    """Returns (train_dataset, val_dataset, num_classes)."""
    tfm = get_diffusion_transform(image_size, augment)
    train = datasets.CIFAR10(root=root, train=True,  transform=tfm, download=True)
    val   = datasets.CIFAR10(root=root, train=False, transform=tfm, download=True)
    return train, val, 10


def get_cifar100(
    root: str,
    image_size: int = 32,
    augment: bool = True,
) -> Tuple[Dataset, Dataset, int]:
    tfm = get_diffusion_transform(image_size, augment)
    train = datasets.CIFAR100(root=root, train=True,  transform=tfm, download=True)
    val   = datasets.CIFAR100(root=root, train=False, transform=tfm, download=True)
    return train, val, 100


def get_celeba(
    root: str,
    image_size: int = 64,
    augment: bool = True,
) -> Tuple[Dataset, Dataset, int]:
    """CelebA - requires manual download (torchvision handles it if google drive works)."""
    tfm = get_diffusion_transform(image_size, augment)
    train = datasets.CelebA(root=root, split='train', transform=tfm, download=True)
    val   = datasets.CelebA(root=root, split='valid', transform=tfm, download=True)
    return train, val, 0   # Unconditional for CelebA by default


def get_tiny_imagenet(
    root: str,
    image_size: int = 64,
    augment: bool = True,
) -> Tuple[Dataset, Dataset, int]:
    """
    Tiny ImageNet - expects folder structure:
      root/tiny-imagenet-200/train/<class_id>/images/*.JPEG
      root/tiny-imagenet-200/val/images/*.JPEG
    Download: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    tfm = get_diffusion_transform(image_size, augment)
    train_dir = os.path.join(root, 'tiny-imagenet-200', 'train')
    val_dir   = os.path.join(root, 'tiny-imagenet-200', 'val')
    train = datasets.ImageFolder(train_dir, transform=tfm)
    val   = datasets.ImageFolder(val_dir,   transform=tfm)
    return train, val, 200


# ---------------------------------------------------------------------------
# Custom ImageFolder Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """
    Generic image folder dataset.
    Expected structure:
      root/
        class_a/img1.jpg img2.jpg ...
        class_b/img3.jpg ...
    Or flat structure (unconditional):
      root/img1.jpg img2.jpg ...
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    def __init__(
        self,
        root: str,
        image_size: int = 64,
        augment: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform or get_diffusion_transform(image_size, augment)
        self.samples, self.class_to_idx = self._scan(root)

    def _scan(self, root):
        samples = []
        class_to_idx = {}

        # Check if root contains subdirectories (class structure) or flat images
        entries = os.listdir(root)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]

        if subdirs:
            # Class-structured
            for idx, cls in enumerate(sorted(subdirs)):
                class_to_idx[cls] = idx
                cls_dir = os.path.join(root, cls)
                for fname in sorted(os.listdir(cls_dir)):
                    if os.path.splitext(fname)[1].lower() in self.EXTENSIONS:
                        samples.append((os.path.join(cls_dir, fname), idx))
        else:
            # Flat (unconditional)
            for fname in sorted(entries):
                if os.path.splitext(fname)[1].lower() in self.EXTENSIONS:
                    samples.append((os.path.join(root, fname), -1))

        return samples, class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

    @property
    def num_classes(self):
        return len(self.class_to_idx) if self.class_to_idx else 0


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'cifar10':        get_cifar10,
    'cifar100':       get_cifar100,
    'celeba':         get_celeba,
    'tiny_imagenet':  get_tiny_imagenet,
}


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
    val_split: float = 0.05,  # used only for custom datasets
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, num_classes).
    num_classes=0 means unconditional training.
    """
    if dataset_name in DATASET_REGISTRY:
        train_ds, val_ds, num_classes = DATASET_REGISTRY[dataset_name](
            data_root, image_size, augment
        )
    else:
        # Treat as path to custom image folder
        full_ds = ImageFolderDataset(data_root, image_size, augment)
        n_val = max(1, int(len(full_ds) * val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        num_classes = full_ds.num_classes

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train_loader, val_loader, nc = build_dataloaders(
        'cifar10', './data', image_size=32, batch_size=8, num_workers=0
    )
    imgs, labels = next(iter(train_loader))
    print(f"Batch: {imgs.shape}, labels: {labels}, num_classes: {nc}")
    print(f"Pixel range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print("Data loading OK ✓")
