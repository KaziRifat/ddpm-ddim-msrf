"""
UNet Architecture for DDPM/DDIM
Architectural Twist: Multi-Scale Residual Fusion (MSRF)
  - Each decoder block fuses features from ALL encoder scales (not just the skip connection)
  - Learned gating weights determine contribution of each scale
  - Improves detail preservation at fine-grained and coarse-grained levels simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize the parameters of a module (used for final conv layers)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---------------------------------------------------------------------------
# Time / Sinusoidal Embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Classic sinusoidal timestep embeddings from 'Attention Is All You Need'.
    Maps scalar timestep t -> R^dim vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """Projects sinusoidal embedding into model dimension."""
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ---------------------------------------------------------------------------
# Normalization & Attention
# ---------------------------------------------------------------------------

class GroupNorm(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        # Gracefully handle cases where channels < num_groups
        groups = min(num_groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x):
        return self.norm(x)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention for spatial feature maps.
    Applied at specified resolutions to capture long-range dependencies.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(channels // num_heads, 1)
        self.scale = self.head_dim ** -0.5

        self.norm = GroupNorm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)

        qkv = self.qkv(h)                                   # B, 3C, HW
        q, k, v = qkv.chunk(3, dim=1)                        # B, C, HW each

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)

        # Scaled dot-product attention
        attn = torch.einsum('bncd,bnce->bnde', q, k) * self.scale  # B, heads, HW, HW
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bnde,bnce->bncd', attn, v)      # B, heads, head_dim, HW
        out = out.reshape(B, C, H * W)
        out = self.proj(out)

        return (x + out.view(B, C, H, W))


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    ResNet-style residual block with time embedding injection and optional class conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 32,
        dropout: float = 0.1,
        class_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = GroupNorm(in_channels, num_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = GroupNorm(out_channels, num_groups)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.time_proj = nn.Sequential(Swish(), nn.Linear(time_dim, out_channels * 2))

        if class_dim is not None:
            self.class_proj = nn.Sequential(Swish(), nn.Linear(class_dim, out_channels))
        else:
            self.class_proj = None

        self.dropout = nn.Dropout(dropout)
        self.act = Swish()

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        c_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Time conditioning via scale + shift (AdaGN style)
        t = self.time_proj(t_emb)[:, :, None, None]
        scale, shift = t.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Class conditioning (additive)
        if self.class_proj is not None and c_emb is not None:
            h = h + self.class_proj(c_emb)[:, :, None, None]

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# ★ Architectural Twist: Multi-Scale Residual Fusion (MSRF) Gate
# ---------------------------------------------------------------------------

class MSRFGate(nn.Module):
    """
    Multi-Scale Residual Fusion Gate.

    In a standard UNet decoder, each block receives one skip connection from
    the corresponding encoder level. MSRF instead collects skip connections
    from ALL encoder levels, resizes them to the current resolution, and
    combines them via learned soft gates (sigmoid-weighted sum).

    This gives each decoder stage a global view of the encoder hierarchy,
    allowing it to draw on both fine local texture (shallow skips) and
    semantic context (deep skips) simultaneously.

    Gate weights are conditioned on the timestep embedding so the model can
    learn to rely more on semantic features at high noise levels and more on
    fine features at low noise levels.
    """
    def __init__(self, num_scales: int, channels: int, time_dim: int):
        super().__init__()
        self.num_scales = num_scales
        # One gate weight per encoder scale, conditioned on time
        self.gate_net = nn.Sequential(
            Swish(),
            nn.Linear(time_dim, num_scales),
        )
        # 1x1 conv to project all concatenated scales back to `channels`
        self.proj = nn.Conv2d(channels * num_scales, channels, kernel_size=1)
        self.norm = GroupNorm(channels)

    def forward(
        self,
        skips: List[torch.Tensor],   # list of skip tensors from all encoder levels
        t_emb: torch.Tensor,          # time embedding B x time_dim
        target_shape: Tuple[int, int],  # (H, W) of current decoder level
    ) -> torch.Tensor:
        H, W = target_shape
        # Resize all skips to current resolution
        resized = []
        for s in skips:
            if s.shape[2:] != (H, W):
                s = F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
            resized.append(s)

        # Soft gate weights: B x num_scales
        gates = torch.sigmoid(self.gate_net(t_emb))  # B x num_scales

        # Weight each scale
        weighted = [resized[i] * gates[:, i, None, None, None] for i in range(self.num_scales)]
        fused = torch.cat(weighted, dim=1)   # B x (C*num_scales) x H x W
        return self.norm(self.proj(fused))


# ---------------------------------------------------------------------------
# Encoder & Decoder Blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, time_dim: int,
        num_res: int = 2, use_attn: bool = False,
        dropout: float = 0.1, class_dim: Optional[int] = None,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResidualBlock(
                in_ch if i == 0 else out_ch, out_ch,
                time_dim, dropout=dropout, class_dim=class_dim
            ) for i in range(num_res)
        ])
        self.attns = nn.ModuleList([
            SelfAttention(out_ch) if use_attn else nn.Identity()
            for _ in range(num_res)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb, c_emb=None):
        for res, attn in zip(self.resnets, self.attns):
            x = res(x, t_emb, c_emb)
            x = attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int,
        num_res: int = 2, use_attn: bool = False,
        dropout: float = 0.1, class_dim: Optional[int] = None,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.resnets = nn.ModuleList([
            ResidualBlock(
                (in_ch + skip_ch) if i == 0 else out_ch, out_ch,
                time_dim, dropout=dropout, class_dim=class_dim
            ) for i in range(num_res)
        ])
        self.attns = nn.ModuleList([
            SelfAttention(out_ch) if use_attn else nn.Identity()
            for _ in range(num_res)
        ])

    def forward(self, x, skip, t_emb, c_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res, attn in zip(self.resnets, self.attns):
            x = res(x, t_emb, c_emb)
            x = attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.1, class_dim: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_dim, dropout=dropout, class_dim=class_dim)
        self.attn = SelfAttention(channels)
        self.res2 = ResidualBlock(channels, channels, time_dim, dropout=dropout, class_dim=class_dim)

    def forward(self, x, t_emb, c_emb=None):
        x = self.res1(x, t_emb, c_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb, c_emb)
        return x


# ---------------------------------------------------------------------------
# Full UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Noise-prediction UNet for DDPM/DDIM with:
      - Sinusoidal time embeddings
      - Multi-head self-attention at configurable resolutions
      - AdaGN time conditioning (scale + shift)
      - Optional classifier-free guidance class conditioning
      - ★ MSRF: Multi-Scale Residual Fusion in the decoder
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        num_classes: Optional[int] = None,   # None = unconditional
        use_msrf: bool = True,               # Toggle MSRF twist
    ):
        super().__init__()
        self.use_msrf = use_msrf
        self.num_levels = len(channel_mults)

        time_dim = base_channels * 4
        self.time_embed = TimeEmbedding(base_channels, time_dim)

        # Class embedding for classifier-free guidance
        class_dim = None
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes + 1, time_dim)  # +1 for null class
            class_dim = time_dim
        else:
            self.class_embed = None

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---- Encoder ----
        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        current_res = image_size

        for i, out_ch in enumerate(channels):
            use_attn = current_res in attn_resolutions
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_dim, num_res_blocks, use_attn, dropout, class_dim)
            )
            in_ch = out_ch
            current_res //= 2

        # ---- Bottleneck ----
        self.middle = MiddleBlock(channels[-1], time_dim, dropout, class_dim)

        # ---- Decoder (with MSRF) ----
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))

        if use_msrf:
            # One MSRF gate per decoder level
            self.msrf_gates = nn.ModuleList()

        for i in range(self.num_levels):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[i]           # from MSRF or standard skip
            out_ch = rev_channels[i + 1] if i + 1 < self.num_levels else base_channels
            use_attn = (current_res * 2) in attn_resolutions

            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, time_dim, num_res_blocks, use_attn, dropout, class_dim)
            )
            if use_msrf:
                self.msrf_gates.append(
                    MSRFGate(
                        num_scales=self.num_levels,
                        channels=skip_ch,
                        time_dim=time_dim,
                    )
                )
            current_res *= 2

        # ---- Output head ----
        self.output_norm = GroupNorm(base_channels)
        self.output_act = Swish()
        self.output_conv = zero_module(nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:             Noisy image, shape (B, C, H, W)
            t:             Timesteps, shape (B,)
            class_labels:  Class indices (B,) or None. Pass num_classes for null class (CFG).
        Returns:
            Predicted noise, shape (B, C, H, W)
        """
        # Embeddings
        t_emb = self.time_embed(t)
        c_emb = None
        if self.class_embed is not None and class_labels is not None:
            c_emb = self.class_embed(class_labels)
            t_emb = t_emb + c_emb   # fuse into time stream

        # Encode
        x = self.input_conv(x)
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t_emb, c_emb)
            skips.append(skip)

        # Bottleneck
        x = self.middle(x, t_emb, c_emb)

        # Decode with MSRF or standard skip
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]

            if self.use_msrf:
                # Fuse ALL encoder skips into this decoder level
                skip = self.msrf_gates[i](skips, t_emb, skip.shape[2:])

            x = up(x, skip, t_emb, c_emb)

        # Output
        x = self.output_act(self.output_norm(x))
        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    model = UNet(
        image_size=64,
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 3, 4),
        num_res_blocks=2,
        attn_resolutions=(16, 8),
        num_classes=10,
        use_msrf=True,
    ).to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.2f}M")

    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    c = torch.randint(0, 10, (2,)).to(device)

    out = model(x, t, c)
    print(f"Output shape: {out.shape}")   # Should be (2, 3, 64, 64)
