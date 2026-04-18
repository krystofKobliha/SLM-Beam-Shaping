# ──────────────────────────────────────────────────────────────────────────────
# File: model.py               Project: DNN-SLM Phase Mask Generation
# Purpose: U-Net that maps a target intensity image to a phase mask [0, 2π]
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → GroupNorm → ReLU) blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool → DoubleConv  (halves spatial dims, doubles channels)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsample → concatenate skip → DoubleConv."""
    def __init__(self, in_ch, skip_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, skip_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd resolutions / progressive scaling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class HoloNet(nn.Module):
    """
    U-Net that maps a target intensity image (1-ch) to a phase mask [0, 2π].

    6-level encoder/decoder for large receptive field (~572 px).
    At S1 (256×320) bottleneck is 4×5; at S7 (1024×1280) it is 16×20.
    Channels capped at 1024 for the deepest 3 levels.

    Output encoding: two-channel logit → atan2 → remainder.
    This avoids the discontinuity of a simple sigmoid×2π mapping and gives
    smooth gradients across the 0/2π wrap-around.
    """
    def __init__(self, n_channels=1):
        super().__init__()
        # --- Encoder (6 downsampling levels) ---
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)       # H/2
        self.down2 = Down(128, 256)      # H/4
        self.down3 = Down(256, 512)      # H/8
        self.down4 = Down(512, 1024)     # H/16
        self.down5 = Down(1024, 1024)    # H/32
        self.down6 = Down(1024, 1024)    # H/64 (bottleneck)

        # --- Decoder (6 upsampling levels) ---
        self.up0 = Up(1024, 1024)  # → H/32  (skip from down5)
        self.up1 = Up(1024, 1024)  # → H/16  (skip from down4)
        self.up2 = Up(1024, 512)   # → H/8   (skip from down3)
        self.up3 = Up(512, 256)    # → H/4   (skip from down2)
        self.up4 = Up(256, 128)    # → H/2   (skip from down1)
        self.up5 = Up(128, 64)     # → H     (skip from inc)

        # 2-channel output: (sin-like, cos-like) → atan2
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # 64-ch,   H
        x2 = self.down1(x1)    # 128-ch,  H/2
        x3 = self.down2(x2)    # 256-ch,  H/4
        x4 = self.down3(x3)    # 512-ch,  H/8
        x5 = self.down4(x4)    # 1024-ch, H/16
        x6 = self.down5(x5)    # 1024-ch, H/32
        x7 = self.down6(x6)    # 1024-ch, H/64 (bottleneck)

        # Decoder
        x = self.up0(x7, x6)   # 1024-ch, H/32
        x = self.up1(x, x5)    # 1024-ch, H/16
        x = self.up2(x, x4)    # 512-ch,  H/8
        x = self.up3(x, x3)    # 256-ch,  H/4
        x = self.up4(x, x2)    # 128-ch,  H/2
        x = self.up5(x, x1)    # 64-ch,   H

        logits = self.outc(x)

        # Normalize logits to unit circle before atan2 — keeps gradient
        # magnitude O(1) regardless of logit scale, preventing the
        # vanishing-gradient phase collapse observed at higher resolutions.
        logits = F.normalize(logits, dim=1, eps=1e-6)
        phase = torch.atan2(logits[:, 0, :, :], logits[:, 1, :, :])

        # Shift from [-π, π] to [0, 2π], add channel dim → [B, 1, H, W]
        phase = torch.remainder(phase, 2 * math.pi)
        return phase.unsqueeze(1)
