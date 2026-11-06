"""U-Net for binary segmentation, written from scratch.

Shape walkthrough for the default config (base_channels=32, input 512x512):

    input        1 x 512 x 512
    enc1        32 x 512 x 512  ───────────────────────────┐ skip
    enc2        64 x 256 x 256  ──────────────────┐ skip   │
    enc3       128 x 128 x 128  ─────────┐ skip   │        │
    enc4       256 x  64 x  64  ┐ skip   │        │        │
    bottleneck 512 x  32 x  32  │        │        │        │
    dec4       256 x  64 x  64 ◄┘        │        │        │
    dec3       128 x 128 x 128 ◄─────────┘        │        │
    dec2        64 x 256 x 256 ◄──────────────────┘        │
    dec1        32 x 512 x 512 ◄───────────────────────────┘
    head         1 x 512 x 512  (logits)

The skip connections hand fine spatial detail from the encoder straight to the
decoder. Pooling throws away "where" in exchange for "what", so without them
the predicted bubble outlines would be blurry.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_LEVELS = 4


class DoubleConv(nn.Module):
    """Two rounds of Conv3x3 -> BatchNorm -> ReLU. Keeps height and width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # bias=False because the BatchNorm right after has its own learned shift.
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Double the resolution, concatenate the matching skip, then convolve."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat the tensor holds out_channels (up) + out_channels (skip).
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Four-level U-Net that outputs raw logits.

    Sigmoid is applied at inference time, not here, because BCEWithLogitsLoss
    combines the sigmoid and the log in one numerically stable step.
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.dec4 = UpBlock(c * 16, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2)
        self.dec1 = UpBlock(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2**NUM_LEVELS
        if x.shape[-1] % factor or x.shape[-2] % factor:
            raise ValueError(
                f"Input height and width must be divisible by {factor}, got {tuple(x.shape[-2:])}"
            )
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        return self.head(d)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
