"""
Lightweight CNN for segment-level edge detection (Stage 1).
Replace/extend as needed. Designed for log-mel inputs of shape (B, 1, MELS, T).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn(self.conv(x))))
        return x


class Stage1CNNSegment(nn.Module):
    def __init__(self, n_classes: int = 4, in_ch: int = 1):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 16, pool=(2, 2))
        self.block2 = ConvBlock(16, 32, pool=(2, 2))
        self.block3 = ConvBlock(32, 64, pool=(2, 2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


def create_model(n_classes: int, in_ch: int = 1) -> nn.Module:
    return Stage1CNNSegment(n_classes=n_classes, in_ch=in_ch)
