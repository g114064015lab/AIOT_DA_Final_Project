"""
Simple dataset wrapper for log-mel features and labels.
Expects precomputed NPZ files with keys: 'feat' (C x MELS x T) and 'label' (int).
Adapt as needed to match your preprocessing pipeline.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LogMelDataset(Dataset):
    def __init__(self, root: str | pathlib.Path, transform: Callable | None = None):
        self.root = pathlib.Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        self.transform = transform
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        feat = data["feat"]  # shape: C x MELS x T
        label = data["label"]  # scalar class id
        if self.transform:
            feat = self.transform(feat)
        feat_t = torch.tensor(feat, dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.long)
        return feat_t, label_t
