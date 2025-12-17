"""
Minimal training script for Stage-1 CNN on precomputed log-mel NPZ data.
This is a scaffold; plug in your preprocessing pipeline and data paths.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.baseline_cnn import create_model
from src.train.dataset import LogMelDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=pathlib.Path, required=True, help="Folder with *.npz (feat, label)")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-path", type=pathlib.Path, default="checkpoints/stage1_cnn.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = LogMelDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = create_model(n_classes=args.num_classes).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for feats, labels in dl:
            feats = feats.to(args.device)
            labels = labels.to(args.device)
            opt.zero_grad()
            logits = model(feats)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(
            f"epoch {epoch}: loss={total_loss/total:.4f}, "
            f"acc={total_correct/total:.4f}"
        )

    torch.save(model.state_dict(), args.save_path)
    print(f"saved model -> {args.save_path}")


if __name__ == "__main__":
    main()
