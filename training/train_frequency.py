"""Training script for the Frequency Forensics Stream.

Trains a modified ResNet-18 on concatenated FFT/DCT spectral maps
for detecting compression artifacts and GAN-based upscaling.

Usage:
    python -m training.train_frequency --data_dir data/processed --epochs 50
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path

from streams.frequency.model import FrequencyForensicsStream
from streams.frequency.preprocess import prepare_freq_tensor


class FrequencyDataset(Dataset):
    """Dataset that converts images to FFT/DCT spectral maps."""

    def __init__(self, data_dir: str, split: str = "train", size: int = 224):
        self.data_dir = Path(data_dir)
        self.size = size
        self.samples = []

        for label_name, label_val in [("genuine", 0), ("synthetic", 1)]:
            label_dir = self.data_dir / split / label_name
            if label_dir.exists():
                for img_path in sorted(label_dir.glob("*")):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((str(img_path), label_val))

        print(f"[{split}] Loaded {len(self.samples)} frequency samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        freq_tensor = prepare_freq_tensor(image, size=self.size)
        return freq_tensor, torch.tensor(label, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(tensors)
        scores = output["frequency_score"]
        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * tensors.size(0)
        preds = (scores > 0.5).float()
        correct += (preds == labels).sum().item()
        total += tensors.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for tensors, labels in loader:
            tensors, labels = tensors.to(device), labels.to(device)
            output = model(tensors)
            scores = output["frequency_score"]
            loss = criterion(scores, labels)

            total_loss += loss.item() * tensors.size(0)
            preds = (scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += tensors.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Frequency Forensics Stream")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_dataset = FrequencyDataset(args.data_dir, split="train")
    val_dataset = FrequencyDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = FrequencyForensicsStream(pretrained=False).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }, "checkpoints/frequency_best.pth")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
