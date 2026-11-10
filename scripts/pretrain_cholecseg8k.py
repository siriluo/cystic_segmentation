#!/usr/bin/env python3
"""Pretrain DeepLabV3-ResNet50 on CholecSeg8k laparoscopic dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from cystic_seg.cholecseg8k import (
    CHOLECSEG8K_CLASSES,
    CholecSeg8kDataset,
    discover_cholecseg8k_samples,
)
from cystic_seg.constants import IMAGE_SIZE
from cystic_seg.models import build_deeplabv3, get_device, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain ResNet50 / DeepLabV3 on CholecSeg8k")
    parser.add_argument("--data-dir", type=str, default="data/CholecSeg8k", help="Path to CholecSeg8k root")
    parser.add_argument("--experiment-dir", type=str, default="experiments/cholecseg8k_pretrain")
    parser.add_argument("--target-space", choices=["cholecseg", "cystic"], default="cholecseg")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    return parser.parse_args()


def split_samples_by_video(
    samples: list[dict[str, str]], val_ratio: float = 0.15, seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Group frames by video/clip to avoid leakage across splits."""
    by_video = defaultdict(list)
    for s in samples:
        by_video[s["video_id"]].append(s)

    video_ids = sorted(list(by_video.keys()))
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n_val = max(1, int(round(len(video_ids) * val_ratio)))
    val_videos = set(video_ids[:n_val])
    train_videos = set(video_ids[n_val:])

    train_samples = [s for s in samples if s["video_id"] in train_videos]
    val_samples = [s for s in samples if s["video_id"] in val_videos]
    print(f"Video-level split: {len(train_videos)} train videos ({len(train_samples)} frames), "
          f"{len(val_videos)} val videos ({len(val_samples)} frames)")
    return train_samples, val_samples


def main() -> None:
    args = parse_args()
    device = get_device()
    os.makedirs(args.experiment_dir, exist_ok=True)
    image_size = tuple(args.image_size)

    num_classes = 8 if args.target_space == "cystic" else 13
    print(f"Using device: {device} | Target space: {args.target_space} ({num_classes} classes)")

    samples = discover_cholecseg8k_samples(args.data_dir)
    if not samples:
        raise SystemExit(f"No CholecSeg8k samples found under {args.data_dir}")

    train_samples, val_samples = split_samples_by_video(samples, val_ratio=args.val_ratio, seed=args.seed)

    # Save split info
    split_info = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "seed": args.seed,
    }
    with open(os.path.join(args.experiment_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    train_dataset = CholecSeg8kDataset(
        train_samples, image_size=image_size, augment=True, target_space=args.target_space
    )
    val_dataset = CholecSeg8kDataset(
        val_samples, image_size=image_size, augment=False, target_space=args.target_space
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_deeplabv3(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    checkpoint_path = os.path.join(args.experiment_dir, "cholecseg8k_deeplabv3_best.pth")
    backbone_path = os.path.join(args.experiment_dir, "cholecseg8k_resnet50_backbone_best.pth")
    log_path = os.path.join(args.experiment_dir, "pretrain_log.txt")

    best_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    with open(log_path, "w") as log_file:
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]"):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)["out"]
                    loss = criterion(outputs, masks)
                    running_val_loss += loss.item() * images.size(0)

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            msg = (
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                epochs_no_improve = 0
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_loss=best_loss,
                    num_classes=num_classes,
                    target_space=args.target_space,
                )
                # Also save the ResNet50 backbone state_dict cleanly
                torch.save(
                    {"backbone": model.backbone.state_dict(), "epoch": epoch + 1, "best_loss": best_loss},
                    backbone_path,
                )
                print(f" Saved new best model checkpoint to {checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping after {args.patience} epochs with no improvement.")
                    break

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.title("CholecSeg8k Pretraining Loss")
    plt.savefig(os.path.join(args.experiment_dir, "pretrain_loss_curve.png"))
    plt.close()
    print("Pretraining completed successfully.")


if __name__ == "__main__":
    main()
