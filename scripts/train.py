#!/usr/bin/env python3
"""Train DeepLabV3-ResNet50 on laparoscopic cholecystectomy landmarks."""

from __future__ import annotations

import argparse
import os
import sys
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

from cystic_seg.constants import IMAGE_SIZE
from cystic_seg.dataset import SegmentationDataset, load_data_with_masks
from cystic_seg.evaluate import evaluate_loader, format_metrics, write_metrics
from cystic_seg.losses import BileDuctLoss
from cystic_seg.models import build_deeplabv3, get_device, load_checkpoint, save_checkpoint
from cystic_seg.splits import load_split, make_split_info, samples_for_subset, save_split
from cystic_seg.tasks import duct_mean_dice, get_task, wrap_for_task
from cystic_seg.viz import save_prediction_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cystic landmark segmentation model")
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--experiment-dir", type=str, default="experiments/train_test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save/load best weights")
    parser.add_argument("--split-json", type=str, default=None, help="Reuse a split from scripts/make_splits.py")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists")
    parser.add_argument("--num-vis", type=int, default=8, help="Validation overlays to save at the end")
    parser.add_argument("--split", choices=["case", "sample"], default="case", help="Holdout unit")
    parser.add_argument("--class-weights", action="store_true", help="Inverse-frequency CrossEntropy weights")
    parser.add_argument(
        "--task",
        choices=["landmarks", "bile-ducts", "ducts-only"],
        default="landmarks",
        help="landmarks=8 classes; bile-ducts=CBD/CD/other; ducts-only=CBD/CD vs everything else as background",
    )
    parser.add_argument(
        "--duct-crop-prob",
        type=float,
        default=None,
        help="Train-time crop around CBD/CD. Default 0.7 for duct tasks, 0 otherwise.",
    )
    parser.add_argument(
        "--ignore-other",
        action="store_true",
        help="For bile-ducts, ignore the dump class in CE so GB/liver/instrument pixels do not drive the loss.",
    )
    return parser.parse_args()


def class_counts(samples, num_classes: int, remap: np.ndarray | None = None) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for sample in samples:
        mask = np.asarray(sample["mask"]).reshape(-1)
        if remap is not None:
            mask = remap[mask.astype(np.int64)]
        counts += torch.bincount(torch.as_tensor(mask).long(), minlength=num_classes).to(torch.float64)
    return counts


def inverse_frequency_weights(counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    weights = 1.0 / (counts + 1.0)
    return (weights / weights.mean()).float().to(device)


def main() -> None:
    args = parse_args()
    device = get_device()
    os.makedirs(args.experiment_dir, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(args.experiment_dir, "deeplabv3_best.pth")
    log_path = os.path.join(args.experiment_dir, "train_val_log.txt")
    image_size = tuple(args.image_size)
    spec = get_task(args.task)
    num_classes = spec.num_classes
    class_names = spec.class_names
    duct_crop_prob = args.duct_crop_prob
    if duct_crop_prob is None:
        duct_crop_prob = 0.7 if spec.duct_focus else 0.0
    monitor = "duct-dice" if spec.duct_focus else "val-loss"

    samples = load_data_with_masks(args.data_root)
    if not samples:
        raise SystemExit(f"No annotated samples found under {args.data_root}")

    split_path = args.split_json or os.path.join(args.experiment_dir, "split.json")
    if args.split_json and Path(args.split_json).exists():
        split_info = load_split(args.split_json)
        print(f"Loaded split from {args.split_json}")
    else:
        split_info = make_split_info(
            samples,
            unit=args.split,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        save_split(split_info, split_path)
        print(f"Wrote new split to {split_path}")

    train_samples = samples_for_subset(samples, split_info, "train")
    val_samples = samples_for_subset(samples, split_info, "val")
    test_samples = (
        samples_for_subset(samples, split_info, "test")
        if split_info.get("test_images") or split_info.get("test_cases")
        else []
    )
    print(
        f"Task {args.task} ({num_classes} classes) | "
        f"Train {len(train_samples)} frames / {len(split_info.get('train_cases', []))} cases | "
        f"Val {len(val_samples)} / {len(split_info.get('val_cases', []))} | "
        f"Test {len(test_samples)} / {len(split_info.get('test_cases', []))} (held out)"
    )
    if not train_samples or not val_samples:
        raise SystemExit("Train or val split is empty.")
    split_copy = Path(args.experiment_dir) / "split.json"
    if Path(split_path).resolve() != split_copy.resolve():
        save_split(split_info, split_copy)

    train_dataset = wrap_for_task(
        SegmentationDataset.from_samples(
            train_samples, augment=True, image_size=image_size, duct_crop_prob=duct_crop_prob
        ),
        args.task,
    )
    val_dataset = wrap_for_task(
        SegmentationDataset.from_samples(val_samples, augment=False, image_size=image_size),
        args.task,
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
    weights = None
    if args.class_weights:
        counts = class_counts(train_samples, num_classes, remap=spec.remap)
        weights = inverse_frequency_weights(counts, device)
        if spec.duct_focus:
            weights[spec.cbd_id] *= 1.5
            weights[spec.cd_id] *= 1.5
            if num_classes > 3:
                weights[3] *= 0.5
            weights = weights / weights.mean()
        print("Class weights:", {class_names[i]: float(weights[i]) for i in range(num_classes)})

    if spec.duct_focus:
        criterion = BileDuctLoss(
            class_weights=weights,
            cbd_id=spec.cbd_id,
            cd_id=spec.cd_id,
            ignore_index=3 if args.ignore_other and args.task == "bile-ducts" else -100,
        )
        if args.ignore_other and args.task == "bile-ducts":
            print("Ignoring 'other' pixels in cross-entropy.")
    elif weights is not None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched_mode = "max" if monitor == "duct-dice" else "min"
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, patience=5, factor=0.5, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, patience=5, factor=0.5
        )

    best_loss = float("inf")
    best_duct_dice = -1.0
    start_epoch = 0
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        extra = load_checkpoint(model, checkpoint_path, device)
        if extra.get("optimizer"):
            optimizer.load_state_dict(extra["optimizer"])
        best_loss = float(extra.get("best_loss", best_loss))
        best_duct_dice = float(extra.get("best_duct_dice", best_duct_dice))
        start_epoch = int(extra.get("epoch", 0))

    train_losses = []
    val_losses = []
    duct_dices = []
    lrs = []
    epochs_no_improve = 0

    with open(log_path, "w") as log_file:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} train"):
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)["out"]
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            val_metrics = evaluate_loader(
                model, val_loader, device, num_classes=num_classes, class_names=class_names
            )
            epoch_duct_dice = duct_mean_dice(val_metrics, cbd_id=spec.cbd_id, cd_id=spec.cd_id)
            if epoch_duct_dice is None:
                epoch_duct_dice = 0.0
            duct_dices.append(epoch_duct_dice)

            print(
                f"Epoch {epoch + 1}/{args.epochs}, "
                f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val CBD+CD Dice: {epoch_duct_dice:.4f}, LR: {current_lr:.6f}"
            )
            log_file.write(
                f"Epoch {epoch + 1}/{args.epochs}, "
                f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val CBD+CD Dice: {epoch_duct_dice:.4f}, LR: {current_lr:.6f}\n"
            )
            log_file.flush()
            scheduler.step(epoch_duct_dice if monitor == "duct-dice" else val_loss)

            improved = (
                epoch_duct_dice > best_duct_dice + 1e-6
                if monitor == "duct-dice"
                else val_loss < best_loss
            )
            if improved:
                best_loss = min(best_loss, val_loss)
                best_duct_dice = max(best_duct_dice, epoch_duct_dice)
                epochs_no_improve = 0
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_loss=best_loss,
                    best_duct_dice=best_duct_dice,
                    task=args.task,
                    num_classes=num_classes,
                )
                print("Model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no {monitor} improvement for {args.patience} epochs)."
                    )
                    break

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(args.experiment_dir, "loss_curve.png"))
    plt.close()

    if duct_dices:
        plt.figure()
        plt.plot(range(1, len(duct_dices) + 1), duct_dices, label="Val CBD+CD Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()
        plt.title("Validation CBD + cystic duct Dice")
        plt.savefig(os.path.join(args.experiment_dir, "duct_dice_curve.png"))
        plt.close()

    plt.figure()
    plt.plot(range(1, len(lrs) + 1), lrs, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.savefig(os.path.join(args.experiment_dir, "lr_curve.png"))
    plt.close()

    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    n_save = args.num_vis
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            imgs = images.cpu().numpy()
            gts = masks.numpy()
            for j in range(min(n_save, images.shape[0])):
                save_prediction_triplet(
                    imgs[j],
                    gts[j],
                    preds[j],
                    os.path.join(args.experiment_dir, f"val_result_{i}_{j}.png"),
                    num_classes=num_classes,
                )
            n_save -= images.shape[0]
            if n_save <= 0:
                break

    metrics = evaluate_loader(model, val_loader, device, num_classes=num_classes, class_names=class_names)
    metrics["subset"] = "val"
    metrics["task"] = args.task
    metrics["split"] = {k: v for k, v in split_info.items() if not k.endswith("_images")}
    write_metrics(metrics, os.path.join(args.experiment_dir, "val_metrics.json"))
    print("=== validation (used for early stopping, not the test set) ===")
    print(format_metrics(metrics))
    print("Run scripts/evaluate_split.py --subset test after training for held-out metrics.")


if __name__ == "__main__":
    main()
