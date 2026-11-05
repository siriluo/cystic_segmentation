#!/usr/bin/env python3
"""Evaluate a checkpoint on train, val, or the held-out test split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from cystic_seg.constants import IMAGE_SIZE
from cystic_seg.dataset import SegmentationDataset, load_data_with_masks
from cystic_seg.evaluate import evaluate_loader, format_metrics, write_metrics
from cystic_seg.metrics import compute_dice, compute_iou
from cystic_seg.models import build_deeplabv3, get_device, load_checkpoint
from cystic_seg.splits import load_split, samples_for_subset
from cystic_seg.tasks import task_class_names, task_num_classes, wrap_for_task
from cystic_seg.viz import save_prediction_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect IoU/Dice on a named split")
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subset", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", type=str, default=None, help="Metrics JSON path")
    parser.add_argument("--csv", type=str, default=None, help="Per-image metrics CSV")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    parser.add_argument("--num-vis", type=int, default=8)
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument(
        "--task",
        choices=["landmarks", "bile-ducts", "ducts-only"],
        default="landmarks",
        help="Must match the task used to train the checkpoint.",
    )
    return parser.parse_args()


def per_image_rows(model, samples, image_size, device, num_classes: int, task: str, class_names: dict) -> list[dict]:
    dataset = wrap_for_task(
        SegmentationDataset.from_samples(samples, augment=False, image_size=image_size),
        task,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, mask = dataset[idx]
            output = model(image.unsqueeze(0).to(device))["out"]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu()
            ious = compute_iou(pred, mask, num_classes)
            dices = compute_dice(pred, mask, num_classes)
            sample = samples[idx]
            row = {
                "case_id": sample.get("case_id", ""),
                "image_path": sample["image_path"],
                "mean_iou_fg": float(np.nanmean(ious[1:])) if len(ious) > 1 else float("nan"),
                "mean_dice_fg": float(np.nanmean(dices[1:])) if len(dices) > 1 else float("nan"),
            }
            for cls_id, name in class_names.items():
                key = name.replace(" ", "_").replace("'", "")
                row[f"iou_{key}"] = ious[cls_id]
                row[f"dice_{key}"] = dices[cls_id]
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    info = load_split(args.split_json)
    samples = load_data_with_masks(args.data_root)
    if not samples:
        raise SystemExit(f"No samples under {args.data_root}")
    subset_samples = samples_for_subset(samples, info, args.subset)
    if not subset_samples:
        raise SystemExit(f"Split {args.subset} is empty in {args.split_json}")

    num_classes = task_num_classes(args.task)
    class_names = task_class_names(args.task)
    image_size = tuple(args.image_size)
    dataset = wrap_for_task(
        SegmentationDataset.from_samples(subset_samples, augment=False, image_size=image_size),
        args.task,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = get_device()
    model = build_deeplabv3(num_classes=num_classes, pretrained=False).to(device)
    extra = load_checkpoint(model, args.checkpoint, device)
    ckpt_task = extra.get("task")
    if ckpt_task and ckpt_task != args.task:
        print(f"Warning: checkpoint task={ckpt_task} but --task {args.task}")
    metrics = evaluate_loader(model, loader, device, num_classes=num_classes, class_names=class_names)
    metrics["subset"] = args.subset
    metrics["task"] = args.task
    metrics["split_json"] = str(Path(args.split_json).resolve())
    metrics["n_cases"] = len({s.get("case_id") for s in subset_samples})
    metrics["cases"] = sorted({s.get("case_id") for s in subset_samples})

    out = Path(args.out) if args.out else Path(args.split_json).parent / f"{args.subset}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics, out)
    print(f"=== {args.subset} ({metrics['n_images']} frames, {metrics['n_cases']} cases) ===")
    print(format_metrics(metrics))
    print(f"Wrote {out}")

    csv_path = Path(args.csv) if args.csv else out.with_name(f"{args.subset}_per_image.csv")
    rows = per_image_rows(model, subset_samples, image_size, device, num_classes, args.task, class_names)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    vis_dir = Path(args.vis_dir) if args.vis_dir else out.parent / f"{args.subset}_overlays"
    if args.num_vis > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
        n_save = args.num_vis
        with torch.no_grad():
            for i, (images, masks) in enumerate(loader):
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
                        str(vis_dir / f"{args.subset}_result_{i}_{j}.png"),
                        num_classes=num_classes,
                    )
                n_save -= images.shape[0]
                if n_save <= 0:
                    break
        print(f"Wrote overlays under {vis_dir}")


if __name__ == "__main__":
    main()
