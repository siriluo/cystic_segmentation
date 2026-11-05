"""Run validation metrics on a loader and optionally write JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from cystic_seg.constants import CLASS_NAMES, NUM_CLASSES
from cystic_seg.metrics import compute_dice, compute_iou, mean_iou


def _duct_ids(class_names: Dict[int, str]) -> tuple[int | None, int | None]:
    by_name = {str(v).lower(): int(k) for k, v in class_names.items()}
    return by_name.get("common bile duct"), by_name.get("cystic duct")


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    class_names: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    names = class_names or CLASS_NAMES
    cbd_id, cd_id = _duct_ids(names)
    model.eval()
    ious: List[List[float]] = []
    dices: List[List[float]] = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_images = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1).cpu()
            n_images += int(masks.shape[0])
            for b in range(preds.shape[0]):
                ious.append(compute_iou(preds[b], masks[b], num_classes))
                dices.append(compute_dice(preds[b], masks[b], num_classes))
                t = masks[b].reshape(-1).numpy()
                p = preds[b].reshape(-1).numpy()
                valid = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
                idx = t[valid].astype(np.int64) * num_classes + p[valid].astype(np.int64)
                confusion += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    iou_arr = np.array(ious, dtype=float)
    dice_arr = np.array(dices, dtype=float)
    per_iou, miou, fg_miou = mean_iou(iou_arr)
    per_dice, mdice, fg_mdice = mean_iou(dice_arr)
    per_class = []
    for cls_id in range(num_classes):
        per_class.append(
            {
                "id": cls_id,
                "name": names.get(cls_id, str(cls_id)),
                "iou": None if np.isnan(per_iou[cls_id]) else float(per_iou[cls_id]),
                "dice": None if np.isnan(per_dice[cls_id]) else float(per_dice[cls_id]),
            }
        )
    return {
        "n_batches": int(len(ious)),
        "n_images": n_images,
        "mean_iou": miou,
        "foreground_miou": fg_miou,
        "mean_dice": mdice,
        "foreground_mdice": fg_mdice,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "swap_rates": swap_rates(confusion, cbd_id=cbd_id, cd_id=cd_id) if cbd_id is not None and cd_id is not None else {},
    }


def write_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def swap_rates(
    confusion: list | np.ndarray,
    cbd_id: int | None = 1,
    cd_id: int | None = 2,
) -> Dict[str, float | None]:
    """Pixel fractions of each duct predicted as the other (the BDI-relevant error)."""
    if cbd_id is None or cd_id is None:
        return {}
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] <= max(cbd_id, cd_id):
        return {}

    def row_frac(src: int, dst: int) -> float | None:
        total = cm[src].sum()
        if total <= 0:
            return None
        return float(cm[src, dst] / total)

    return {
        "cbd_recall": row_frac(cbd_id, cbd_id),
        "cd_recall": row_frac(cd_id, cd_id),
        "cbd_predicted_as_cd": row_frac(cbd_id, cd_id),
        "cd_predicted_as_cbd": row_frac(cd_id, cbd_id),
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    lines = ["Per-class metrics:"]
    for row in metrics["per_class"]:
        iou = "  nan" if row["iou"] is None else f"{row['iou']:.4f}"
        dice = "  nan" if row["dice"] is None else f"{row['dice']:.4f}"
        lines.append(f"  {row['id']} {row['name']:20s}  IoU {iou}  Dice {dice}")
    lines.append(f"Mean IoU (all):         {metrics['mean_iou']:.4f}")
    lines.append(f"Mean IoU (foreground):  {metrics['foreground_miou']:.4f}")
    lines.append(f"Mean Dice (all):        {metrics['mean_dice']:.4f}")
    lines.append(f"Mean Dice (foreground): {metrics['foreground_mdice']:.4f}")
    if metrics.get("swap_rates"):
        rates = metrics["swap_rates"]
    elif metrics.get("confusion"):
        names = {row["id"]: row["name"] for row in metrics["per_class"]}
        cbd_id, cd_id = _duct_ids(names)
        rates = swap_rates(metrics["confusion"], cbd_id=cbd_id, cd_id=cd_id)
    else:
        rates = {}
    if rates:
        def fmt(v: float | None) -> str:
            return "  nan" if v is None else f"{v:.4f}"

        lines.append(f"CBD pixel recall:       {fmt(rates['cbd_recall'])}")
        lines.append(f"CD pixel recall:        {fmt(rates['cd_recall'])}")
        lines.append(f"CBD predicted as CD:    {fmt(rates['cbd_predicted_as_cd'])}")
        lines.append(f"CD predicted as CBD:    {fmt(rates['cd_predicted_as_cbd'])}")
    return "\n".join(lines)
