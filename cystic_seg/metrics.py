"""Segmentation metrics."""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def _per_class_scores(pred: torch.Tensor, target: torch.Tensor, num_classes: int, kind: str) -> List[float]:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    scores: List[float] = []
    for cls in range(num_classes):
        pred_inds = pred_flat == cls
        target_inds = target_flat == cls
        intersection = (pred_inds & target_inds).sum().item()
        if kind == "iou":
            denom = pred_inds.sum().item() + target_inds.sum().item() - intersection
        else:
            denom = pred_inds.sum().item() + target_inds.sum().item()
            intersection = 2 * intersection
        if denom == 0:
            scores.append(float("nan"))
        else:
            scores.append(intersection / denom)
    return scores


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> List[float]:
    """Per-class IoU for a single batch of integer masks."""
    return _per_class_scores(pred, target, num_classes, "iou")


def compute_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> List[float]:
    """Per-class Dice for a single batch of integer masks."""
    return _per_class_scores(pred, target, num_classes, "dice")


def mean_iou(ious: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return per-class mean IoU, mIoU over all classes, and foreground mIoU."""
    per_class = np.nanmean(ious, axis=0)
    miou = float(np.nanmean(per_class))
    fg = float(np.nanmean(per_class[1:])) if per_class.size > 1 else miou
    return per_class, miou, fg
