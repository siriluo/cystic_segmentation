"""Losses that emphasize common bile duct vs cystic duct."""

from __future__ import annotations

import torch
import torch.nn as nn


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, class_id: int, eps: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, class_id]
    gt = (target == class_id).float()
    if gt.sum() + probs.sum() < eps:
        return logits.new_tensor(1.0)
    inter = (probs * gt).sum()
    return (2 * inter + eps) / (probs.sum() + gt.sum() + eps)


def swap_penalty(logits: torch.Tensor, target: torch.Tensor, id_a: int, id_b: int) -> torch.Tensor:
    """Penalize probability of the sibling duct on pixels labeled as this duct."""
    probs = torch.softmax(logits, dim=1)
    loss = logits.new_tensor(0.0)
    n = 0
    for src, wrong in ((id_a, id_b), (id_b, id_a)):
        mask = target == src
        if mask.any():
            loss = loss + probs[:, wrong][mask].mean()
            n += 1
    if n == 0:
        return loss
    return loss / n


class BileDuctLoss(nn.Module):
    """CE + Dice on CBD/CD + extra penalty for swapping the two ducts."""

    def __init__(
        self,
        cbd_id: int = 1,
        cd_id: int = 2,
        ce_weight: float = 1.0,
        dice_weight: float = 2.0,
        swap_weight: float = 1.5,
        ignore_index: int = -100,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cbd_id = cbd_id
        self.cd_id = cd_id
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.swap_weight = swap_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice_cbd = dice_score_from_logits(logits, target, self.cbd_id)
        dice_cd = dice_score_from_logits(logits, target, self.cd_id)
        dice_loss = 1.0 - 0.5 * (dice_cbd + dice_cd)
        swap = swap_penalty(logits, target, self.cbd_id, self.cd_id)
        return self.ce_weight * ce + self.dice_weight * dice_loss + self.swap_weight * swap
