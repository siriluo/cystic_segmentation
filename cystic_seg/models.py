"""Model factory for DeepLabV3-ResNet50."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

from cystic_seg.constants import NUM_CLASSES


def build_deeplabv3(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build DeepLabV3-ResNet50 with a custom classifier head.

    Supports both current torchvision (`weights=`) and older (`pretrained=`) APIs.
    """
    try:
        if pretrained:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            model = deeplabv3_resnet50(weights=None)
    except TypeError:
        model = deeplabv3_resnet50(pretrained=pretrained)

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a raw state_dict or a dict checkpoint with a `model` key."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    extra: Dict[str, Any] = {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        extra = ckpt
    elif isinstance(ckpt, dict) and "backbone" in ckpt:
        state = {f"backbone.{k}": v for k, v in ckpt["backbone"].items()}
        extra = ckpt
    else:
        state = ckpt

    model_state = model.state_dict()
    filtered_state = {}
    ignored_keys = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            ignored_keys.append(k)

    if ignored_keys:
        print(f"load_checkpoint: matched {len(filtered_state)} keys, skipped {len(ignored_keys)} mismatched/unused keys (e.g. {ignored_keys[:3]})")

    model.load_state_dict(filtered_state, strict=False)
    return extra


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_loss: float = float("inf"),
    **kwargs: Any,
) -> None:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        **kwargs,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
