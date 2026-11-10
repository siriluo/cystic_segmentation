"""CholecSeg8k dataset loading, parsing, and preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cystic_seg.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# 13 standard classes in CholecSeg8k
CHOLECSEG8K_CLASSES: Dict[int, str] = {
    0: "Background / Frame Border",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}

# Watershed mask grayscale value -> 13 CholecSeg8k classes (0..12)
WATERSHED_TO_CHOLECSEG_CLASS: Dict[int, int] = {
    0: 0,    # Border / background
    50: 0,   # Black Background (outside FOV)
    255: 0,  # Watershed boundary artifact
    11: 1,   # Abdominal Wall
    21: 2,   # Liver
    13: 3,   # Gastrointestinal Tract
    12: 4,   # Fat
    31: 5,   # Grasper
    23: 6,   # Connective Tissue
    24: 7,   # Blood
    25: 8,   # Cystic Duct
    32: 9,   # L-hook Electrocautery
    22: 10,  # Gallbladder
    33: 11,  # Hepatic Vein
    5: 12,   # Liver Ligament
}

# Mapping directly from watershed mask value to cystic_seg 8-class target space:
# 0: background
# 1: Gallbladder
# 2: Common bile duct
# 3: Cystic duct
# 4: Instrument (Grasper, L-hook)
# 5: Liver edge / Liver
# 6: Cystic artery
# 7: Rouvière's sulcus
WATERSHED_TO_CYSTIC_CLASS: Dict[int, int] = {
    0: 0,
    50: 0,
    255: 0,
    11: 0,   # Abdominal wall -> background
    21: 5,   # Liver -> 5 (Liver edge / liver)
    13: 0,   # GI tract -> background
    12: 0,   # Fat -> background
    31: 4,   # Grasper -> 4 (Instrument)
    23: 0,   # Connective tissue -> background
    24: 0,   # Blood -> background
    25: 3,   # Cystic Duct -> 3 (Cystic duct)
    32: 4,   # L-hook -> 4 (Instrument)
    22: 1,   # Gallbladder -> 1 (Gallbladder)
    33: 0,   # Hepatic vein -> background
    5: 0,    # Liver ligament -> background
}

_LOOKUP_13 = np.zeros(256, dtype=np.int64)
for val, cid in WATERSHED_TO_CHOLECSEG_CLASS.items():
    _LOOKUP_13[val] = cid

_LOOKUP_CYSTIC = np.zeros(256, dtype=np.int64)
for val, cid in WATERSHED_TO_CYSTIC_CLASS.items():
    _LOOKUP_CYSTIC[val] = cid


def parse_watershed_mask(mask_path: str | Path, target_space: str = "cholecseg") -> np.ndarray:
    """Read a CholecSeg8k watershed mask and map to class IDs.

    target_space: 'cholecseg' (0..12) or 'cystic' (0..7)
    """
    with Image.open(mask_path) as img:
        arr = np.asarray(img.convert("L"))
    lookup = _LOOKUP_CYSTIC if target_space == "cystic" else _LOOKUP_13
    return lookup[arr]


def discover_cholecseg8k_samples(root_dir: str | Path) -> List[Dict[str, str]]:
    """Recursively search root_dir for CholecSeg8k frames and matching watershed masks."""
    root = Path(root_dir)
    samples: List[Dict[str, str]] = []

    for img_path in sorted(root.glob("**/*_endo.png")):
        base = str(img_path)[:-len("_endo.png")]
        ws_mask = Path(f"{base}_endo_watershed_mask.png")
        if not ws_mask.exists():
            ws_mask = Path(f"{base}_endo_mask.png")
        if ws_mask.exists():
            rel_parts = img_path.relative_to(root).parts
            video_id = rel_parts[0] if len(rel_parts) > 1 else "video_unknown"
            samples.append({
                "image_path": str(img_path),
                "mask_path": str(ws_mask),
                "video_id": video_id,
            })
    return samples


class CholecSeg8kDataset(Dataset):
    """Dataset for CholecSeg8k frames with augmentations."""

    def __init__(
        self,
        samples: Sequence[Dict[str, str]],
        image_size: Tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        target_space: str = "cholecseg",
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.target_space = target_space
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        with Image.open(item["image_path"]) as img:
            image = img.convert("RGB")
        mask = parse_watershed_mask(item["mask_path"], target_space=self.target_space)

        if self.augment:
            if np.random.rand() > 0.5:
                image = transforms.functional.hflip(image)
                mask = np.fliplr(mask)
            if np.random.rand() > 0.5:
                image = transforms.functional.vflip(image)
                mask = np.flipud(mask)
            angle = int(np.random.choice([0, 90, 180, 270]))
            if angle != 0:
                image = transforms.functional.rotate(image, angle)
                mask = np.rot90(mask, k=angle // 90)
            image = self.color_jitter(image)

        image = transforms.functional.resize(image, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(
            Image.fromarray(mask.astype(np.uint8)),
            self.image_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        image_t = self.normalize(self.to_tensor(image))
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_t, mask_t
