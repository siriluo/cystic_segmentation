"""Dataset, augmentations, and VIA polygon-to-mask loading."""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cystic_seg.constants import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
    LABEL_PRIORITY,
    canonicalize_label,
)


class SegmentationAugmentation:
    """Apply the same geometric transform to image and mask; color jitter on image only."""

    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE, duct_crop_prob: float = 0.0):
        self.image_size = image_size
        self.duct_crop_prob = duct_crop_prob
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if self.duct_crop_prob > 0 and np.random.rand() < self.duct_crop_prob:
            cropped = _crop_around_classes(image, np.ascontiguousarray(mask), classes=(2, 3), margin=48)
            if cropped is not None:
                image, mask = cropped
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(
            Image.fromarray(mask),
            self.image_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        mask = np.array(mask)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image, torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.long)


def _crop_around_classes(
    image: Image.Image,
    mask: np.ndarray,
    classes: Tuple[int, ...],
    margin: int = 48,
) -> Optional[Tuple[Image.Image, np.ndarray]]:
    ys, xs = np.where(np.isin(mask, classes))
    if ys.size < 20:
        return None
    h, w = mask.shape
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h, y1 + margin + 1)
    x1 = min(w, x1 + margin + 1)
    side = max(y1 - y0, x1 - x0, 64)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    y1 = min(h, y0 + side)
    x1 = min(w, x0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, x1 - side)
    return image.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1]


class RemapDataset(Dataset):
    """Remap integer masks after the base dataset returns (image, mask)."""

    def __init__(self, base: Dataset, mapping: np.ndarray):
        self.base = base
        self.mapping = np.asarray(mapping, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, mask = self.base[idx]
        mapped = torch.from_numpy(self.mapping[mask.cpu().numpy()]).to(dtype=torch.long)
        return image, mapped


def eval_image_transform(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ANN_EXTS = {".txt", ".json"}
OVERLAY_RE = re.compile(r"(annotat|labels?\b|_label\b|スライド)", re.I)
CASE_RE = re.compile(r"case[_ ]?(\d+)", re.I)
CASE_RANGE_RE = re.compile(r"case[_ ]?\d+\s*-\s*\d+", re.I)
CASE_X_RE = re.compile(r"casex(\d+)", re.I)
FOLDER_X_RE = re.compile(r"^x(\d+)(?:_|$)", re.I)


def case_id_from_path(path: str) -> str:
    """Patient id from a file path, preferring the deepest folder/filename."""
    for part in reversed(Path(path).parts):
        if CASE_RANGE_RE.search(part):
            continue
        x_match = CASE_X_RE.search(part) or FOLDER_X_RE.match(part)
        if x_match:
            return f"CaseX{int(x_match.group(1)):02d}"
        match = CASE_RE.search(part)
        if match:
            return f"Case{int(match.group(1)):02d}"
    return "CaseUnknown"


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def is_overlay_name(name: str) -> bool:
    stem = Path(name).stem
    return bool(OVERLAY_RE.search(stem))


def train_val_split_samples(
    samples: Sequence[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split = int(len(samples) * (1 - val_ratio))
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]
    return train_samples, val_samples


def train_val_split_by_case(
    samples: Sequence[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Hold out entire patients so frames from one case are not in both splits."""
    cases = sorted({s.get("case_id") or case_id_from_path(s["image_path"]) for s in samples})
    random.seed(seed)
    random.shuffle(cases)
    n_val = max(1, int(round(len(cases) * val_ratio)))
    val_cases = set(cases[:n_val])
    train_samples = [s for s in samples if (s.get("case_id") or case_id_from_path(s["image_path"])) not in val_cases]
    val_samples = [s for s in samples if (s.get("case_id") or case_id_from_path(s["image_path"])) in val_cases]
    if not train_samples or not val_samples:
        return train_val_split_samples(samples, val_ratio=val_ratio, seed=seed)
    return train_samples, val_samples


class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        samples: Optional[List[dict]] = None,
        transform: Optional[Callable] = None,
        augment: bool = False,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        duct_crop_prob: float = 0.0,
    ):
        if samples is not None:
            self.samples = samples
        elif data_root is not None:
            self.samples = load_data_with_masks(data_root)
        else:
            raise ValueError("Either data_root or samples must be provided")
        self.image_size = image_size
        self.augment = augment
        if augment:
            self.transform = SegmentationAugmentation(image_size=image_size, duct_crop_prob=duct_crop_prob)
        else:
            self.transform = transform or eval_image_transform(image_size)

    @classmethod
    def from_samples(cls, samples: List[dict], **kwargs) -> "SegmentationDataset":
        return cls(samples=samples, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = sample["mask"]
        if self.augment:
            image, mask = self.transform(image, mask)
        else:
            image = self.transform(image)
            mask = Image.fromarray(mask)
            mask = transforms.functional.resize(
                mask,
                self.image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask


def get_dataloader(
    data_root: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = False,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> DataLoader:
    dataset = SegmentationDataset(data_root, augment=augment, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _annotation_regions(anno: dict, image_key: str) -> dict:
    if image_key in anno:
        rec = anno[image_key]
        if isinstance(rec, dict):
            return rec.get("regions", {}) or {}
    wanted = normalize_name(image_key)
    for key, rec in anno.items():
        if isinstance(rec, dict) and normalize_name(str(key)) == wanted:
            return rec.get("regions", {}) or {}
        if isinstance(rec, dict) and normalize_name(str(rec.get("filename", ""))) == wanted:
            return rec.get("regions", {}) or {}
    if len(anno) == 1:
        only = next(iter(anno.values()))
        if isinstance(only, dict):
            return only.get("regions", {}) or {}
    return {}


def _read_via_json(path: str) -> Optional[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def load_data_with_masks(data_root: str) -> List[dict]:
    """Load VIA polygon annotations and rasterize them into class masks.

    Expected layout::

        data_root/
          CaseXX/
            clip_name/
              image.jpg
              annotation.json   # or matching .txt / .json

    Overlapping polygons are resolved by priority:
    common bile duct > cystic duct > cystic artery > remaining classes.
    """
    samples: List[dict] = []
    for dirpath, _, filenames in os.walk(data_root):
        images = sorted(
            f
            for f in filenames
            if Path(f).suffix.lower() in IMAGE_EXTS and not is_overlay_name(f)
        )
        anns = sorted(f for f in filenames if Path(f).suffix.lower() in ANN_EXTS)
        if not images or not anns:
            continue
        stem_to_ann = {Path(a).stem: a for a in anns}
        for img_name in images:
            stem = Path(img_name).stem
            ann_name = stem_to_ann.get(stem)
            if ann_name is None and stem == "image" and "annotation" in stem_to_ann:
                ann_name = stem_to_ann["annotation"]
            if ann_name is None and len(images) == 1 and len(anns) == 1:
                ann_name = anns[0]
            if ann_name is None:
                continue
            img_path = os.path.join(dirpath, img_name)
            ann_path = os.path.join(dirpath, ann_name)
            try:
                sample = _load_one_sample(img_path, ann_path)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
            if sample is not None:
                samples.append(sample)
    return samples


def _load_one_sample(img_path: str, txt_path: str) -> Optional[dict]:
    with Image.open(img_path) as img:
        width, height = img.size
    anno = _read_via_json(txt_path)
    if not anno:
        return None
    regions = _annotation_regions(anno, os.path.basename(img_path))
    if not regions:
        # VIA keys often use the original filename, not image.jpg
        for key in anno:
            rec = anno[key]
            if isinstance(rec, dict) and rec.get("regions"):
                regions = rec["regions"]
                break
    if not regions:
        return None

    region_list = []
    for region in regions.values():
        raw = region.get("region_attributes", {}).get("label")
        label = canonicalize_label(raw)
        if label is None or label not in LABEL_MAP:
            continue
        priority = LABEL_PRIORITY.get(label, 100)
        region_list.append((priority, label, region))
    region_list.sort(key=lambda x: x[0])

    mask = np.zeros((height, width), dtype=np.uint8)
    for _, label, region in region_list:
        class_id = LABEL_MAP[label]
        shape = region["shape_attributes"]
        if shape.get("name") != "polygon":
            continue
        all_x = np.array(shape["all_points_x"])
        all_y = np.array(shape["all_points_y"])
        pts = list(zip(all_x.tolist(), all_y.tolist()))
        region_img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(region_img).polygon(pts, outline=1, fill=1)
        region_mask = np.array(region_img)
        rr, cc = np.where(region_mask > 0)
        if rr.size == 0:
            continue
        if label == "Commonbileduct":
            mask[rr, cc] = class_id
        else:
            overwrite = mask[rr, cc] == 0
            mask[rr[overwrite], cc[overwrite]] = class_id
    if int(mask.max()) == 0:
        return None
    return {
        "image_path": img_path,
        "mask": mask,
        "regions": regions,
        "case_id": case_id_from_path(img_path),
        "ann_path": txt_path,
    }
