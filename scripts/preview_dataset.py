#!/usr/bin/env python3
"""Preview rasterized annotation masks for a few random samples."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cystic_seg.dataset import load_data_with_masks
from cystic_seg.viz import class_colormap, mask_legend_handles


def main() -> None:
    parser = argparse.ArgumentParser(description="Save image/mask previews")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="experiments/previews")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_data_with_masks(args.data_root)
    print(f"Loaded {len(samples)} samples from {args.data_root}")
    if not samples:
        raise SystemExit("No samples found.")

    random.seed(args.seed)
    selected = random.sample(samples, min(args.n, len(samples)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = class_colormap()

    for i, sample in enumerate(selected, start=1):
        print(f"Sample {i}: {sample['image_path']}")
        print(f"  mask shape={sample['mask'].shape} unique={np.unique(sample['mask'])}")
        raw = Image.open(sample["image_path"])
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        axs[0].imshow(raw)
        axs[0].set_title("Raw Image")
        axs[0].axis("off")
        axs[1].imshow(sample["mask"], cmap=cmap, vmin=0, vmax=cmap.N - 1)
        axs[1].set_title("Mask")
        axs[1].axis("off")
        handles = mask_legend_handles(sample["mask"])
        if handles:
            axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(out_dir / f"sample_{i}_mask.png")
        plt.close(fig)
    print(f"Wrote previews to {out_dir}")


if __name__ == "__main__":
    main()
