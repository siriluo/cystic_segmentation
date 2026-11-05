#!/usr/bin/env python3
"""Render processed frames with masks so annotations can be checked by eye.

Writes a three-panel figure per sample (image | class mask | overlay) and an
HTML gallery for browsing.

Examples
--------
    # One random frame from every patient (good first pass)
    python scripts/preview_processed.py --mode per-case

    # 20 random frames
    python scripts/preview_processed.py --mode random --n 20

    # Every frame from two cases
    python scripts/preview_processed.py --mode all --cases Case01,Case45
"""

from __future__ import annotations

import argparse
import html
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cystic_seg.constants import CLASS_NAMES, NUM_CLASSES
from cystic_seg.dataset import load_data_with_masks
from cystic_seg.viz import class_centroids, class_colormap, mask_legend_handles, overlay_mask_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save image / mask / overlay previews from data/processed"
    )
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="experiments/processed_previews")
    parser.add_argument(
        "--mode",
        choices=["per-case", "random", "all"],
        default="per-case",
        help="per-case: one frame per patient; random: --n frames; all: every frame",
    )
    parser.add_argument("--n", type=int, default=20, help="How many frames in random mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Optional comma-separated case ids, e.g. Case01,Case45,CaseX03",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity on the raw image")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--layout",
        choices=["two", "three"],
        default="three",
        help="two: image | mask; three: image | mask | overlay",
    )
    parser.add_argument(
        "--require-classes",
        type=str,
        default="",
        help="Keep frames that contain these class ids, e.g. 2,3 for CBD and cystic duct",
    )
    return parser.parse_args()


def select_samples(samples: list[dict], args: argparse.Namespace) -> list[dict]:
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    if wanted:
        samples = [s for s in samples if s.get("case_id") in wanted]
        missing = wanted - {s.get("case_id") for s in samples}
        if missing:
            print(f"No frames for cases: {sorted(missing)}")
    if not samples:
        return []

    required = {int(x) for x in args.require_classes.split(",") if x.strip()}
    if required:
        samples = [s for s in samples if required.issubset(set(np.unique(s["mask"]).astype(int)))]
        print(f"After requiring classes {sorted(required)}: {len(samples)} frames")
    if not samples:
        return []

    rng = random.Random(args.seed)
    if args.mode == "all":
        return samples
    if args.mode == "random":
        k = min(args.n, len(samples))
        return rng.sample(samples, k)

    by_case: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_case[sample.get("case_id") or "CaseUnknown"].append(sample)
    chosen = []
    for case_id in sorted(by_case):
        chosen.append(rng.choice(by_case[case_id]))
    return chosen


def sample_stem(sample: dict) -> str:
    path = Path(sample["image_path"])
    case = sample.get("case_id") or "CaseUnknown"
    return f"{case}__{path.parent.name}"


def present_labels(mask: np.ndarray) -> str:
    names = [CLASS_NAMES[int(c)] for c in np.unique(mask) if int(c) != 0]
    return ", ".join(names) if names else "(none)"


def save_preview(sample: dict, out_path: Path, alpha: float, dpi: int, layout: str = "three") -> None:
    image = np.array(Image.open(sample["image_path"]).convert("RGB"))
    mask = sample["mask"]
    cmap = class_colormap()
    ncols = 2 if layout == "two" else 3
    fig, axs = plt.subplots(1, ncols, figsize=(7 * ncols, 6.5))
    axs[0].imshow(image)
    axs[0].set_title("Original image")
    axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    axs[1].set_title("Annotated mask")
    if layout == "three":
        overlay = overlay_mask_rgb(image, mask, alpha=alpha)
        centroids = class_centroids(mask)
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay")
        for class_id, (row, col) in centroids.items():
            axs[2].text(
                col,
                row,
                CLASS_NAMES.get(class_id, str(class_id)),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
            )
    for ax in axs:
        ax.axis("off")
    handles = mask_legend_handles(mask)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=True)
    case = sample.get("case_id") or "CaseUnknown"
    fig.suptitle(f"{case}  |  {Path(sample['image_path']).parent.name}\n{present_labels(mask)}", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_index(rows: list[dict], out_dir: Path) -> None:
    cards = []
    for row in rows:
        rel = html.escape(row["file"])
        title = html.escape(row["title"])
        labels = html.escape(row["labels"])
        cards.append(
            f'<div class="card"><a href="{rel}"><img src="{rel}" alt="{title}"></a>'
            f"<p><strong>{title}</strong><br>{labels}</p></div>"
        )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Processed annotation previews</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; background: #111; color: #eee; }}
    h1 {{ font-size: 1.2rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }}
    .card {{ background: #1c1c1c; padding: 8px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; border-radius: 4px; }}
    p {{ font-size: 0.85rem; margin: 0.4rem 0 0; }}
  </style>
</head>
<body>
  <h1>{len(rows)} annotation previews from data/processed</h1>
  <p>Left: original image. Right: annotated mask. Open a card to view the full figure.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples = load_data_with_masks(args.data_root)
    print(f"Loaded {len(samples)} samples from {args.data_root}")
    selected = select_samples(samples, args)
    if not selected:
        raise SystemExit("No samples selected.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(selected, start=1):
        stem = sample_stem(sample)
        filename = f"{i:03d}_{stem}.png"
        dest = out_dir / filename
        save_preview(sample, dest, alpha=args.alpha, dpi=args.dpi, layout=args.layout)
        rows.append(
            {
                "file": filename,
                "title": f"{sample.get('case_id')} / {Path(sample['image_path']).parent.name}",
                "labels": present_labels(sample["mask"]),
            }
        )
        print(f"[{i}/{len(selected)}] {filename}  ({rows[-1]['labels']})")

    write_index(rows, out_dir)
    print(f"Wrote {len(rows)} figures and {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
