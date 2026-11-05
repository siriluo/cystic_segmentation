#!/usr/bin/env python3
"""Split a video into frames, run segmentation, and stitch an overlay video.

Does not require OpenCV. Frame decode/encode uses ffmpeg (`module load ffmpeg/6.1.1`
on DeltaAI).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cystic_seg.constants import IMAGE_SIZE
from cystic_seg.dataset import eval_image_transform
from cystic_seg.models import build_deeplabv3, get_device, load_checkpoint
from cystic_seg.tasks import get_task
from cystic_seg.video import extract_frames, probe_video, stitch_frames
from cystic_seg.viz import draw_class_labels, draw_legend, overlay_mask_rgb, resize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay landmark predictions on a video")
    parser.add_argument("--video", type=str, default="chunked_video.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="outputs/chunked_video_segmented.mp4")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/train_test/deeplabv3_best.pth",
        help="Trained weights (must match --task)",
    )
    parser.add_argument(
        "--task",
        choices=["landmarks", "bile-ducts", "ducts-only"],
        default="landmarks",
        help="Must match how the checkpoint was trained.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth extracted frame")
    parser.add_argument("--work-dir", type=str, default=None, help="Where to write raw/overlay JPEGs")
    parser.add_argument("--keep-frames", action="store_true", help="Do not delete the work directory")
    parser.add_argument("--no-labels", action="store_true", help="Skip class-name text on the overlay")
    parser.add_argument("--no-legend", action="store_true", help="Skip the color key")
    return parser.parse_args()


class FrameFolder(Dataset):
    def __init__(self, paths: list[Path], image_size: tuple[int, int]):
        self.paths = paths
        self.transform = eval_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        tensor = self.transform(image)
        return tensor, idx, width, height


def paint_overlay(
    rgb: np.ndarray,
    mask_small: np.ndarray,
    class_names: dict,
    num_classes: int,
    alpha: float,
    labels: bool,
    legend: bool,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    mask = resize_mask(mask_small, (width, height))
    overlay = overlay_mask_rgb(rgb, mask, alpha=alpha, num_classes=num_classes)
    if labels:
        overlay = draw_class_labels(overlay, mask, class_names=class_names)
    if legend:
        overlay = draw_legend(overlay, class_names, num_classes=num_classes)
    return overlay


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    spec = get_task(args.task)
    image_size = tuple(args.image_size)
    work = Path(args.work_dir) if args.work_dir else Path(args.output).with_suffix("").parent / f"{video.stem}_frames"
    raw_dir = work / "raw"
    overlay_dir = work / "overlay"
    if work.exists() and not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(video)
    dur = f"  {info['duration']:.1f}s" if info["duration"] is not None else ""
    print(f"{video}  {info['width']}x{info['height']}  {info['fps']:.2f} fps  {info['nb_frames'] or '?'} frames{dur}")

    print("Extracting frames with ffmpeg…")
    frames = extract_frames(video, raw_dir, max_frames=args.max_frames)
    if args.stride > 1:
        frames = frames[:: args.stride]
    print(f"Running inference on {len(frames)} frames ({args.task}, {spec.num_classes} classes)")

    device = get_device()
    model = build_deeplabv3(num_classes=spec.num_classes, pretrained=False).to(device)
    extra = load_checkpoint(model, ckpt, device)
    ckpt_task = extra.get("task")
    if ckpt_task and ckpt_task != args.task:
        print(f"Warning: checkpoint task={ckpt_task} but --task {args.task}")
    model.eval()

    dataset = FrameFolder(frames, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        for batch, indices, widths, heights in tqdm(loader, desc="infer"):
            outputs = model(batch.to(device))["out"]
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
            for pred, idx, width, height in zip(preds, indices.tolist(), widths.tolist(), heights.tolist()):
                src = frames[int(idx)]
                rgb = np.array(Image.open(src).convert("RGB"))
                overlay = paint_overlay(
                    rgb,
                    pred,
                    spec.class_names,
                    spec.num_classes,
                    args.alpha,
                    labels=not args.no_labels,
                    legend=not args.no_legend,
                )
                dest = overlay_dir / src.name
                Image.fromarray(overlay).save(dest, quality=90)

    fps_str = info["fps_str"]
    if args.stride > 1:
        fps_str = str(info["fps"] / args.stride)
    print("Stitching overlay video with ffmpeg…")
    stitch_frames(overlay_dir, args.output, fps_str=fps_str)
    print(f"Wrote {args.output}  ({len(frames)} frames)")

    if not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"Kept frames under {work}")


if __name__ == "__main__":
    main()
