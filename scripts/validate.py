#!/usr/bin/env python3
"""Backward-compatible entry: evaluate the val split. Prefer evaluate_split.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cystic_seg.constants import IMAGE_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate val split (wrapper around evaluate_split.py)")
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--checkpoint", type=str, default="experiments/train_test/deeplabv3_best.pth")
    parser.add_argument("--split-json", type=str, default="experiments/train_test/split.json")
    parser.add_argument("--out", type=str, default="experiments/train_test/val_metrics.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_split.py"),
        "--data-root", args.data_root,
        "--checkpoint", args.checkpoint,
        "--split-json", args.split_json,
        "--subset", "val",
        "--out", args.out,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--image-size", str(args.image_size[0]), str(args.image_size[1]),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
