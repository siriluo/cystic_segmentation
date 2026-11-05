#!/usr/bin/env python3
"""Clip a video to a time interval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cystic_seg.video import chunk_video, parse_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a video file by time interval.")
    parser.add_argument("input_file", type=str, help="Input video file path")
    parser.add_argument("output_file", type=str, help="Output video file path")
    parser.add_argument("start_time", type=str, help="Start time (hh:mm:ss, mm:ss, or ss)")
    parser.add_argument("end_time", type=str, help="End time (hh:mm:ss, mm:ss, or ss)")
    args = parser.parse_args()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    chunk_video(args.input_file, args.output_file, parse_time(args.start_time), parse_time(args.end_time))


if __name__ == "__main__":
    main()
