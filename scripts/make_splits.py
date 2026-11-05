#!/usr/bin/env python3
"""Write a patient-level train/val/test split for data/processed."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cystic_seg.dataset import load_data_with_masks
from cystic_seg.splits import case_id_of, make_split_info, save_split, samples_for_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split JSON")
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="experiments/train_test/split.json")
    parser.add_argument("--unit", choices=["case", "sample"], default="case")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_data_with_masks(args.data_root)
    if not samples:
        raise SystemExit(f"No samples under {args.data_root}")
    info = make_split_info(
        samples,
        unit=args.unit,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    out = Path(args.out)
    save_split(info, out)

    csv_path = out.with_name("split_cases.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "case_id", "n_frames"])
        writer.writeheader()
        for subset in ("train", "val", "test"):
            counts = Counter(case_id_of(s) for s in samples_for_subset(samples, info, subset))
            for case_id, n in sorted(counts.items()):
                writer.writerow({"split": subset, "case_id": case_id, "n_frames": n})

    print(f"Unit={info['unit']} seed={info['seed']}")
    print(f"Train {info['n_train']} frames / {len(info['train_cases'])} cases")
    print(f"Val   {info['n_val']} frames / {len(info['val_cases'])} cases")
    print(f"Test  {info['n_test']} frames / {len(info['test_cases'])} cases")
    print(f"Wrote {out}")
    print(f"Wrote {csv_path}")
    overlap_tv = set(info["train_cases"]) & set(info["val_cases"])
    overlap_tt = set(info["train_cases"]) & set(info["test_cases"])
    overlap_vt = set(info["val_cases"]) & set(info["test_cases"])
    if overlap_tv or overlap_tt or overlap_vt:
        raise SystemExit(f"Case leak: train∩val={overlap_tv} train∩test={overlap_tt} val∩test={overlap_vt}")
    print("No patient overlap across splits.")


if __name__ == "__main__":
    main()
