#!/usr/bin/env python3
"""Merge CBD + cystic duct in the 8-class checkpoint and score the 48-frame test set."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cystic_seg.constants import CLASS_NAMES, NUM_CLASSES
from cystic_seg.dataset import SegmentationDataset, load_data_with_masks
from cystic_seg.evaluate import swap_rates
from cystic_seg.metrics import compute_dice, compute_iou
from cystic_seg.models import build_deeplabv3, get_device, load_checkpoint
from cystic_seg.splits import load_split, samples_for_subset

CBD_ID = 2
CD_ID = 3
WHITE_RE = re.compile(r"(non|no|without|w/?o)[_\s.-]*icg", re.I)
ICG_RE = re.compile(r"icg", re.I)


def classify_name(sample_id: str, source_image: str = "") -> str:
    text = f"{sample_id} {Path(source_image).name}"
    if WHITE_RE.search(text):
        return "white_light"
    if ICG_RE.search(text):
        return "icg"
    return "unlabeled"


def green_stats(image_path: str) -> dict:
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    # Laparoscopic frames have a dark circular FOV; ignore near-black pixels.
    lum = img.mean(axis=2)
    vis = lum > 15
    if vis.sum() < 1000:
        vis = np.ones(lum.shape, dtype=bool)
    pix = img[vis]
    r, g, b = pix.mean(axis=0)
    s = r + g + b + 1e-6
    return {
        "mean_r": float(r),
        "mean_g": float(g),
        "mean_b": float(b),
        "g_share": float(g / s),
        "g_minus_r": float(g - r),
    }


def binary_scores(pred_pos: np.ndarray, gt_pos: np.ndarray) -> dict:
    tp = int((pred_pos & gt_pos).sum())
    fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    gt_n = int(gt_pos.sum())
    pred_n = int(pred_pos.sum())
    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": gt_n,
        "pred_pixels": pred_n,
        "iou": None if denom_iou == 0 else tp / denom_iou,
        "dice": None if denom_dice == 0 else (2 * tp) / denom_dice,
        "recall": None if gt_n == 0 else tp / gt_n,
        "precision": None if pred_n == 0 else tp / pred_n,
    }


def nanmean(vals) -> float | None:
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not arr:
        return None
    return float(np.mean(arr))


def analyze_manifest(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_name = defaultdict(lambda: {"frames": 0, "patients": set()})
    patient_mods = defaultdict(set)
    unlabeled_rows = []
    for row in rows:
        mod = classify_name(row["sample_id"], row.get("source_image", ""))
        by_name[mod]["frames"] += 1
        by_name[mod]["patients"].add(row["case_id"])
        if mod != "unlabeled":
            patient_mods[row["case_id"]].add(mod)
        else:
            unlabeled_rows.append(row)

    # Color heuristic on unlabeled frames, calibrated later from named ICG vs white-light.
    color_rows = []
    named = [r for r in rows if classify_name(r["sample_id"], r.get("source_image", "")) != "unlabeled"]
    named_stats = {"icg": [], "white_light": []}
    for row in named:
        mod = classify_name(row["sample_id"], row.get("source_image", ""))
        st = green_stats(row["image"])
        st["case_id"] = row["case_id"]
        st["sample_id"] = row["sample_id"]
        named_stats[mod].append(st)

    def summarize(stats):
        if not stats:
            return {}
        return {
            "n": len(stats),
            "g_share_mean": float(np.mean([s["g_share"] for s in stats])),
            "g_share_p10": float(np.percentile([s["g_share"] for s in stats], 10)),
            "g_share_p90": float(np.percentile([s["g_share"] for s in stats], 90)),
            "g_minus_r_mean": float(np.mean([s["g_minus_r"] for s in stats])),
        }

    named_summary = {k: summarize(v) for k, v in named_stats.items()}
    # Threshold: midway between named ICG and white-light mean g_share if both exist.
    icg_g = named_summary.get("icg", {}).get("g_share_mean")
    wl_g = named_summary.get("white_light", {}).get("g_share_mean")
    if icg_g is not None and wl_g is not None:
        thresh = 0.5 * (icg_g + wl_g)
    else:
        thresh = 0.40

    inferred = defaultdict(lambda: {"frames": 0, "patients": set()})
    combined_patient = defaultdict(set)
    for row in rows:
        name_mod = classify_name(row["sample_id"], row.get("source_image", ""))
        if name_mod != "unlabeled":
            inferred[name_mod]["frames"] += 1
            inferred[name_mod]["patients"].add(row["case_id"])
            combined_patient[row["case_id"]].add(name_mod)
            continue
        st = green_stats(row["image"])
        st["case_id"] = row["case_id"]
        st["sample_id"] = row["sample_id"]
        color_mod = "icg" if st["g_share"] >= thresh else "white_light"
        st["inferred"] = color_mod
        color_rows.append(st)
        inferred[color_mod]["frames"] += 1
        inferred[color_mod]["patients"].add(row["case_id"])
        combined_patient[row["case_id"]].add(color_mod)

    def freeze(d):
        return {
            k: {"frames": v["frames"], "patients": sorted(v["patients"]), "n_patients": len(v["patients"])}
            for k, v in d.items()
        }

    both_name = sorted(cid for cid, mods in patient_mods.items() if mods >= {"icg", "white_light"})
    both_inferred = sorted(cid for cid, mods in combined_patient.items() if mods >= {"icg", "white_light"})
    only_icg_name = sorted(cid for cid, mods in patient_mods.items() if mods == {"icg"})
    only_wl_name = sorted(cid for cid, mods in patient_mods.items() if mods == {"white_light"})
    unlabeled_only_patients = sorted(
        {r["case_id"] for r in unlabeled_rows} - set(patient_mods)
    )

    return {
        "n_frames": len(rows),
        "n_patients": len({r["case_id"] for r in rows}),
        "manifest_columns": list(rows[0].keys()) if rows else [],
        "has_modality_column": any("modal" in c.lower() or c.lower() in {"icg", "white_light"} for c in (rows[0].keys() if rows else [])),
        "from_filename": freeze(by_name),
        "filename_patients_with_both": both_name,
        "filename_patients_icg_only": only_icg_name,
        "filename_patients_white_light_only": only_wl_name,
        "filename_patients_unlabeled_only": unlabeled_only_patients,
        "named_color_calibration": named_summary,
        "color_g_share_threshold": thresh,
        "unlabeled_color_inferred": freeze(
            defaultdict(lambda: {"frames": 0, "patients": set()}, {
                "icg": {
                    "frames": sum(1 for r in color_rows if r["inferred"] == "icg"),
                    "patients": {r["case_id"] for r in color_rows if r["inferred"] == "icg"},
                },
                "white_light": {
                    "frames": sum(1 for r in color_rows if r["inferred"] == "white_light"),
                    "patients": {r["case_id"] for r in color_rows if r["inferred"] == "white_light"},
                },
            })
        ),
        "filename_plus_color": freeze(inferred),
        "filename_plus_color_patients_with_both": both_inferred,
        "n_unlabeled_frames": len(unlabeled_rows),
        "unlabeled_sample_ids_by_case": {
            cid: sorted({r["sample_id"] for r in unlabeled_rows if r["case_id"] == cid})
            for cid in sorted({r["case_id"] for r in unlabeled_rows})
        },
    }


def evaluate_merged(checkpoint: Path, split_json: Path, data_root: Path) -> dict:
    info = load_split(split_json)
    samples = load_data_with_masks(str(data_root))
    subset = samples_for_subset(samples, info, "test")
    dataset = SegmentationDataset.from_samples(subset, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = get_device()
    print(f"device={device} n_test={len(subset)}", flush=True)
    model = build_deeplabv3(num_classes=NUM_CLASSES, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image = []
    pooled_bd = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
    ious = []
    dices = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            pred = torch.argmax(model(images.to(device))["out"], dim=1).cpu()
            ious.append(compute_iou(pred[0], masks[0], NUM_CLASSES))
            dices.append(compute_dice(pred[0], masks[0], NUM_CLASSES))
            t = masks[0].numpy().reshape(-1)
            p = pred[0].numpy().reshape(-1)
            valid = (t >= 0) & (t < NUM_CLASSES)
            t, p = t[valid].astype(np.int64), p[valid].astype(np.int64)
            confusion += np.bincount(t * NUM_CLASSES + p, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
                NUM_CLASSES, NUM_CLASSES
            )
            gt_bd = (t == CBD_ID) | (t == CD_ID)
            pred_bd = (p == CBD_ID) | (p == CD_ID)
            sc = binary_scores(pred_bd, gt_bd)
            sample = subset[idx]
            per_image.append(
                {
                    "case_id": sample.get("case_id", ""),
                    "image_path": sample["image_path"],
                    "gt_has_bile_duct": bool(sc["gt_pixels"] > 0),
                    "pred_has_bile_duct": bool(sc["pred_pixels"] > 0),
                    "iou_bile_duct": sc["iou"],
                    "dice_bile_duct": sc["dice"],
                    "recall_bile_duct": sc["recall"],
                    "precision_bile_duct": sc["precision"],
                    "gt_pixels": sc["gt_pixels"],
                    "pred_pixels": sc["pred_pixels"],
                    "tp": sc["tp"],
                }
            )
            pooled_bd["tp"] += sc["tp"]
            pooled_bd["fp"] += sc["fp"]
            pooled_bd["fn"] += sc["fn"]
            pooled_bd["gt"] += sc["gt_pixels"]
            pooled_bd["pred"] += sc["pred_pixels"]
            if (idx + 1) % 8 == 0 or idx == 0:
                print(f"  frame {idx + 1}/{len(subset)}", flush=True)

    iou_arr = np.array(ious, dtype=float)
    dice_arr = np.array(dices, dtype=float)
    per_class = []
    for cls_id in range(NUM_CLASSES):
        per_class.append(
            {
                "id": cls_id,
                "name": CLASS_NAMES[cls_id],
                "iou": None if np.isnan(np.nanmean(iou_arr[:, cls_id])) else float(np.nanmean(iou_arr[:, cls_id])),
                "dice": None if np.isnan(np.nanmean(dice_arr[:, cls_id])) else float(np.nanmean(dice_arr[:, cls_id])),
            }
        )

    tp, fp, fn = pooled_bd["tp"], pooled_bd["fp"], pooled_bd["fn"]
    pooled = {
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else None,
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
        "recall": tp / pooled_bd["gt"] if pooled_bd["gt"] else None,
        "precision": tp / pooled_bd["pred"] if pooled_bd["pred"] else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": pooled_bd["gt"],
        "pred_pixels": pooled_bd["pred"],
    }

    gt_frames = [r for r in per_image if r["gt_has_bile_duct"]]
    detected = [r for r in gt_frames if r["tp"] > 0]
    rates = swap_rates(confusion, cbd_id=CBD_ID, cd_id=CD_ID)

    return {
        "n_images": len(subset),
        "n_cases": len({s.get("case_id") for s in subset}),
        "cases": sorted({s.get("case_id") for s in subset}),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "eight_class_per_image_mean": per_class,
        "eight_class_swap_rates": rates,
        "merged_bile_duct": {
            "per_image_mean_iou": nanmean(r["iou_bile_duct"] for r in per_image),
            "per_image_mean_dice": nanmean(r["dice_bile_duct"] for r in per_image),
            "per_image_mean_recall": nanmean(r["recall_bile_duct"] for r in per_image),
            "per_image_mean_precision": nanmean(r["precision_bile_duct"] for r in per_image),
            "pixel_pooled": pooled,
            "frames_with_gt_bile_duct": len(gt_frames),
            "frames_with_any_overlap": len(detected),
            "frame_detection_recall": (len(detected) / len(gt_frames)) if gt_frames else None,
        },
        "per_image": per_image,
    }


def main() -> None:
    out_dir = ROOT / "experiments" / "train_test"
    modality = analyze_manifest(ROOT / "data" / "processed" / "manifest.csv")
    (out_dir / "modality_counts.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")
    print("=== modality ===")
    print(json.dumps({k: modality[k] for k in modality if k != "unlabeled_sample_ids_by_case"}, indent=2))
    print("Wrote", out_dir / "modality_counts.json", flush=True)

    merged = evaluate_merged(
        ROOT / "experiments" / "train_test" / "deeplabv3_best.pth",
        ROOT / "experiments" / "train_test" / "split.json",
        ROOT / "data" / "processed",
    )
    slim = {k: v for k, v in merged.items() if k != "per_image"}
    (out_dir / "merged_bile_duct_test.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("=== merged bile duct ===")
    print(json.dumps(slim, indent=2))
    print("Wrote", out_dir / "merged_bile_duct_test.json")


if __name__ == "__main__":
    main()
