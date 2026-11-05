#!/usr/bin/env python3
"""Turn the messy case-picture zip into a patient/sample training tree.

Output layout::

    data/processed/Case01/after_cut_no_icg/image.jpg
    data/processed/Case01/after_cut_no_icg/annotation.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cystic_seg.constants import LABEL_MAP, canonicalize_label
from cystic_seg.dataset import CASE_RE, IMAGE_EXTS, case_id_from_path, is_overlay_name, normalize_name


def sanitize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u3000", " ")
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._")
    return text or "sample"


def case_id_from_parts(parts) -> str:
    return case_id_from_path("/".join(str(p) for p in parts))


def read_via(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data, _ = json.JSONDecoder().raw_decode(text)
        return data if isinstance(data, dict) else None
    except OSError:
        return None


def region_labels(regions: dict) -> list[str]:
    labels = []
    if not isinstance(regions, dict):
        return labels
    for region in regions.values():
        if not isinstance(region, dict):
            continue
        raw = (region.get("region_attributes") or {}).get("label")
        canon = canonicalize_label(raw)
        if canon:
            labels.append(canon)
        elif raw:
            labels.append(f"UNKNOWN:{raw}")
    return labels


def file_digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in IMAGE_EXTS | {".txt", ".json"}:
                continue
            target = dest / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


def candidate_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_overlay_name(p.name)
    )


def match_image(folder: Path, wanted_name: str, images: list[Path]) -> Path | None:
    wanted = normalize_name(wanted_name)
    wanted_stem = normalize_name(Path(wanted_name).stem)
    by_name = {normalize_name(p.name): p for p in images}
    if wanted in by_name:
        return by_name[wanted]
    by_stem = {normalize_name(p.stem): p for p in images}
    if wanted_stem in by_stem:
        return by_stem[wanted_stem]
    # Unicode space / missing extension mismatches
    for p in images:
        if normalize_name(p.stem).replace(" ", "") == wanted_stem.replace(" ", ""):
            return p
    return None


def collect_pairs(source_roots: list[Path]) -> list[dict]:
    pairs = []
    unknown_labels = Counter()
    for root in source_roots:
        if not root.exists():
            continue
        for folder in [root, *root.rglob("*")]:
            if not folder.is_dir():
                continue
            anns = sorted(
                p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".json"}
            )
            if not anns:
                continue
            images = candidate_images(folder)
            for ann_path in anns:
                if ann_path.name.lower() == "label definition.txt":
                    continue
                data = read_via(ann_path)
                if not data:
                    pairs.append({"status": "bad_json", "ann": str(ann_path)})
                    continue
                records = []
                for key, rec in data.items():
                    if not isinstance(rec, dict) or not rec.get("regions"):
                        continue
                    records.append((str(key), rec))
                if not records:
                    pairs.append({"status": "no_regions", "ann": str(ann_path)})
                    continue
                for key, rec in records:
                    filename = rec.get("filename") or key
                    img = match_image(folder, filename, images)
                    if img is None and len(images) == 1 and len(records) == 1:
                        img = images[0]
                    if img is None:
                        pairs.append(
                            {
                                "status": "missing_image",
                                "ann": str(ann_path),
                                "wanted": filename,
                            }
                        )
                        continue
                    labels = region_labels(rec.get("regions") or {})
                    for lab in labels:
                        if lab.startswith("UNKNOWN:"):
                            unknown_labels[lab] += 1
                    known = [lab for lab in labels if lab in LABEL_MAP]
                    if not known:
                        pairs.append(
                            {
                                "status": "no_known_labels",
                                "ann": str(ann_path),
                                "image": str(img),
                                "labels": labels,
                            }
                        )
                        continue
                    case_id = case_id_from_parts((*img.parts, *ann_path.parts))
                    sample_id = sanitize(img.stem)
                    pairs.append(
                        {
                            "status": "ok",
                            "case_id": case_id,
                            "sample_id": sample_id,
                            "image": str(img),
                            "ann": str(ann_path),
                            "labels": known,
                            "via_key": key,
                            "record": rec,
                        }
                    )
    return pairs, unknown_labels


def export_pairs(pairs: list[dict], out_root: Path) -> tuple[int, list[dict]]:
    out_root.mkdir(parents=True, exist_ok=True)
    seen_hash = {}
    used_dirs = set()
    kept = []
    n = 0
    for pair in pairs:
        if pair.get("status") != "ok":
            continue
        digest = file_digest(Path(pair["image"]))
        if digest in seen_hash:
            pair["status"] = "duplicate"
            pair["duplicate_of"] = seen_hash[digest]
            continue
        case_id = pair["case_id"]
        sample_id = pair["sample_id"]
        dest_dir = out_root / case_id / sample_id
        suffix = 2
        while str(dest_dir) in used_dirs:
            dest_dir = out_root / case_id / f"{sample_id}_{suffix}"
            suffix += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_img = Path(pair["image"])
        dest_img = dest_dir / f"image{src_img.suffix.lower()}"
        shutil.copy2(src_img, dest_img)
        rec = pair["record"]
        payload = {dest_img.name: rec}
        (dest_dir / "annotation.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        seen_hash[digest] = str(dest_dir)
        used_dirs.add(str(dest_dir))
        n += 1
        kept.append(
            {
                "case_id": case_id,
                "sample_id": dest_dir.name,
                "image": str(dest_img),
                "n_labels": len(pair["labels"]),
                "labels": "|".join(sorted(set(pair["labels"]))),
                "source_image": pair["image"],
                "source_ann": pair["ann"],
            }
        )
    return n, kept


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize messy case folders into data/processed")
    parser.add_argument("--zip", type=str, default=str(ROOT / "AI project case pictures.zip"))
    parser.add_argument("--staging", type=str, default=str(ROOT / "data/raw_import"))
    parser.add_argument("--out", type=str, default=str(ROOT / "data/processed"))
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    staging = Path(args.staging)
    out_root = Path(args.out)

    if zip_path.exists() and not args.skip_extract:
        print(f"Extracting images/annotations from {zip_path} -> {staging}")
        n = extract_zip(zip_path, staging)
        print(f"Extracted {n} files")
    elif not zip_path.exists():
        print(f"Zip not found: {zip_path} (continuing with existing folders)")

    source_roots = [staging]
    data_root = ROOT / "data"
    for child in sorted(data_root.iterdir()) if data_root.exists() else []:
        if child.is_dir() and CASE_RE.search(child.name) and child.name not in {"processed", "raw_import"}:
            source_roots.append(child)

    print("Scanning:", *[str(p) for p in source_roots], sep="\n  ")
    pairs, unknown = collect_pairs(source_roots)
    status_counts = Counter(p["status"] for p in pairs)
    print("Pair status:", dict(status_counts))
    if unknown:
        print("Unknown labels:", dict(unknown))

    if out_root.exists():
        shutil.rmtree(out_root)
    n_kept, kept = export_pairs(pairs, out_root)
    write_manifest(kept, out_root / "manifest.csv")

    by_case = Counter(r["case_id"] for r in kept)
    print(f"Wrote {n_kept} unique samples under {out_root}")
    print(f"Patients: {len(by_case)}")
    print("Samples per case (min/median/max):", end=" ")
    counts = sorted(by_case.values())
    if counts:
        mid = counts[len(counts) // 2]
        print(f"{counts[0]}/{mid}/{counts[-1]}")
    skip_path = out_root / "skipped.json"
    skipped = [p for p in pairs if p.get("status") != "ok"]
    # drop bulky record objects
    slim = []
    for p in skipped:
        slim.append({k: v for k, v in p.items() if k != "record"})
    skip_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Skip log: {skip_path} ({len(slim)} entries)")


if __name__ == "__main__":
    main()
