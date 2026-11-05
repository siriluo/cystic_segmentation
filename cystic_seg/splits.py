"""Train / validation / test splits.

Default is a patient-level split so frames from one case never leak across sets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from cystic_seg.dataset import case_id_from_path, train_val_split_samples

SplitBundle = Tuple[List[dict], List[dict], List[dict]]


def case_id_of(sample: dict) -> str:
    return sample.get("case_id") or case_id_from_path(sample["image_path"])


def _partition_cases(
    cases: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    cases = list(cases)
    rng = random.Random(seed)
    rng.shuffle(cases)
    n = len(cases)
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    if n_test + n_val >= n:
        n_test = max(1, n // 5) if test_ratio > 0 else 0
        n_val = max(1, n // 5) if val_ratio > 0 else 0
        if n_test + n_val >= n:
            n_test = 1 if test_ratio > 0 else 0
            n_val = 1 if val_ratio > 0 and n > n_test + 1 else 0
    test_cases = cases[:n_test]
    val_cases = cases[n_test : n_test + n_val]
    train_cases = cases[n_test + n_val :]
    if not train_cases:
        raise ValueError("Train split is empty; lower --val-ratio / --test-ratio.")
    return train_cases, val_cases, test_cases


def train_val_test_split_by_case(
    samples: Sequence[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitBundle:
    cases = sorted({case_id_of(s) for s in samples})
    train_cases, val_cases, test_cases = _partition_cases(cases, val_ratio, test_ratio, seed)
    train_set, val_set, test_set = set(train_cases), set(val_cases), set(test_cases)
    train = [s for s in samples if case_id_of(s) in train_set]
    val = [s for s in samples if case_id_of(s) in val_set]
    test = [s for s in samples if case_id_of(s) in test_set]
    return train, val, test


def train_val_test_split_by_sample(
    samples: Sequence[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitBundle:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(indices)
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    test_idx = set(indices[:n_test])
    val_idx = set(indices[n_test : n_test + n_val])
    train_idx = set(indices[n_test + n_val :])
    train = [samples[i] for i in sorted(train_idx)]
    val = [samples[i] for i in sorted(val_idx)]
    test = [samples[i] for i in sorted(test_idx)]
    if not train:
        train, rest = train_val_split_samples(list(samples), val_ratio=val_ratio + test_ratio, seed=seed)
        val, test = train_val_split_samples(rest, val_ratio=0.5, seed=seed)
    return train, val, test


def make_split_info(
    samples: Sequence[dict],
    unit: str = "case",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    if unit == "case":
        train, val, test = train_val_test_split_by_case(samples, val_ratio, test_ratio, seed)
    elif unit == "sample":
        train, val, test = train_val_test_split_by_sample(samples, val_ratio, test_ratio, seed)
    else:
        raise ValueError(f"Unknown split unit: {unit}")
    return {
        "unit": unit,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_cases": sorted({case_id_of(s) for s in train}),
        "val_cases": sorted({case_id_of(s) for s in val}),
        "test_cases": sorted({case_id_of(s) for s in test}),
        "train_images": [s["image_path"] for s in train],
        "val_images": [s["image_path"] for s in val],
        "test_images": [s["image_path"] for s in test],
    }


def save_split(info: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def samples_for_subset(samples: Sequence[dict], info: Dict[str, Any], subset: str) -> List[dict]:
    key = f"{subset}_images"
    if key in info and info[key]:
        wanted = set(info[key])
        chosen = [s for s in samples if s["image_path"] in wanted]
        if chosen:
            return chosen
    cases_key = f"{subset}_cases"
    if cases_key in info:
        wanted_cases = set(info[cases_key])
        return [s for s in samples if case_id_of(s) in wanted_cases]
    raise KeyError(f"Split file has no {subset} assignment")
