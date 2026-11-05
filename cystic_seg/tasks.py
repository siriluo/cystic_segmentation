"""Task-specific class remaps. Bile-duct focus trades other landmarks for CBD vs CD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from cystic_seg.constants import CLASS_NAMES, NUM_CLASSES

# Original ids: 0 bg, 1 GB, 2 CBD, 3 CD, 4 instrument, 5 liver, 6 artery, 7 Rouvière
# 4-class:      0 bg, 1 CBD, 2 CD, 3 other
BILE_DUCT_REMAP = np.array([0, 3, 1, 2, 3, 3, 3, 3], dtype=np.int64)
BILE_DUCT_NAMES: Dict[int, str] = {
    0: "background",
    1: "Common bile duct",
    2: "Cystic duct",
    3: "other",
}
BILE_DUCT_NUM_CLASSES = 4

# 3-class: 0 everything else, 1 CBD, 2 CD
DUCTS_ONLY_REMAP = np.array([0, 0, 1, 2, 0, 0, 0, 0], dtype=np.int64)
DUCTS_ONLY_NAMES: Dict[int, str] = {
    0: "background",
    1: "Common bile duct",
    2: "Cystic duct",
}
DUCTS_ONLY_NUM_CLASSES = 3

BILE_CBD_ID = 1
BILE_CD_ID = 2
ORIG_CBD_ID = 2
ORIG_CD_ID = 3


@dataclass(frozen=True)
class TaskSpec:
    num_classes: int
    class_names: Dict[int, str]
    remap: np.ndarray | None
    cbd_id: int
    cd_id: int
    duct_focus: bool


TASKS: Dict[str, TaskSpec] = {
    "landmarks": TaskSpec(NUM_CLASSES, CLASS_NAMES, None, ORIG_CBD_ID, ORIG_CD_ID, False),
    "bile-ducts": TaskSpec(
        BILE_DUCT_NUM_CLASSES, BILE_DUCT_NAMES, BILE_DUCT_REMAP, BILE_CBD_ID, BILE_CD_ID, True
    ),
    "ducts-only": TaskSpec(
        DUCTS_ONLY_NUM_CLASSES, DUCTS_ONLY_NAMES, DUCTS_ONLY_REMAP, BILE_CBD_ID, BILE_CD_ID, True
    ),
}


def get_task(task: str) -> TaskSpec:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from {sorted(TASKS)}")
    return TASKS[task]


def remap_mask_array(mask: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return mapping[np.asarray(mask, dtype=np.int64)]


def task_num_classes(task: str) -> int:
    return get_task(task).num_classes


def task_class_names(task: str) -> Dict[int, str]:
    return get_task(task).class_names


def wrap_for_task(dataset, task: str):
    spec = get_task(task)
    if spec.remap is None:
        return dataset
    from cystic_seg.dataset import RemapDataset

    return RemapDataset(dataset, spec.remap)


def duct_mean_dice(metrics: dict, cbd_id: int = BILE_CBD_ID, cd_id: int = BILE_CD_ID) -> Optional[float]:
    """Mean Dice of CBD and CD, ignoring classes that are absent in the split."""
    by_id = {row["id"]: row.get("dice") for row in metrics.get("per_class", [])}
    vals = [by_id.get(cbd_id), by_id.get(cd_id)]
    present = [v for v in vals if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))
