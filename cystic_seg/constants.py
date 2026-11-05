"""Shared class definitions and preprocessing constants.

Annotation files use VIA-style region attributes. Several labels were entered
with typos or inconsistent casing; LABEL_ALIASES maps those strings onto the
canonical names in LABEL_MAP.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Background is class 0. Landmark classes are 1..7.
NUM_CLASSES = 8
IMAGE_SIZE: Tuple[int, int] = (512, 512)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LABEL_MAP: Dict[str, int] = {
    "Gallbladder": 1,
    "Commonbileduct": 2,
    "Cysticduct": 3,
    "Instrument": 4,
    "liveredge": 5,
    "Cysticartery": 6,
    "LuviereSulcus": 7,
}

LABEL_ALIASES: Dict[str, str] = {
    "Gallbladder": "Gallbladder",
    "Gallbradder": "Gallbladder",
    "gallbladder": "Gallbladder",
    "Commonbileduct": "Commonbileduct",
    "CommonBileDuct": "Commonbileduct",
    "commonbileduct": "Commonbileduct",
    "Cysticduct": "Cysticduct",
    "CysticDuct": "Cysticduct",
    "cysticduct": "Cysticduct",
    "Instrument": "Instrument",
    "instrument": "Instrument",
    "liveredge": "liveredge",
    "Liveredge": "liveredge",
    "LiverEdge": "liveredge",
    "Cysticartery": "Cysticartery",
    "CysticArtery": "Cysticartery",
    "cysticartery": "Cysticartery",
    "LuviereSulcus": "LuviereSulcus",
    "RouviereSulcus": "LuviereSulcus",
    "RouvieresSulcus": "LuviereSulcus",
    "liver edge": "liveredge",
    "Liver edge": "liveredge",
    "Cystic duct": "Cysticduct",
    "Cystic artery": "Cysticartery",
    "Common bile duct": "Commonbileduct",
    "Luviere Sulcus": "LuviereSulcus",
    "Rouviere Sulcus": "LuviereSulcus",
    "Rouvière's sulcus": "LuviereSulcus",
}

_NORMALIZED_ALIASES: Dict[str, str] = {
    re.sub(r"[^a-z0-9]+", "", k.lower()): v for k, v in LABEL_ALIASES.items()
}

# When polygons overlap, lower numbers win (written first).
LABEL_PRIORITY: Dict[str, int] = {
    "Commonbileduct": 0,
    "Cysticduct": 1,
    "Cysticartery": 2,
}

CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "Gallbladder",
    2: "Common bile duct",
    3: "Cystic duct",
    4: "Instrument",
    5: "Liver edge",
    6: "Cystic artery",
    7: "Rouvière's sulcus",
}

DISPLAY_NAMES: List[str] = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]


def canonicalize_label(raw_label: Optional[str]) -> Optional[str]:
    """Return the canonical LABEL_MAP key, or None if the label is unknown."""
    if raw_label is None:
        return None
    text = unicodedata.normalize("NFKC", str(raw_label)).replace("\u3000", " ").strip()
    if not text:
        return None
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    collapsed = re.sub(r"[^a-z0-9]+", "", text.lower())
    return _NORMALIZED_ALIASES.get(collapsed)
