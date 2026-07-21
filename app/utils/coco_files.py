from __future__ import annotations

import json
from pathlib import Path

ANNOTATIONS_FILENAME = "_annotations.coco.json"


def find_annotation_file(split_dir: Path) -> Path | None:
    candidates = list(split_dir.glob(f"*{ANNOTATIONS_FILENAME}"))
    return candidates[0] if candidates else None


def load_coco(annotation_path: Path) -> dict:
    with open(annotation_path) as file:
        return json.load(file)


def save_coco(annotation_path: Path, data: dict, indent: int = 2, ensure_ascii: bool = True):
    with open(annotation_path, "w") as file:
        json.dump(data, file, indent=indent, ensure_ascii=ensure_ascii)
