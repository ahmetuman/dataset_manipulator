from __future__ import annotations

from pathlib import Path

import yaml


def find_yaml_file(dataset_root: Path) -> Path | None:
    candidates = list(dataset_root.glob("*.yaml")) + list(dataset_root.glob("*.yml"))
    return candidates[0] if candidates else None


def load_yaml_config(yaml_path: Path) -> dict:
    with open(yaml_path) as file:
        return yaml.safe_load(file) or {}


def normalize_class_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: name for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(key): value for key, value in raw_names.items()}
    return {}


def load_class_names(yaml_path: Path) -> dict[int, str]:
    config = load_yaml_config(yaml_path)
    return normalize_class_names(config.get("names", {}))
