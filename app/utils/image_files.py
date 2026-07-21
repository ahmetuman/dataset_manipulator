from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def find_matching_image(directory: Path, file_stem: str) -> Path | None:
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = directory / (file_stem + extension)
        if candidate.exists():
            return candidate
    return None
