from __future__ import annotations

from pathlib import Path

from PIL import Image
from tabulate import tabulate

from app.utils.coco_files import ANNOTATIONS_FILENAME
from app.utils.coco_files import find_annotation_file
from app.utils.coco_files import load_coco
from app.utils.coco_files import save_coco
from app.utils.tiling import clip_box_to_tile
from app.utils.tiling import tile_origins

SPLIT_NAMES = ["train", "valid", "val", "test"]
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 0.2
DEFAULT_MIN_VISIBILITY = 0.3


class CocoDatasetTiler:
    def __init__(self, dataset_root: str, output_directory: str = "tiled_dataset",
                 tile_size: int = DEFAULT_TILE_SIZE, overlap: float = DEFAULT_OVERLAP,
                 min_visibility: float = DEFAULT_MIN_VISIBILITY, keep_empty_tiles: bool = False):
        self.dataset_root = Path(dataset_root)
        self.output_directory = Path(output_directory)
        self.tile_size = int(tile_size)
        self.overlap = float(overlap)
        self.min_visibility = float(min_visibility)
        self.keep_empty_tiles = keep_empty_tiles
        self._validate_params()
        self.split_dirs = self._discover_splits()

    def _validate_params(self):
        if self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0.0, 1.0)")
        if not 0.0 <= self.min_visibility <= 1.0:
            raise ValueError("min_visibility must be in [0.0, 1.0]")

    def _discover_splits(self) -> list[str]:
        found_splits = []
        for split in SPLIT_NAMES:
            split_dir = self.dataset_root / split
            if split_dir.is_dir() and find_annotation_file(split_dir):
                found_splits.append(split)
        if not found_splits:
            raise FileNotFoundError(f"No splits with annotation files found in {self.dataset_root}")
        return found_splits

    @staticmethod
    def _group_annotations_by_image(data: dict) -> dict[int, list[dict]]:
        grouped = {}
        for annotation in data.get("annotations", []):
            grouped.setdefault(annotation.get("image_id"), []).append(annotation)
        return grouped

    def _image_boxes(self, annotations: list[dict]) -> list[tuple[int, tuple]]:
        boxes = []
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                x, y, width, height = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            if width <= 0 or height <= 0:
                continue
            boxes.append((annotation.get("category_id"), (x, y, x + width, y + height)))
        return boxes

    @staticmethod
    def _save_image(image, suffix: str, destination: Path):
        if suffix.lower() in (".jpg", ".jpeg") and image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(destination)

    def _tile_split(self, split: str, totals: dict):
        split_dir = self.dataset_root / split
        data = load_coco(find_annotation_file(split_dir))
        annotations_by_image = self._group_annotations_by_image(data)

        output_split_dir = self.output_directory / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        new_images = []
        new_annotations = []
        next_image_id = 1
        next_annotation_id = 1

        for image in data.get("images", []):
            file_name = image.get("file_name")
            if not file_name:
                continue
            image_path = split_dir / file_name
            if not image_path.exists():
                continue

            totals["source_images"] += 1
            boxes = self._image_boxes(annotations_by_image.get(image.get("id"), []))

            with Image.open(image_path) as pil_image:
                width, height = pil_image.size
                tile_width = min(self.tile_size, width)
                tile_height = min(self.tile_size, height)
                stride_x = max(1, int(tile_width * (1 - self.overlap)))
                stride_y = max(1, int(tile_height * (1 - self.overlap)))

                for row, tile_y in enumerate(tile_origins(height, tile_height, stride_y)):
                    for col, tile_x in enumerate(tile_origins(width, tile_width, stride_x)):
                        tile = (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height)

                        kept = []
                        for category_id, box in boxes:
                            local_box = clip_box_to_tile(box, tile, self.min_visibility)
                            if local_box is None:
                                continue
                            local_x1, local_y1, local_x2, local_y2 = local_box
                            local_width = local_x2 - local_x1
                            local_height = local_y2 - local_y1
                            kept.append({
                                "category_id": category_id,
                                "bbox": [round(local_x1, 2), round(local_y1, 2),
                                         round(local_width, 2), round(local_height, 2)],
                                "area": round(local_width * local_height, 2),
                                "iscrowd": 0,
                            })

                        if not kept and not self.keep_empty_tiles:
                            continue

                        tile_stem = f"{Path(file_name).stem}_r{row}c{col}"
                        tile_name = tile_stem + image_path.suffix
                        self._save_image(pil_image.crop(tile), image_path.suffix, output_split_dir / tile_name)

                        image_id = next_image_id
                        next_image_id += 1
                        new_images.append({
                            "id": image_id,
                            "file_name": tile_name,
                            "width": tile_width,
                            "height": tile_height,
                        })

                        for annotation in kept:
                            annotation["id"] = next_annotation_id
                            annotation["image_id"] = image_id
                            next_annotation_id += 1
                            new_annotations.append(annotation)
                            totals["annotations"] += 1

                        totals["tiles"] += 1

        save_coco(output_split_dir / ANNOTATIONS_FILENAME, {
            "images": new_images,
            "annotations": new_annotations,
            "categories": data.get("categories", []),
        })

    def _print_report(self, totals: dict):
        rows = [
            ["Source images", totals["source_images"]],
            ["Tiles written", totals["tiles"]],
            ["Annotations written", totals["annotations"]],
            ["Tile size", self.tile_size],
            ["Overlap", self.overlap],
            ["Min visibility", self.min_visibility],
        ]
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))
        print(f"\n  Tiled dataset saved to: {self.output_directory}")

    def tile(self):
        print(f"\n  Tile (COCO): {self.dataset_root.name}")
        print(f"  Splits: {', '.join(self.split_dirs)}")
        print(f"  Tile:   {self.tile_size}px, overlap {self.overlap}, min visibility {self.min_visibility}\n")

        totals = {"source_images": 0, "tiles": 0, "annotations": 0}
        for split in self.split_dirs:
            self._tile_split(split, totals)

        self._print_report(totals)
