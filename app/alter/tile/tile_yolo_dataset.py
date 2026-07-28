from __future__ import annotations

from pathlib import Path

from PIL import Image
from tabulate import tabulate

from app.utils.image_files import find_matching_image
from app.utils.tiling import clip_box_to_tile
from app.utils.tiling import tile_origins
from app.utils.yaml_config import find_yaml_file
from app.utils.yaml_config import load_class_names

SPLIT_NAMES = ["train", "valid", "val", "test"]
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 0.2
DEFAULT_MIN_VISIBILITY = 0.3


class YoloDatasetTiler:
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
        self.class_names = self._load_class_names()
        self.split_dirs = self._discover_splits()

    def _validate_params(self):
        if self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0.0, 1.0)")
        if not 0.0 <= self.min_visibility <= 1.0:
            raise ValueError("min_visibility must be in [0.0, 1.0]")

    def _load_class_names(self) -> dict[int, str]:
        yaml_path = find_yaml_file(self.dataset_root)
        if yaml_path is None:
            raise FileNotFoundError(f"No YAML file found in {self.dataset_root}")
        class_names = load_class_names(yaml_path)
        if not class_names:
            raise ValueError(f"Could not parse class names from {yaml_path}")
        return class_names

    def _discover_splits(self) -> list[str]:
        found_splits = [split for split in SPLIT_NAMES if (self.dataset_root / split / "labels").is_dir()]
        if not found_splits:
            raise FileNotFoundError(f"No splits with a labels/ directory found in {self.dataset_root}")
        return found_splits

    def _load_boxes(self, label_file: Path, width: int, height: int) -> list[tuple[int, tuple]]:
        boxes = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
            except ValueError:
                continue
            box_x1 = (center_x - box_width / 2) * width
            box_y1 = (center_y - box_height / 2) * height
            box_x2 = (center_x + box_width / 2) * width
            box_y2 = (center_y + box_height / 2) * height
            boxes.append((class_id, (box_x1, box_y1, box_x2, box_y2)))
        return boxes

    @staticmethod
    def _to_yolo_line(class_id: int, local_box, tile_width: int, tile_height: int) -> str:
        local_x1, local_y1, local_x2, local_y2 = local_box
        center_x = ((local_x1 + local_x2) / 2) / tile_width
        center_y = ((local_y1 + local_y2) / 2) / tile_height
        width = (local_x2 - local_x1) / tile_width
        height = (local_y2 - local_y1) / tile_height
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

    @staticmethod
    def _save_image(image, suffix: str, destination: Path):
        if suffix.lower() in (".jpg", ".jpeg") and image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(destination)

    def _tile_image(self, image_path: Path, label_file: Path, out_images: Path, out_labels: Path) -> dict:
        tiles_written = 0
        annotations_written = 0

        with Image.open(image_path) as image:
            width, height = image.size
            boxes = self._load_boxes(label_file, width, height)

            tile_width = min(self.tile_size, width)
            tile_height = min(self.tile_size, height)
            stride_x = max(1, int(tile_width * (1 - self.overlap)))
            stride_y = max(1, int(tile_height * (1 - self.overlap)))

            for row, tile_y in enumerate(tile_origins(height, tile_height, stride_y)):
                for col, tile_x in enumerate(tile_origins(width, tile_width, stride_x)):
                    tile = (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height)

                    kept_lines = []
                    for class_id, box in boxes:
                        local_box = clip_box_to_tile(box, tile, self.min_visibility)
                        if local_box is None:
                            continue
                        kept_lines.append(self._to_yolo_line(class_id, local_box, tile_width, tile_height))

                    if not kept_lines and not self.keep_empty_tiles:
                        continue

                    tile_stem = f"{image_path.stem}_r{row}c{col}"
                    self._save_image(image.crop(tile), image_path.suffix, out_images / (tile_stem + image_path.suffix))
                    (out_labels / (tile_stem + ".txt")).write_text(
                        ("\n".join(kept_lines) + "\n") if kept_lines else ""
                    )

                    tiles_written += 1
                    annotations_written += len(kept_lines)

        return {"tiles": tiles_written, "annotations": annotations_written}

    def _write_data_yaml(self):
        sorted_names = [self.class_names[class_id] for class_id in sorted(self.class_names)]
        lines = [
            f"nc: {len(sorted_names)}",
            f"names: {sorted_names}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
        ]
        (self.output_directory / "data.yaml").write_text("\n".join(lines) + "\n")

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
        print(f"\n  Tile (YOLO): {self.dataset_root.name}")
        print(f"  Splits: {', '.join(self.split_dirs)}")
        print(f"  Tile:   {self.tile_size}px, overlap {self.overlap}, min visibility {self.min_visibility}\n")

        totals = {"source_images": 0, "tiles": 0, "annotations": 0}
        for split in self.split_dirs:
            images_dir = self.dataset_root / split / "images"
            labels_dir = self.dataset_root / split / "labels"
            out_images = self.output_directory / split / "images"
            out_labels = self.output_directory / split / "labels"
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            for label_file in sorted(labels_dir.glob("*.txt")):
                if label_file.name == "classes.txt":
                    continue
                image_path = find_matching_image(images_dir, label_file.stem)
                if image_path is None:
                    continue
                result = self._tile_image(image_path, label_file, out_images, out_labels)
                totals["source_images"] += 1
                totals["tiles"] += result["tiles"]
                totals["annotations"] += result["annotations"]

        self._write_data_yaml()
        self._print_report(totals)
