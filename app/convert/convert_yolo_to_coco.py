from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

from app.utils.coco_files import ANNOTATIONS_FILENAME
from app.utils.coco_files import save_coco
from app.utils.image_files import find_matching_image
from app.utils.yaml_config import load_yaml_config
from app.utils.yaml_config import normalize_class_names


EXPECTED_SPLITS = ["train", "valid", "test"]
YOLO_DETECTION_FIELDS_COUNT = 5


class YOLOtoCOCOConverter:
    def __init__(self, dataset_directory):
        self.dataset_directory = Path(dataset_directory)

    def _load_class_names(self):
        yaml_path = self.dataset_directory / "data.yaml"
        if not yaml_path.exists():
            print(f"Error: data.yaml not found in {self.dataset_directory}")
            sys.exit(1)

        raw_names = load_yaml_config(yaml_path).get("names", {})
        if isinstance(raw_names, (list, dict)):
            return normalize_class_names(raw_names)

        print(f"Error: unexpected 'names' format in data.yaml: {type(raw_names)}")
        sys.exit(1)

    def _get_image_dimensions(self, image_path):
        with Image.open(image_path) as image:
            return image.width, image.height

    def _parse_yolo_label_line(self, line, image_width, image_height):
        parts = line.strip().split()
        if len(parts) != YOLO_DETECTION_FIELDS_COUNT:
            return None

        class_id = int(parts[0])
        center_x_normalized = float(parts[1])
        center_y_normalized = float(parts[2])
        box_width_normalized = float(parts[3])
        box_height_normalized = float(parts[4])

        box_width_pixels = box_width_normalized * image_width
        box_height_pixels = box_height_normalized * image_height
        top_left_x = (center_x_normalized * image_width) - (box_width_pixels / 2)
        top_left_y = (center_y_normalized * image_height) - (box_height_pixels / 2)

        return {
            "class_id": class_id,
            "bbox": [
                round(top_left_x, 2),
                round(top_left_y, 2),
                round(box_width_pixels, 2),
                round(box_height_pixels, 2),
            ],
            "area": round(box_width_pixels * box_height_pixels, 2),
        }

    def _convert_split(self, split_directory: Path, class_names, output_directory: Path):
        images_directory = split_directory / "images"
        labels_directory = split_directory / "labels"

        if not images_directory.is_dir() or not labels_directory.is_dir():
            print(f"  Skipping: missing images/ or labels/ in {split_directory}")
            return None

        label_files = sorted(labels_directory.glob("*.txt"))
        if not label_files:
            print(f"  No label files found in {labels_directory}")
            return None

        output_directory.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        used_class_ids = set()
        annotation_id = 1
        image_id = 0
        skipped_no_image = 0
        skipped_lines = 0

        for label_file in label_files:
            source_image_path = find_matching_image(images_directory, label_file.stem)

            if source_image_path is None:
                skipped_no_image += 1
                continue

            image_id += 1
            image_width, image_height = self._get_image_dimensions(source_image_path)
            image_filename = source_image_path.name

            destination_image_path = output_directory / image_filename
            if not destination_image_path.exists():
                shutil.copy2(source_image_path, destination_image_path)

            coco_images.append({
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height,
            })

            for line in label_file.read_text().splitlines():
                if not line.strip():
                    continue

                parsed = self._parse_yolo_label_line(line, image_width, image_height)
                if parsed is None:
                    skipped_lines += 1
                    continue

                used_class_ids.add(parsed["class_id"])
                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": parsed["class_id"],
                    "bbox": parsed["bbox"],
                    "area": parsed["area"],
                    "iscrowd": 0,
                })
                annotation_id += 1

        if skipped_no_image:
            print(f"  Warning: {skipped_no_image} label files had no matching image")
        if skipped_lines:
            print(f"  Warning: {skipped_lines} malformed or non-detection lines skipped")

        categories = []
        for class_id in sorted(used_class_ids):
            name = class_names.get(class_id, f"class_{class_id}")
            categories.append({"id": class_id, "name": name, "supercategory": "none"})

        return {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }

    def convert(self):
        output_root = self.dataset_directory.parent / (self.dataset_directory.name + "_coco")

        class_names = self._load_class_names()
        print(f"Loaded {len(class_names)} classes from data.yaml")

        splits_converted = 0

        for split_name in EXPECTED_SPLITS:
            split_input_directory = self.dataset_directory / split_name
            if not split_input_directory.exists():
                continue

            split_output_directory = output_root / split_name
            print(f"Processing: {split_name}")

            coco_data = self._convert_split(
                split_input_directory, class_names, split_output_directory
            )

            if coco_data is None:
                continue

            annotations_path = split_output_directory / ANNOTATIONS_FILENAME
            save_coco(annotations_path, coco_data)

            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
            print(f"  Categories: {[c['name'] for c in coco_data['categories']]}")
            print(f"  Saved: {annotations_path}")
            splits_converted += 1

        if splits_converted == 0:
            print("No valid splits found in the dataset directory")
            sys.exit(1)

        print(f"\nDone. Converted {splits_converted} splits to: {output_root}")
