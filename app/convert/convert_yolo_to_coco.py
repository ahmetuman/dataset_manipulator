from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_SPLITS = ["train", "valid", "test"]
YOLO_DETECTION_FIELDS_COUNT = 5


class YOLOtoCOCOConverter:
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory

    def _load_class_names(self):
        yaml_path = os.path.join(self.dataset_directory, "data.yaml")
        if not os.path.exists(yaml_path):
            print(f"Error: data.yaml not found in {self.dataset_directory}")  # noqa E713
            sys.exit(1)

        with open(yaml_path) as file:
            config = yaml.safe_load(file)

        raw_names = config.get("names", {})

        if isinstance(raw_names, list):
            return {index: name for index, name in enumerate(raw_names)}
        if isinstance(raw_names, dict):
            return {int(key): value for key, value in raw_names.items()}

        print(f"Error: unexpected 'names' format in data.yaml: {type(raw_names)}")
        sys.exit(1)

    def _get_image_dimensions(self, image_path):
        with Image.open(image_path) as image:
            return image.width, image.height

    def _find_matching_image(self, directory, file_stem):
        for extension in SUPPORTED_IMAGE_EXTENSIONS:
            candidate_path = os.path.join(directory, file_stem + extension)
            if os.path.exists(candidate_path):
                return candidate_path
        return None

    def _parse_yolo_label_line(self, line, image_width, image_height):
        parts = line.strip().split()
        if len(parts) < YOLO_DETECTION_FIELDS_COUNT:
            return None

        if len(parts) > YOLO_DETECTION_FIELDS_COUNT:
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

    def _convert_split(self, split_directory, class_names, output_directory):
        images_directory = os.path.join(split_directory, "images")
        labels_directory = os.path.join(split_directory, "labels")

        if not os.path.isdir(images_directory) or not os.path.isdir(labels_directory):
            print(f"  Skipping: missing images/ or labels/ in {split_directory}")
            return None

        label_files = sorted(
            filename
            for filename in os.listdir(labels_directory)
            if filename.endswith(".txt")
        )

        if not label_files:
            print(f"  No label files found in {labels_directory}")
            return None

        os.makedirs(output_directory, exist_ok=True)

        coco_images = []
        coco_annotations = []
        used_class_ids = set()
        annotation_id = 1
        image_id = 0
        skipped_no_image = 0
        skipped_lines = 0

        for label_filename in label_files:
            file_stem = Path(label_filename).stem
            source_image_path = self._find_matching_image(images_directory, file_stem)

            if source_image_path is None:
                skipped_no_image += 1
                continue

            image_id += 1
            image_width, image_height = self._get_image_dimensions(source_image_path)
            image_filename = os.path.basename(source_image_path)

            destination_image_path = os.path.join(output_directory, image_filename)
            if not os.path.exists(destination_image_path):
                shutil.copy2(source_image_path, destination_image_path)

            coco_images.append({
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height,
            })

            label_path = os.path.join(labels_directory, label_filename)
            with open(label_path) as file:
                for line in file:
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
        output_root = self.dataset_directory + "_coco"

        class_names = self._load_class_names()
        print(f"Loaded {len(class_names)} classes from data.yaml")

        splits_converted = 0

        for split_name in EXPECTED_SPLITS:
            split_input_directory = os.path.join(self.dataset_directory, split_name)
            if not os.path.exists(split_input_directory):
                continue

            split_output_directory = os.path.join(output_root, split_name)
            print(f"Processing: {split_name}")

            coco_data = self._convert_split(
                split_input_directory, class_names, split_output_directory
            )

            if coco_data is None:
                continue

            annotations_path = os.path.join(split_output_directory, "_annotations.coco.json")
            with open(annotations_path, "w") as file:
                json.dump(coco_data, file, indent=2)

            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
            print(f"  Categories: {[c['name'] for c in coco_data['categories']]}")
            print(f"  Saved: {annotations_path}")
            splits_converted += 1

        if splits_converted == 0:
            print("No valid splits found in the dataset directory")
            sys.exit(1)

        print(f"\nDone. Converted {splits_converted} splits to: {output_root}")
