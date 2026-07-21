from __future__ import annotations

import shutil
import sys
from pathlib import Path

from app.utils.coco_files import ANNOTATIONS_FILENAME
from app.utils.coco_files import load_coco
from app.utils.coco_files import save_coco

SPLIT_NAMES = ["train", "valid", "test"]
PARENT_CATEGORY_PREFIX = "Parent"


class CocoDatasetMerger:
    def __init__(self, datasets_root_directory, output_directory):
        self.datasets_root_directory = Path(datasets_root_directory)
        self.output_directory = Path(output_directory)
        self.dataset_folders = []
        self.unified_category_map = {}

    def _find_dataset_folders(self) -> list[Path]:
        folders = []
        for entry in sorted(self.datasets_root_directory.iterdir()):
            if not entry.is_dir():
                continue
            for split_name in SPLIT_NAMES:
                if (entry / split_name / ANNOTATIONS_FILENAME).exists():
                    folders.append(entry)
                    break
        self.dataset_folders = folders
        return folders

    def _build_unified_category_map(self):
        all_category_names = set()
        for folder in self.dataset_folders:
            for split_name in SPLIT_NAMES:
                annotations_path = folder / split_name / ANNOTATIONS_FILENAME
                if not annotations_path.exists():
                    continue
                data = load_coco(annotations_path)
                for category in data["categories"]:
                    if PARENT_CATEGORY_PREFIX not in category["name"]:
                        all_category_names.add(category["name"])

        self.unified_category_map = {
            name: index
            for index, name in enumerate(sorted(all_category_names), start=1)
        }
        return self.unified_category_map

    def _merge_split(self, split_name):
        split_output_directory = self.output_directory / split_name
        split_output_directory.mkdir(parents=True, exist_ok=True)

        merged_images = []
        merged_annotations = []
        image_id_counter = 1
        annotation_id_counter = 1
        total_files_copied = 0
        skipped_parent_annotations = 0

        for folder in self.dataset_folders:
            split_directory = folder / split_name
            annotations_path = split_directory / ANNOTATIONS_FILENAME
            if not annotations_path.exists():
                continue

            data = load_coco(annotations_path)
            dataset_name = folder.name

            old_id_to_unified_category_id = {}
            parent_category_ids = set()
            for category in data["categories"]:
                if category["name"].startswith("Parent_"):
                    parent_category_ids.add(category["id"])
                elif category["name"] in self.unified_category_map:
                    old_id_to_unified_category_id[category["id"]] = self.unified_category_map[category["name"]]

            old_to_new_image_id = {}
            for image in data["images"]:
                old_image_id = image["id"]
                new_image_id = image_id_counter
                image_id_counter += 1
                old_to_new_image_id[old_image_id] = new_image_id

                file_extension = Path(image["file_name"]).suffix
                new_filename = f"{dataset_name}_{split_name}_{new_image_id}{file_extension}"

                source_path = split_directory / image["file_name"]
                destination_path = split_output_directory / new_filename

                if source_path.exists():
                    shutil.copy2(source_path, destination_path)
                    total_files_copied += 1

                merged_images.append({
                    "id": new_image_id,
                    "file_name": new_filename,
                    "width": image["width"],
                    "height": image["height"],
                })

            for annotation in data["annotations"]:
                if annotation["category_id"] in parent_category_ids:
                    skipped_parent_annotations += 1
                    continue
                if annotation["category_id"] not in old_id_to_unified_category_id:
                    continue
                if annotation["image_id"] not in old_to_new_image_id:
                    continue

                merged_annotations.append({
                    "id": annotation_id_counter,
                    "image_id": old_to_new_image_id[annotation["image_id"]],
                    "category_id": old_id_to_unified_category_id[annotation["category_id"]],
                    "bbox": annotation["bbox"],
                    "area": annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]),
                    "iscrowd": annotation.get("iscrowd", 0),
                })
                annotation_id_counter += 1

        if not merged_images:
            return None, 0, 0

        category_list = [
            {"id": category_id, "name": name, "supercategory": "none"}
            for name, category_id in sorted(self.unified_category_map.items(), key=lambda item: item[1])
        ]

        coco_output = {
            "images": merged_images,
            "annotations": merged_annotations,
            "categories": category_list,
        }

        save_coco(split_output_directory / ANNOTATIONS_FILENAME, coco_output)

        return coco_output, total_files_copied, skipped_parent_annotations

    def merge(self):
        self._find_dataset_folders()
        if not self.dataset_folders:
            print("No datasets with COCO annotations found")
            sys.exit(1)

        print(f"Found {len(self.dataset_folders)} datasets: ")
        for folder in self.dataset_folders:
            print(f"  {folder.name}")

        self._build_unified_category_map()
        print(f"\nUnified categories ({len(self.unified_category_map)}): ")
        for name, category_id in sorted(self.unified_category_map.items(), key=lambda item: item[1]):
            print(f"  {category_id}: {name}")

        print()
        for split_name in SPLIT_NAMES:
            coco_output, files_copied, skipped_parents = self._merge_split(split_name)
            if coco_output is None:
                print(f"[{split_name}] No data found, skipping")
                continue
            print(
                f"[{split_name}] {len(coco_output['images'])} images, "
                f"{len(coco_output['annotations'])} annotations, "
                f"{files_copied} files copied, "
                f"{skipped_parents} parent annotations stripped"
            )

        print(f"\nMerged dataset saved to: {self.output_directory}")
