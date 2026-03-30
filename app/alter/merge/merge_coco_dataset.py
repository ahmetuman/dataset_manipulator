from __future__ import annotations

import json
import os
import shutil
import sys

SPLIT_NAMES = ["train", "valid", "test"]
ANNOTATIONS_FILENAME = "_annotations.coco.json"
PARENT_CATEGORY_PREFIX = "Parent"


class CocoDatasetMerger:
    def __init__(self, datasets_root_directory, output_directory):
        self.root_directory = datasets_root_directory
        self.output_directory = output_directory
        self.dataset_folders = []
        self.unified_category_map = {}

    def _find_dataset_folders(self):
        folders = []
        for entry in sorted(os.listdir(self.root_directory)):
            entry_path = os.path.join(self.root_directory, entry)
            if not os.path.isdir(entry_path):
                continue
            for split_name in SPLIT_NAMES:
                annotations_path = os.path.join(entry_path, split_name, ANNOTATIONS_FILENAME)
                if os.path.exists(annotations_path):
                    folders.append(entry_path)
                    break
        self.dataset_folders = folders
        return folders

    def _build_unified_category_map(self):
        all_category_names = set()
        for folder in self.dataset_folders:
            for split_name in SPLIT_NAMES:
                annotations_path = os.path.join(folder, split_name, ANNOTATIONS_FILENAME)
                if not os.path.exists(annotations_path):
                    continue
                with open(annotations_path) as file:
                    data = json.load(file)
                for category in data["categories"]:
                    if PARENT_CATEGORY_PREFIX not in category["name"]:
                        all_category_names.add(category["name"])

        self.unified_category_map = {
            name: index
            for index, name in enumerate(sorted(all_category_names), start=1)
        }
        return self.unified_category_map

    def _merge_split(self, split_name):
        split_output_directory = os.path.join(self.output_directory, split_name)
        os.makedirs(split_output_directory, exist_ok=True)

        merged_images = []
        merged_annotations = []
        image_id_counter = 1
        annotation_id_counter = 1
        total_files_copied = 0
        skipped_parent_annotations = 0

        for folder in self.dataset_folders:
            split_directory = os.path.join(folder, split_name)
            annotations_path = os.path.join(split_directory, ANNOTATIONS_FILENAME)
            if not os.path.exists(annotations_path):
                continue

            with open(annotations_path) as file:
                data = json.load(file)

            dataset_name = os.path.basename(folder)

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

                file_extension = os.path.splitext(image["file_name"])[1]
                new_filename = f"{dataset_name}_{split_name}_{new_image_id}{file_extension}"

                source_path = os.path.join(split_directory, image["file_name"])
                destination_path = os.path.join(split_output_directory, new_filename)

                if os.path.exists(source_path):
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

        output_path = os.path.join(split_output_directory, ANNOTATIONS_FILENAME)
        with open(output_path, "w") as file:
            json.dump(coco_output, file, indent=2)

        return coco_output, total_files_copied, skipped_parent_annotations

    def merge(self):
        self._find_dataset_folders()
        if not self.dataset_folders:
            print("No datasets with COCO annotations found")
            sys.exit(1)

        print(f"Found {len(self.dataset_folders)} datasets: ")
        for folder in self.dataset_folders:
            print(f"  {os.path.basename(folder)}")

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


if __name__ == "__main__":
    merger = CocoDatasetMerger(root_directory=sys.argv[1], output_directory=sys.argv[2])
    merger.merge()
