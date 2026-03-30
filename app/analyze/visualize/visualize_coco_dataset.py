from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import cv2


class CocoDatasetVisualizer:
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    CATEGORY_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
        (200, 100, 50), (50, 200, 100), (100, 50, 200), (220, 220, 0),
    ]

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOUNDING_BOX_THICKNESS = 2

    KEY_QUIT_Q = ord("q")
    KEY_QUIT_ESC = 27
    KEY_PREVIOUS = ord("p")

    CANDIDATE_ANNOTATION_FILENAMES = [
        "_annotations.coco.json",
        "annotations.json",
        "instances.json",
    ]

    def __init__(self, dataset_directory, detailed_mode):
        self.dataset_directory = dataset_directory
        self.detailed_mode = detailed_mode

        self.splits = self._discover_available_splits()

    def _load_coco_annotations(self, annotation_file_path):
        with open(annotation_file_path) as annotation_file:
            coco_data = json.load(annotation_file)
        return coco_data

    def _build_category_mapping(self, coco_data):
        category_id_to_name = {}
        for category in coco_data.get("categories", []):
            category_id_to_name[category["id"]] = category["name"]
        return category_id_to_name

    def _build_image_id_to_filename_mapping(self, coco_data):
        image_id_to_filename = {}
        for image_info in coco_data.get("images", []):
            image_id_to_filename[image_info["id"]] = image_info["file_name"]
        return image_id_to_filename

    def _group_annotations_by_image(self, coco_data):
        annotations_by_image_id = defaultdict(list)
        for annotation in coco_data.get("annotations", []):
            annotations_by_image_id[annotation["image_id"]].append(annotation)
        return annotations_by_image_id

    def _find_annotation_file(self, split_folder_path):
        json_files = [
            filename for filename in os.listdir(split_folder_path)
            if filename.endswith(".json")
        ]
        if len(json_files) == 1:
            return os.path.join(split_folder_path, json_files[0])
        for candidate_name in self.CANDIDATE_ANNOTATION_FILENAMES:
            candidate_path = os.path.join(split_folder_path, candidate_name)
            if os.path.exists(candidate_path):
                return candidate_path
        if json_files:
            return os.path.join(split_folder_path, json_files[0])
        return None

    def _discover_available_splits(self):
        available_splits = []
        for split_name in ["train", "test", "valid", "data"]:
            split_path = os.path.join(self.dataset_directory, split_name)
            if not os.path.isdir(split_path):
                continue
            annotation_file_path = self._find_annotation_file(split_path)
            if annotation_file_path is None:
                print(f"Warning: '{split_name}' folder exists but no annotation JSON found, skipping.")
                continue
            coco_data = self._load_coco_annotations(annotation_file_path)
            image_id_to_filename = self._build_image_id_to_filename_mapping(coco_data)
            if image_id_to_filename:
                available_splits.append({
                    "name": split_name,
                    "split_path": split_path,
                    "annotation_file_path": annotation_file_path,
                    "coco_data": coco_data,
                    "image_id_to_filename": image_id_to_filename,
                })
        return available_splits

    def _get_color_for_category(self, category_id):
        return self.CATEGORY_COLORS[category_id % len(self.CATEGORY_COLORS)]

    def _draw_bounding_boxes_on_image(self, image, annotations_for_image, category_id_to_name):
        for annotation in annotations_for_image:
            bbox_x, bbox_y, bbox_width, bbox_height = (int(value) for value in annotation["bbox"])
            category_id = annotation["category_id"]
            category_name = category_id_to_name.get(category_id, f"unknown_{category_id}")
            color = self._get_color_for_category(category_id)

            cv2.rectangle(
                image,
                (bbox_x, bbox_y),
                (bbox_x + bbox_width, bbox_y + bbox_height),
                color,
                self.BOUNDING_BOX_THICKNESS,
            )

            label_text = f"{category_name} ({category_id})"
            text_size, _ = cv2.getTextSize(label_text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            text_background_top_left = (bbox_x, bbox_y - text_size[1] - 6)
            text_background_bottom_right = (bbox_x + text_size[0] + 4, bbox_y)

            cv2.rectangle(image, text_background_top_left, text_background_bottom_right, color, -1)
            cv2.putText(
                image,
                label_text,
                (bbox_x + 2, bbox_y - 4),
                self.FONT,
                self.FONT_SCALE,
                (0, 0, 0),
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return image

    def _print_color_legend(self, category_id_to_name):
        print("\nLabel Colors:")
        for category_id in sorted(category_id_to_name.keys()):
            blue, green, red = self._get_color_for_category(category_id)
            category_name = category_id_to_name[category_id]
            print(f"  {category_name} (id: {category_id}) -> RGB({red}, {green}, {blue})")
        print()

    def _collect_all_category_ids_across_splits(self):
        all_category_id_to_name = {}
        for split in self.splits:
            category_id_to_name = self._build_category_mapping(split["coco_data"])
            all_category_id_to_name.update(category_id_to_name)
        return all_category_id_to_name

    def _prompt_user_for_desired_category_ids(self, category_id_to_name):
        print("Available labels:")
        for category_id in sorted(category_id_to_name.keys()):
            print(f"  {category_id}: {category_id_to_name[category_id]}")

        user_input = input("\nEnter category IDs to display (comma-separated): ").strip()
        selected_category_ids = set()
        for token in user_input.split(","):
            token = token.strip()
            if token.isdigit() and int(token) in category_id_to_name:
                selected_category_ids.add(int(token))
            elif token:
                print(f"  Ignoring invalid category ID: '{token}'")

        if not selected_category_ids:
            print("No valid categories selected, showing all labels.")
            return None

        selected_names = [category_id_to_name[cid] for cid in sorted(selected_category_ids)]
        print(f"Filtering to: {', '.join(selected_names)}")
        return selected_category_ids

    def _filter_annotations_by_category_ids(self, annotations, desired_category_ids):
        if desired_category_ids is None:
            return annotations
        return [a for a in annotations if a["category_id"] in desired_category_ids]

    def _filter_image_ids_by_category_ids(self, image_ids, annotations_by_image_id, desired_category_ids):
        if desired_category_ids is None:
            return image_ids
        return [
            image_id for image_id in image_ids
            if any(a["category_id"] in desired_category_ids for a in annotations_by_image_id.get(image_id, []))
        ]

    def visualize_split(self, split_name, shuffle=True, desired_category_ids=None):
        matching_splits = [split for split in self.splits if split["name"] == split_name]
        if not matching_splits:
            print(f"Split '{split_name}' not found. Available: {[s['name'] for s in self.splits]}")
            return

        split = matching_splits[0]
        coco_data = split["coco_data"]
        split_path = split["split_path"]
        image_id_to_filename = split["image_id_to_filename"]

        category_id_to_name = self._build_category_mapping(coco_data)
        annotations_by_image_id = self._group_annotations_by_image(coco_data)

        image_ids = list(image_id_to_filename.keys())
        if shuffle:
            random.shuffle(image_ids)

        if desired_category_ids is not None:
            image_ids = self._filter_image_ids_by_category_ids(
                image_ids, annotations_by_image_id, desired_category_ids
            )

        total_image_count = len(image_ids)
        if total_image_count == 0:
            print(f"\n[{split_name}] No images contain the selected labels.")
            return

        print(f"\n[{split_name}] {total_image_count} images")
        self._print_color_legend(category_id_to_name)

        current_index = 0
        while 0 <= current_index < total_image_count:
            current_image_id = image_ids[current_index]
            image_filename = image_id_to_filename[current_image_id]
            image_path = os.path.join(split_path, image_filename)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}, skipping...")
                current_index += 1
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read: {image_path}, skipping...")
                current_index += 1
                continue

            annotations_for_current_image = annotations_by_image_id.get(current_image_id, [])
            filtered_annotations = self._filter_annotations_by_category_ids(
                annotations_for_current_image, desired_category_ids
            )
            annotated_image = self._draw_bounding_boxes_on_image(
                image.copy(), filtered_annotations, category_id_to_name
            )

            window_title = f"[{split_name}] [{current_index + 1}/{total_image_count}] {image_filename}"
            cv2.imshow(window_title, annotated_image)

            pressed_key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if pressed_key in (self.KEY_QUIT_Q, self.KEY_QUIT_ESC):
                break
            elif pressed_key == self.KEY_PREVIOUS:
                current_index = max(0, current_index - 1)
            else:
                current_index += 1

    def visualize_all_splits(self, shuffle=True):
        if not self.splits:
            print("No valid splits found in the dataset.")
            return

        print(f"Found splits: {[split['name'] for split in self.splits]}")

        desired_category_ids = None
        if self.detailed_mode:
            all_category_id_to_name = self._collect_all_category_ids_across_splits()
            self._print_color_legend(all_category_id_to_name)
            desired_category_ids = self._prompt_user_for_desired_category_ids(all_category_id_to_name)

        for split in self.splits:
            print(f"\nLoading [{split['name']}] from: {split['annotation_file_path']}")
            self.visualize_split(split["name"], shuffle=shuffle, desired_category_ids=desired_category_ids)
