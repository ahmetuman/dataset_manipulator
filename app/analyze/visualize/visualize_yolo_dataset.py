import os
import ast
import random

import cv2


class YoloDatasetVisualizer:
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

    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.class_names = self._load_class_names_from_yaml()
        self.splits = self._discover_available_splits()

    def _load_class_names_from_yaml(self):
        yaml_path = os.path.join(self.dataset_directory, "data.yaml")
        if not os.path.exists(yaml_path):
            return {}

        class_names = {}
        with open(yaml_path, "r") as yaml_file:
            for line in yaml_file:
                line = line.strip()
                if not line.startswith("names:"):
                    continue

                remaining_content = line[len("names:"):].strip()
                if remaining_content.startswith("["):
                    for index, name in enumerate(ast.literal_eval(remaining_content)):
                        class_names[index] = name
                    return class_names

                for subsequent_line in yaml_file:
                    subsequent_line = subsequent_line.strip()
                    if ":" in subsequent_line and subsequent_line[0].isdigit():
                        class_id, class_name = subsequent_line.split(":", 1)
                        class_names[int(class_id.strip())] = class_name.strip().strip("'\"")
                    elif subsequent_line.startswith("- "):
                        class_names[len(class_names)] = subsequent_line[2:].strip().strip("'\"")
                    else:
                        break
        return class_names

    def _resolve_split_directories(self, split_path):
        images_directory = os.path.join(split_path, "images")
        labels_directory = os.path.join(split_path, "labels")
        if os.path.isdir(images_directory) and os.path.isdir(labels_directory):
            return images_directory, labels_directory
        return split_path, split_path

    def _collect_image_label_pairs(self, split_path):
        images_directory, labels_directory = self._resolve_split_directories(split_path)
        image_label_pairs = []

        sorted_filenames = sorted(os.listdir(images_directory))
        image_filenames = [
            filename for filename in sorted_filenames
            if os.path.splitext(filename)[1].lower() in self.SUPPORTED_IMAGE_EXTENSIONS
        ]

        for image_filename in image_filenames:
            name_without_extension = os.path.splitext(image_filename)[0]
            label_filename = name_without_extension + ".txt"
            label_path = os.path.join(labels_directory, label_filename)

            if os.path.exists(label_path):
                image_label_pairs.append((image_filename, label_filename))

        return image_label_pairs, images_directory, labels_directory

    def _discover_available_splits(self):
        available_splits = []
        for split_name in ["train", "test", "valid"]:
            split_path = os.path.join(self.dataset_directory, split_name)
            if not os.path.isdir(split_path):
                continue
            image_label_pairs, images_directory, labels_directory = self._collect_image_label_pairs(split_path)
            if image_label_pairs:
                available_splits.append({
                    "name": split_name,
                    "images_directory": images_directory,
                    "labels_directory": labels_directory,
                    "image_label_pairs": image_label_pairs,
                })
        return available_splits

    def _parse_yolo_label_file(self, label_file_path):
        annotations = []
        with open(label_file_path, "r") as label_file:
            for line in label_file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                annotations.append({
                    "category_id": int(parts[0]),
                    "center_x_normalized": float(parts[1]),
                    "center_y_normalized": float(parts[2]),
                    "width_normalized": float(parts[3]),
                    "height_normalized": float(parts[4]),
                })
        return annotations

    def _convert_normalized_to_pixel_coordinates(self, annotation, image_width, image_height):
        center_x = annotation["center_x_normalized"] * image_width
        center_y = annotation["center_y_normalized"] * image_height
        box_width = annotation["width_normalized"] * image_width
        box_height = annotation["height_normalized"] * image_height

        top_left_x = int(center_x - box_width / 2)
        top_left_y = int(center_y - box_height / 2)
        bottom_right_x = int(center_x + box_width / 2)
        bottom_right_y = int(center_y + box_height / 2)

        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    def _get_color_for_category(self, category_id):
        return self.CATEGORY_COLORS[category_id % len(self.CATEGORY_COLORS)]

    def _get_label_text_for_category(self, category_id):
        return self.class_names.get(category_id, f"class_{category_id}")

    def _draw_bounding_boxes_on_image(self, image, annotations):
        image_height, image_width = image.shape[:2]

        for annotation in annotations:
            category_id = annotation["category_id"]
            color = self._get_color_for_category(category_id)
            label_text = self._get_label_text_for_category(category_id)

            top_left_x, top_left_y, bottom_right_x, bottom_right_y = (
                self._convert_normalized_to_pixel_coordinates(annotation, image_width, image_height)
            )

            cv2.rectangle(
                image,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                color,
                self.BOUNDING_BOX_THICKNESS,
            )

            text_size, _ = cv2.getTextSize(label_text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            text_background_top_left = (top_left_x, top_left_y - text_size[1] - 6)
            text_background_bottom_right = (top_left_x + text_size[0] + 4, top_left_y)

            cv2.rectangle(image, text_background_top_left, text_background_bottom_right, color, -1)
            cv2.putText(
                image,
                label_text,
                (top_left_x + 2, top_left_y - 4),
                self.FONT,
                self.FONT_SCALE,
                (0, 0, 0),
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return image

    def _print_color_legend(self, category_ids):
        print("\nLabel Colors:")
        for category_id in category_ids:
            blue, green, red = self._get_color_for_category(category_id)
            label_name = self._get_label_text_for_category(category_id)
            print(f"  {label_name} (id: {category_id}) -> RGB({red}, {green}, {blue})")
        print()

    def _collect_all_category_ids_in_split(self, labels_directory, image_label_pairs):
        all_category_ids = set()
        for _, label_filename in image_label_pairs:
            label_path = os.path.join(labels_directory, label_filename)
            for annotation in self._parse_yolo_label_file(label_path):
                all_category_ids.add(annotation["category_id"])
        return sorted(all_category_ids)

    def visualize_split(self, split_name, shuffle=True):
        matching_splits = [split for split in self.splits if split["name"] == split_name]
        if not matching_splits:
            print(f"Split '{split_name}' not found. Available: {[s['name'] for s in self.splits]}")
            return

        split = matching_splits[0]
        images_directory = split["images_directory"]
        labels_directory = split["labels_directory"]
        image_label_pairs = list(split["image_label_pairs"])

        if shuffle:
            random.shuffle(image_label_pairs)

        total_image_count = len(image_label_pairs)
        all_category_ids = self._collect_all_category_ids_in_split(labels_directory, image_label_pairs)

        print(f"\n[{split_name}] {total_image_count} images")
        self._print_color_legend(all_category_ids)

        current_index = 0
        while 0 <= current_index < total_image_count:
            image_filename, label_filename = image_label_pairs[current_index]
            image_path = os.path.join(images_directory, image_filename)
            label_path = os.path.join(labels_directory, label_filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read: {image_path}, skipping...")
                current_index += 1
                continue

            annotations = self._parse_yolo_label_file(label_path)
            annotated_image = self._draw_bounding_boxes_on_image(image.copy(), annotations)

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
        for split in self.splits:
            print(f"\nLoading [{split['name']}] from: {split['images_directory']}")
            self.visualize_split(split["name"], shuffle=shuffle)