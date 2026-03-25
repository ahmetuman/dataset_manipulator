import os
import ast
import shutil


SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
SPLIT_NAMES = ["train", "valid", "test"]
PARENT_CLASS_PREFIX = "Parent_"
UNWANTED_CLASS = "Remove"


class YoloDatasetMerger:
    def __init__(self, datasets_root_directory, output_directory):
        self.datasets_root_directory = datasets_root_directory
        self.output_directory = output_directory
        self.dataset_folders = []
        self.unified_class_map = {}

    def merge(self):
        self.dataset_folders = self._find_dataset_folders()
        if not self.dataset_folders:
            raise FileNotFoundError("No YOLO datasets found in the root directory.")

        self.unified_class_map = self._build_unified_class_map()

        merge_results = {}
        for split_name in SPLIT_NAMES:
            result = self._merge_split(split_name)
            merge_results[split_name] = result

        self._write_data_yaml()
        return merge_results

    def _find_dataset_folders(self):
        folders = []
        for entry in sorted(os.listdir(self.datasets_root_directory)):
            dataset_path = os.path.join(self.datasets_root_directory, entry)
            if not os.path.isdir(dataset_path):
                continue
            if self._dataset_has_labels(dataset_path):
                folders.append(dataset_path)
        return folders

    def _dataset_has_labels(self, dataset_path):
        for split_name in SPLIT_NAMES:
            split_path = os.path.join(dataset_path, split_name)
            if not os.path.isdir(split_path):
                continue
            _, labels_directory = self._detect_split_layout(split_path)
            label_files = [f for f in os.listdir(labels_directory) if f.endswith(".txt")]
            if label_files:
                return True
        return False

    @staticmethod
    def _detect_split_layout(split_path):
        images_directory = os.path.join(split_path, "images")
        labels_directory = os.path.join(split_path, "labels")
        if os.path.isdir(images_directory) and os.path.isdir(labels_directory):
            return images_directory, labels_directory
        return split_path, split_path

    @staticmethod
    def _find_matching_image(images_directory, label_stem):
        for extension in SUPPORTED_IMAGE_EXTENSIONS:
            image_path = os.path.join(images_directory, label_stem + extension)
            if os.path.exists(image_path):
                return image_path
        return None

    @staticmethod
    def _load_class_names_from_yaml(dataset_directory):
        yaml_path = os.path.join(dataset_directory, "data.yaml")
        if not os.path.exists(yaml_path):
            return {}

        class_names = {}
        with open(yaml_path, "r") as yaml_file:
            for line in yaml_file:
                line = line.strip()
                if not line.startswith("names:"):
                    continue

                value_after_key = line[len("names:"):].strip()
                if value_after_key.startswith("["):
                    parsed_list = ast.literal_eval(value_after_key)
                    return {index: name for index, name in enumerate(parsed_list)}

                for subsequent_line in yaml_file:
                    subsequent_line = subsequent_line.strip()
                    if ":" in subsequent_line and subsequent_line[0].isdigit():
                        parts = subsequent_line.split(":", 1)
                        class_id = int(parts[0].strip())
                        class_name = parts[1].strip().strip("'\"")
                        class_names[class_id] = class_name
                    elif subsequent_line.startswith("- "):
                        class_name = subsequent_line[2:].strip().strip("'\"")
                        class_names[len(class_names)] = class_name
                    else:
                        break
        return class_names

    def _build_unified_class_map(self):
        all_class_names = set()
        for folder in self.dataset_folders:
            class_names = self._load_class_names_from_yaml(folder)
            for name in class_names.values():
                # if UNWANTED_CLASS not in name:
                #     all_class_names.add(name)
                all_class_names.add(name)
                
        return {name: index for index, name in enumerate(sorted(all_class_names))}

    def _build_class_id_remapping(self, class_names):
        old_to_new_id = {}
        parent_class_ids = set()
        for class_id, name in class_names.items():
            if name.startswith(PARENT_CLASS_PREFIX):
                parent_class_ids.add(class_id)
            elif name in self.unified_class_map:
                old_to_new_id[class_id] = self.unified_class_map[name]
        return old_to_new_id, parent_class_ids

    def _merge_split(self, split_name):
        output_images_directory = os.path.join(self.output_directory, split_name, "images")
        output_labels_directory = os.path.join(self.output_directory, split_name, "labels")
        os.makedirs(output_images_directory, exist_ok=True)
        os.makedirs(output_labels_directory, exist_ok=True)

        total_images = 0
        total_annotations = 0
        skipped_parent_annotations = 0

        for folder in self.dataset_folders:
            split_path = os.path.join(folder, split_name)
            if not os.path.isdir(split_path):
                continue

            images_directory, labels_directory = self._detect_split_layout(split_path)
            class_names = self._load_class_names_from_yaml(folder)
            dataset_name = os.path.basename(folder)
            old_to_new_id, parent_class_ids = self._build_class_id_remapping(class_names)

            label_files = sorted(f for f in os.listdir(labels_directory) if f.endswith(".txt"))

            for label_filename in label_files:
                result = self._process_single_label(
                    label_filename,
                    labels_directory,
                    images_directory,
                    dataset_name,
                    old_to_new_id,
                    parent_class_ids,
                    output_images_directory,
                    output_labels_directory,
                )
                if result is None:
                    continue

                total_images += 1
                total_annotations += result["annotations"]
                skipped_parent_annotations += result["skipped_parents"]

        return {
            "images": total_images,
            "annotations": total_annotations,
            "skipped_parent_annotations": skipped_parent_annotations,
        }

    def _process_single_label(
        self,
        label_filename,
        labels_directory,
        images_directory,
        dataset_name,
        old_to_new_id,
        parent_class_ids,
        output_images_directory,
        output_labels_directory,
    ):
        label_path = os.path.join(labels_directory, label_filename)
        stem = os.path.splitext(label_filename)[0]

        image_path = self._find_matching_image(images_directory, stem)
        if image_path is None:
            return None

        with open(label_path, "r") as label_file:
            raw_lines = [line.strip() for line in label_file if line.strip()]

        remapped_lines = []
        skipped_parents = 0

        for line in raw_lines:
            parts = line.split()
            class_id = int(parts[0])
            if class_id in parent_class_ids:
                skipped_parents += 1
                continue
            if class_id not in old_to_new_id:
                continue
            parts[0] = str(old_to_new_id[class_id])
            remapped_lines.append(" ".join(parts))

        if not remapped_lines:
            return None

        image_extension = os.path.splitext(image_path)[1]
        prefixed_name = f"{dataset_name}_{stem}"
        output_image_path = os.path.join(output_images_directory, prefixed_name + image_extension)
        output_label_path = os.path.join(output_labels_directory, prefixed_name + ".txt")

        shutil.copy2(image_path, output_image_path)
        with open(output_label_path, "w") as output_label_file:
            output_label_file.write("\n".join(remapped_lines) + "\n")

        return {"annotations": len(remapped_lines), "skipped_parents": skipped_parents}

    def _write_data_yaml(self):
        yaml_path = os.path.join(self.output_directory, "data.yaml")
        sorted_class_names = [
            name for name, _ in sorted(self.unified_class_map.items(), key=lambda item: item[1])
        ]

        with open(yaml_path, "w") as yaml_file:
            yaml_file.write(f"nc: {len(sorted_class_names)}\n")
            yaml_file.write(f"names: {sorted_class_names}\n")
            yaml_file.write("train: train/images\n")
            yaml_file.write("val: valid/images\n")
            yaml_file.write("test: test/images\n")
