from __future__ import annotations

import shutil
from pathlib import Path

from app.utils.image_files import find_matching_image
from app.utils.yaml_config import load_class_names


SPLIT_NAMES = ["train", "valid", "test"]
PARENT_CLASS_PREFIX = "Parent_"


class YoloDatasetMerger:
    def __init__(self, datasets_root_directory, output_directory):
        self.datasets_root_directory = Path(datasets_root_directory)
        self.output_directory = Path(output_directory)
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

    def _find_dataset_folders(self) -> list[Path]:
        folders = []
        for entry in sorted(self.datasets_root_directory.iterdir()):
            if entry.is_dir() and self._dataset_has_labels(entry):
                folders.append(entry)
        return folders

    def _dataset_has_labels(self, dataset_path: Path) -> bool:
        for split_name in SPLIT_NAMES:
            split_path = dataset_path / split_name
            if not split_path.is_dir():
                continue
            _, labels_directory = self._detect_split_layout(split_path)
            if any(labels_directory.glob("*.txt")):
                return True
        return False

    @staticmethod
    def _detect_split_layout(split_path: Path) -> tuple[Path, Path]:
        images_directory = split_path / "images"
        labels_directory = split_path / "labels"
        if images_directory.is_dir() and labels_directory.is_dir():
            return images_directory, labels_directory
        return split_path, split_path

    @staticmethod
    def _load_class_names_from_yaml(dataset_directory: Path) -> dict[int, str]:
        yaml_path = dataset_directory / "data.yaml"
        if not yaml_path.exists():
            return {}
        return load_class_names(yaml_path)

    def _build_unified_class_map(self) -> dict[str, int]:
        all_class_names = set()
        for folder in self.dataset_folders:
            class_names = self._load_class_names_from_yaml(folder)
            all_class_names.update(class_names.values())
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
        output_images_directory = self.output_directory / split_name / "images"
        output_labels_directory = self.output_directory / split_name / "labels"
        output_images_directory.mkdir(parents=True, exist_ok=True)
        output_labels_directory.mkdir(parents=True, exist_ok=True)

        total_images = 0
        total_annotations = 0
        skipped_parent_annotations = 0

        for folder in self.dataset_folders:
            split_path = folder / split_name
            if not split_path.is_dir():
                continue

            images_directory, labels_directory = self._detect_split_layout(split_path)
            class_names = self._load_class_names_from_yaml(folder)
            dataset_name = folder.name
            old_to_new_id, parent_class_ids = self._build_class_id_remapping(class_names)

            for label_file in sorted(labels_directory.glob("*.txt")):
                result = self._process_single_label(
                    label_file,
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
        label_file: Path,
        images_directory: Path,
        dataset_name,
        old_to_new_id,
        parent_class_ids,
        output_images_directory: Path,
        output_labels_directory: Path,
    ):
        stem = label_file.stem
        image_path = find_matching_image(images_directory, stem)
        if image_path is None:
            return None

        raw_lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]

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

        prefixed_name = f"{dataset_name}_{stem}"
        output_image_path = output_images_directory / (prefixed_name + image_path.suffix)
        output_label_path = output_labels_directory / (prefixed_name + ".txt")

        shutil.copy2(image_path, output_image_path)
        output_label_path.write_text("\n".join(remapped_lines) + "\n")

        return {"annotations": len(remapped_lines), "skipped_parents": skipped_parents}

    def _write_data_yaml(self):
        sorted_class_names = [
            name for name, _ in sorted(self.unified_class_map.items(), key=lambda item: item[1])
        ]

        lines = [
            f"nc: {len(sorted_class_names)}",
            f"names: {sorted_class_names}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
        ]
        (self.output_directory / "data.yaml").write_text("\n".join(lines) + "\n")
