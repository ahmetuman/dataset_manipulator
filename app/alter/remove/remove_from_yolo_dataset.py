import os
import shutil
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
NO_ANNOTATIONS_FOLDER_NAME = "no_annotations"


class YoloLabelRemover:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.class_names = self._load_class_names()
        self.split_dirs = self._discover_splits()

    def _load_class_names(self) -> dict[int, str]:
        yaml_path = self._find_yaml_file()
        class_names = {}
        inside_names_block = False

        with open(yaml_path, "r") as file:
            for line in file:
                stripped = line.strip()

                if stripped.startswith("names:"):
                    remainder = stripped[len("names:"):].strip()
                    if remainder.startswith("["):
                        items = remainder.strip("[]").split(",")
                        return {i: item.strip().strip("'\"") for i, item in enumerate(items)}
                    inside_names_block = True
                    continue

                if inside_names_block:
                    if not stripped or (not stripped[0].isdigit() and stripped[0] != "-"):
                        break
                    if stripped.startswith("-"):
                        index = len(class_names)
                        name = stripped.lstrip("- ").strip("'\"")
                    else:
                        parts = stripped.split(":", 1)
                        index = int(parts[0].strip())
                        name = parts[1].strip().strip("'\"")
                    class_names[index] = name

        if not class_names:
            raise ValueError(f"Could not parse class names from {yaml_path}")
        return class_names

    def _find_yaml_file(self) -> Path:
        yaml_candidates = list(self.dataset_root.glob("*.yaml")) + list(self.dataset_root.glob("*.yml"))
        if not yaml_candidates:
            raise FileNotFoundError(f"No YAML file found in {self.dataset_root}")
        if len(yaml_candidates) > 1:
            print(f"[warning] Multiple YAML files found, using: {yaml_candidates[0].name}")
        return yaml_candidates[0]

    def _discover_splits(self) -> list[str]:
        found_splits = []
        for candidate in ["train", "valid", "val", "test"]:
            labels_dir = self.dataset_root / candidate / "labels"
            if labels_dir.is_dir():
                found_splits.append(candidate)
        return found_splits

    def _collect_label_files(self) -> list[Path]:
        all_label_files = []
        for split in self.split_dirs:
            labels_dir = self.dataset_root / split / "labels"
            all_label_files.extend(labels_dir.glob("*.txt"))
        return all_label_files

    def _count_annotations_per_class(self) -> dict[int, int]:
        counts = {class_id: 0 for class_id in self.class_names}
        for label_file in self._collect_label_files():
            for line in label_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] = counts.get(class_id, 0) + 1
        return counts

    def _print_class_summary(self):
        counts = self._count_annotations_per_class()
        print(f"\n  Dataset: {self.dataset_root.name}")
        print(f"  Splits:  {', '.join(self.split_dirs)}\n")

        print(f"  {'ID':<6}{'Class Name':<30}{'Annotations':>12}")
        print("-" * 55)
        for class_id in sorted(self.class_names.keys()):
            name = self.class_names[class_id]
            count = counts.get(class_id, 0)
            print(f"  {class_id:<6}{name:<30}{count:>12}")

    def _prompt_ids_to_remove(self) -> list[int]:
        raw_input = input("\nEnter class IDs to remove (comma-separated, e.g. 1,3,7): ").strip()
        if not raw_input:
            return []

        ids_to_remove = []
        for token in raw_input.split(","):
            token = token.strip()
            if not token.isdigit():
                print(f"[error] '{token}' is not a valid integer ID. Aborting.")
                return []
            parsed_id = int(token)
            if parsed_id not in self.class_names:
                print(f"[error] ID {parsed_id} does not exist in the dataset. Aborting.")
                return []
            ids_to_remove.append(parsed_id)
        return ids_to_remove

    def _confirm_removal(self, ids_to_remove: list[int]) -> bool:
        remaining_ids = sorted(set(self.class_names.keys()) - set(ids_to_remove))
        remapping = {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}

        print("\n PLANNED CHANGES: \n")

        print("\n  Classes to REMOVE:")
        for class_id in ids_to_remove:
            print(f"    ID {class_id}: {self.class_names[class_id]}")

        print("\n  Classes to KEEP (with new IDs):")
        for old_id in remaining_ids:
            new_id = remapping[old_id]
            marker = f" <- was {old_id}" if new_id != old_id else ""
            print(f"    ID {new_id}: {self.class_names[old_id]}{marker}")

        print("-" * 55)
        confirmation = input("\n  Proceed? (yes/no): ").strip().lower()
        return confirmation in ("yes", "y")

    def _build_remapping(self, ids_to_remove: list[int]) -> dict[int, int]:
        remaining_ids = sorted(set(self.class_names.keys()) - set(ids_to_remove))
        return {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}

    def _rewrite_label_files(self, ids_to_remove: set[int], remapping: dict[int, int]) -> dict[str, list[Path]]:
        emptied_label_files_per_split = {split: [] for split in self.split_dirs}

        for split in self.split_dirs:
            labels_dir = self.dataset_root / split / "labels"
            for label_file in labels_dir.glob("*.txt"):
                original_lines = label_file.read_text().strip().splitlines()
                kept_lines = []

                for line in original_lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if class_id in ids_to_remove:
                        continue
                    parts[0] = str(remapping[class_id])
                    kept_lines.append(" ".join(parts))

                label_file.write_text("\n".join(kept_lines) + "\n" if kept_lines else "")

                if not kept_lines:
                    emptied_label_files_per_split[split].append(label_file)

        return emptied_label_files_per_split

    def _move_unannotated_samples(self, emptied_label_files_per_split: dict[str, list[Path]]):
        total_moved = 0

        for split, empty_label_files in emptied_label_files_per_split.items():
            if not empty_label_files:
                continue

            no_ann_images_dir = self.dataset_root / split / NO_ANNOTATIONS_FOLDER_NAME / "images"
            no_ann_labels_dir = self.dataset_root / split / NO_ANNOTATIONS_FOLDER_NAME / "labels"
            no_ann_images_dir.mkdir(parents=True, exist_ok=True)
            no_ann_labels_dir.mkdir(parents=True, exist_ok=True)

            images_dir = self.dataset_root / split / "images"

            for label_file in empty_label_files:
                stem = label_file.stem
                shutil.move(str(label_file), str(no_ann_labels_dir / label_file.name))

                for extension in SUPPORTED_IMAGE_EXTENSIONS:
                    image_path = images_dir / (stem + extension)
                    if image_path.exists():
                        shutil.move(str(image_path), str(no_ann_images_dir / image_path.name))
                        break

                total_moved += 1

        return total_moved

    def _update_yaml_file(self, ids_to_remove: list[int]):
        remaining_ids = sorted(set(self.class_names.keys()) - set(ids_to_remove))
        new_names = [self.class_names[old_id] for old_id in remaining_ids]

        yaml_path = self._find_yaml_file()
        lines = yaml_path.read_text().splitlines()
        new_lines = []
        inside_names_block = False
        names_written = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("names:"):
                inside_names_block = True
                names_written = True
                new_lines.append("names:")
                for new_id, name in enumerate(new_names):
                    new_lines.append(f"  {new_id}: {name}")
                continue

            if inside_names_block:
                if stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                    continue
                inside_names_block = False

            if stripped.startswith("nc:"):
                new_lines.append(f"nc: {len(new_names)}")
                continue

            new_lines.append(line)

        if not names_written:
            new_lines.append("names:")
            for new_id, name in enumerate(new_names):
                new_lines.append(f"  {new_id}: {name}")

        yaml_path.write_text("\n".join(new_lines) + "\n")

    def remove(self):
        self._print_class_summary()

        ids_to_remove = self._prompt_ids_to_remove()
        if not ids_to_remove:
            print("Nothing to remove. Exiting.")
            return

        if not self._confirm_removal(ids_to_remove):
            print("Cancelled.")
            return

        ids_to_remove_set = set(ids_to_remove)
        remapping = self._build_remapping(ids_to_remove)

        print("\n  Rewriting label files...")
        emptied_label_files_per_split = self._rewrite_label_files(ids_to_remove_set, remapping)

        print("  Moving unannotated samples...")
        total_moved = self._move_unannotated_samples(emptied_label_files_per_split)

        print("  Updating YAML config...")
        self._update_yaml_file(ids_to_remove)

        print(f"\n  Done.")
        print(f"  Removed {len(ids_to_remove)} class(es).")
        print(f"  Moved {total_moved} image(s) with no remaining annotations to '{NO_ANNOTATIONS_FOLDER_NAME}/'.")