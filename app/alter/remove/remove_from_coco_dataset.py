from __future__ import annotations

import shutil
from pathlib import Path

from tabulate import tabulate

from app.utils.coco_files import find_annotation_file
from app.utils.coco_files import load_coco
from app.utils.coco_files import save_coco

NO_ANNOTATIONS_FOLDER_NAME = "no_annotations"


class CocoLabelRemover:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.split_dirs = self._discover_splits()
        self.categories = self._load_categories()

    def _discover_splits(self) -> list[str]:
        found_splits = []
        for candidate in ["train", "valid", "val", "test"]:
            split_dir = self.dataset_root / candidate
            if split_dir.is_dir() and find_annotation_file(split_dir):
                found_splits.append(candidate)
        if not found_splits:
            raise FileNotFoundError(f"No splits with annotation files found in {self.dataset_root}")
        return found_splits

    def _load_annotation_data(self, split: str) -> dict:
        return load_coco(find_annotation_file(self.dataset_root / split))

    def _save_annotation_data(self, split: str, data: dict):
        save_coco(find_annotation_file(self.dataset_root / split), data)

    def _load_categories(self) -> dict[int, str]:
        first_split = self.split_dirs[0]
        data = self._load_annotation_data(first_split)
        return {cat["id"]: cat["name"] for cat in data["categories"]}

    def _count_annotations_per_category(self) -> dict[int, int]:
        counts = {category_id: 0 for category_id in self.categories}
        for split in self.split_dirs:
            data = self._load_annotation_data(split)
            for annotation in data.get("annotations", []):
                category_id = annotation["category_id"]
                counts[category_id] = counts.get(category_id, 0) + 1
        return counts

    def _print_category_summary(self):
        counts = self._count_annotations_per_category()
        print(f"\nDataset: {self.dataset_root.name}")
        print(f"Splits:  {', '.join(self.split_dirs)}\n")

        rows = []
        for category_id in sorted(self.categories.keys()):
            name = self.categories[category_id]
            count = counts.get(category_id, 0)
            rows.append([category_id, name, count])

        print(tabulate(rows, headers=["ID", "Category Name", "Annotations"], tablefmt="simple"))

    def _prompt_ids_to_remove(self) -> list[int]:
        raw_input = input("\nEnter category IDs to remove (comma-separated, e.g. 1,3,7): ").strip()
        if not raw_input:
            return []

        ids_to_remove = []
        for token in raw_input.split(","):
            token = token.strip()
            if not token.lstrip("-").isdigit():
                print(f"[error] '{token}' is not a valid integer ID. Aborting.")
                return []
            parsed_id = int(token)
            if parsed_id not in self.categories:
                print(f"[error] ID {parsed_id} does not exist in the dataset. Aborting.")  # noqa E713
                return []
            ids_to_remove.append(parsed_id)
        return ids_to_remove

    def _confirm_removal(self, ids_to_remove: list[int]) -> bool:
        remaining_ids = sorted(set(self.categories.keys()) - set(ids_to_remove))
        remapping = {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}

        print("\nPLANNED CHANGES:")

        print("\n  Categories to REMOVE:")
        for category_id in ids_to_remove:
            print(f"    ID {category_id}: {self.categories[category_id]}")

        print("\n  Categories to KEEP (with new IDs):")
        for old_id in remaining_ids:
            new_id = remapping[old_id]
            marker = f" <- was {old_id}" if new_id != old_id else ""
            print(f"    ID {new_id}: {self.categories[old_id]}{marker}")

        confirmation = input("\n  Proceed? (yes/no): ").strip().lower()
        return confirmation in ("yes", "y")

    def _build_remapping(self, ids_to_remove: list[int]) -> dict[int, int]:
        remaining_ids = sorted(set(self.categories.keys()) - set(ids_to_remove))
        return {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}

    def _build_remapped_categories(self, ids_to_remove: set[int],
                                   remapping: dict[int, int],
                                   original_categories: list[dict]) -> list[dict]:
        remapped_categories = []
        for category in original_categories:
            if category["id"] in ids_to_remove:
                continue
            remapped_category = dict(category)
            remapped_category["id"] = remapping[category["id"]]
            remapped_categories.append(remapped_category)
        return remapped_categories

    def _filter_and_remap_annotations(self, annotations: list[dict],
                                      ids_to_remove: set[int],
                                      remapping: dict[int, int]) -> tuple[list[dict], set[int]]:
        kept_annotations = []
        annotated_image_ids = set()
        next_annotation_id = 1

        for annotation in annotations:
            if annotation["category_id"] in ids_to_remove:
                continue
            remapped_annotation = dict(annotation)
            remapped_annotation["category_id"] = remapping[annotation["category_id"]]
            remapped_annotation["id"] = next_annotation_id
            next_annotation_id += 1
            kept_annotations.append(remapped_annotation)
            annotated_image_ids.add(annotation["image_id"])

        return kept_annotations, annotated_image_ids

    def _move_unannotated_images(self, split: str, unannotated_images: list[dict]) -> int:
        if not unannotated_images:
            return 0

        split_dir = self.dataset_root / split
        destination_dir = split_dir / NO_ANNOTATIONS_FOLDER_NAME
        destination_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for image_entry in unannotated_images:
            filename = image_entry["file_name"]
            source_path = split_dir / filename
            if source_path.exists():
                shutil.move(str(source_path), str(destination_dir / filename))
                moved_count += 1

        return moved_count

    def _process_split(self, split: str, ids_to_remove: set[int], remapping: dict[int, int]) -> int:
        data = self._load_annotation_data(split)

        data["categories"] = self._build_remapped_categories(
            ids_to_remove, remapping, data.get("categories", [])
        )

        kept_annotations, annotated_image_ids = self._filter_and_remap_annotations(
            data.get("annotations", []), ids_to_remove, remapping
        )
        data["annotations"] = kept_annotations

        unannotated_images = [
            image for image in data.get("images", [])
            if image["id"] not in annotated_image_ids
        ]

        data["images"] = [
            image for image in data["images"]
            if image["id"] in annotated_image_ids
        ]

        self._save_annotation_data(split, data)

        return self._move_unannotated_images(split, unannotated_images)

    def remove(self):
        self._print_category_summary()

        ids_to_remove = self._prompt_ids_to_remove()
        if not ids_to_remove:
            print("Nothing to remove. Exiting.")
            return

        if not self._confirm_removal(ids_to_remove):
            print("Cancelled.")
            return

        ids_to_remove_set = set(ids_to_remove)
        remapping = self._build_remapping(ids_to_remove)

        total_moved = 0
        for split in self.split_dirs:
            print(f"\n  Processing '{split}'...")
            moved = self._process_split(split, ids_to_remove_set, remapping)
            total_moved += moved
            print(f"    Moved {moved} unannotated image(s) to '{NO_ANNOTATIONS_FOLDER_NAME}/'.")

        print("\n  Done.")
        print(f"  Removed {len(ids_to_remove)} category/categories.")
        print(f"  Moved {total_moved} total image(s) with no remaining annotations.")
