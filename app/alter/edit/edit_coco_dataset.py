from __future__ import annotations

from pathlib import Path

from app.utils.coco_files import ANNOTATIONS_FILENAME
from app.utils.coco_files import load_coco
from app.utils.coco_files import save_coco


class CocoLabelEditor:
    def __init__(self, dataset_path: str):
        self._dataset_path = Path(dataset_path)
        self._annotation_files = self._find_annotation_files()
        self._original_names = self._load_original_names()

    def edit(self):
        rename_map = self._collect_renames()

        if not rename_map:
            print("No changes requested.")
            return

        merged_names = self._build_merged_name_list(rename_map)
        index_remap = self._build_index_remap(rename_map, merged_names)

        self._update_annotation_files(index_remap, merged_names)

        print(f"\nDone. {len(self._original_names)} labels -> {len(merged_names)} labels.")

    def _find_annotation_files(self) -> list[Path]:
        annotation_files = list(self._dataset_path.rglob(ANNOTATIONS_FILENAME))
        if not annotation_files:
            raise FileNotFoundError(f"No {ANNOTATIONS_FILENAME} files found under {self._dataset_path}")
        return sorted(annotation_files)

    def _load_original_names(self) -> list[str]:
        data = load_coco(self._annotation_files[0])
        categories_sorted = sorted(data["categories"], key=lambda category: category["id"])
        return [category["name"] for category in categories_sorted]

    def _collect_renames(self) -> dict[str, str]:
        rename_map = {}
        total = len(self._original_names)

        for index, name in enumerate(self._original_names, start=1):
            new_name = input(
                f"({index}/{total}) Current label: '{name}' | "
                "Enter new name or press Enter to skip: "
            ).strip()

            if not new_name:
                continue

            confirmed = input(f"Are you sure: '{name}' -> '{new_name}'? (y/n): ").strip().lower()
            if confirmed == "y":
                rename_map[name] = new_name
                print(f"  Registered: '{name}' -> '{new_name}'")
            else:
                print("  Skipped.")

        return rename_map

    def _build_merged_name_list(self, rename_map: dict[str, str]) -> list[str]:
        seen = {}
        merged_names = []

        for name in self._original_names:
            resolved_name = rename_map.get(name, name)
            if resolved_name not in seen:
                seen[resolved_name] = len(merged_names)
                merged_names.append(resolved_name)

        return merged_names

    def _build_index_remap(self, rename_map: dict[str, str], merged_names: list[str]) -> dict[int, int]:
        name_to_new_index = {name: index for index, name in enumerate(merged_names)}
        index_remap = {}

        for old_index, old_name in enumerate(self._original_names):
            resolved_name = rename_map.get(old_name, old_name)
            new_index = name_to_new_index[resolved_name]
            if old_index != new_index:
                index_remap[old_index] = new_index

        return index_remap

    def _update_annotation_files(self, index_remap: dict[int, int], merged_names: list[str]):
        for annotation_file in self._annotation_files:
            self._remap_single_file(annotation_file, index_remap, merged_names)

    def _remap_single_file(self, file_path: Path, index_remap: dict[int, int], merged_names: list[str]):
        data = load_coco(file_path)

        old_id_to_old_index = {}
        for category in sorted(data["categories"], key=lambda category: category["id"]):
            old_id_to_old_index[category["id"]] = len(old_id_to_old_index)

        old_id_to_new_id = {}
        for old_id, old_index in old_id_to_old_index.items():
            new_index = index_remap.get(old_index, old_index)
            old_id_to_new_id[old_id] = new_index

        data["categories"] = [
            {"id": index, "name": name, "supercategory": "none"}
            for index, name in enumerate(merged_names)
        ]

        for annotation in data["annotations"]:
            old_category_id = annotation["category_id"]
            annotation["category_id"] = old_id_to_new_id[old_category_id]

        save_coco(file_path, data, ensure_ascii=False)
