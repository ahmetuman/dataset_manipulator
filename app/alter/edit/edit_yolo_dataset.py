from __future__ import annotations

from pathlib import Path

import yaml


class YoloLabelEditor:
    def __init__(self, dataset_path: str):
        self._dataset_path = Path(dataset_path)
        self._config_path = self._dataset_path / "data.yaml"
        self._config = self._load_config()
        raw_names = self._config["names"]
        self._original_names = list(raw_names) if isinstance(raw_names, list) else list(raw_names.values())

    def edit(self):
        rename_map = self._collect_renames()

        if not rename_map:
            print("No changes requested.")
            return

        merged_names = self._build_merged_name_list(rename_map)
        index_remap = self._build_index_remap(rename_map, merged_names)

        self._update_label_files(index_remap)
        self._update_config(merged_names)

        print(f"\nDone. {len(self._original_names)} labels -> {len(merged_names)} labels.")

    def _load_config(self) -> dict:
        with open(self._config_path) as file:
            return yaml.safe_load(file)

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

    def _update_label_files(self, index_remap: dict[int, int]):
        if not index_remap:
            return

        label_directories = self._find_label_directories()

        for label_directory in label_directories:
            for label_file in label_directory.glob("*.txt"):
                self._remap_single_file(label_file, index_remap)

    def _find_label_directories(self) -> list[Path]:
        label_directories = []

        for key in ("train", "val", "test"):
            if key not in self._config:
                continue
            image_path = Path(self._config[key])
            if not image_path.is_absolute():
                image_path = self._dataset_path / image_path

            candidate = Path(str(image_path).replace("images", "labels"))
            if candidate.is_dir():
                label_directories.append(candidate)

        if not label_directories:
            for candidate in self._dataset_path.rglob("labels"):
                if candidate.is_dir():
                    label_directories.append(candidate)

        return label_directories

    def _remap_single_file(self, file_path: Path, index_remap: dict[int, int]):
        lines = file_path.read_text().strip().splitlines()
        updated_lines = []

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            old_class_index = int(parts[0])
            parts[0] = str(index_remap.get(old_class_index, old_class_index))
            updated_lines.append(" ".join(parts))

        file_path.write_text("\n".join(updated_lines) + "\n" if updated_lines else "")

    def _update_config(self, merged_names: list[str]):
        self._config["names"] = {index: name for index, name in enumerate(merged_names)}
        self._config["nc"] = len(merged_names)

        with open(self._config_path, "w") as file:
            yaml.dump(self._config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
