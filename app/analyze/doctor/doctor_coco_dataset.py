from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image
from tabulate import tabulate

from app.utils.coco_files import ANNOTATIONS_FILENAME
from app.utils.coco_files import find_annotation_file
from app.utils.coco_files import load_coco
from app.utils.image_files import is_image_file

SPLIT_NAMES = ["train", "valid", "val", "test"]
MAX_REPORTED_ITEMS = 20


class CocoDatasetDoctor:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.split_dirs = self._discover_splits()
        self.load_errors = []
        self.split_data = self._load_split_data()

    def _discover_splits(self) -> list[str]:
        return [split for split in SPLIT_NAMES if (self.dataset_root / split).is_dir()]

    def _load_split_data(self) -> dict[str, dict | None]:
        split_data = {}
        for split in self.split_dirs:
            annotation_file = find_annotation_file(self.dataset_root / split)
            if annotation_file is None:
                self.load_errors.append(f"{split}: no {ANNOTATIONS_FILENAME} found")
                split_data[split] = None
                continue
            try:
                split_data[split] = load_coco(annotation_file)
            except Exception as error:
                self.load_errors.append(f"{split}: failed to parse annotation file ({error})")
                split_data[split] = None
        return split_data

    def _iter_loaded_splits(self):
        for split in self.split_dirs:
            data = self.split_data.get(split)
            if data is not None:
                yield split, data

    def _check_config(self) -> list[str]:
        issues = list(self.load_errors)
        for split, data in self._iter_loaded_splits():
            if not data.get("categories"):
                issues.append(f"{split}: annotation file has no categories")
        return issues

    def _check_category_consistency(self) -> list[str]:
        issues = []
        reference = None
        reference_split = None
        for split, data in self._iter_loaded_splits():
            mapping = {category.get("id"): category.get("name") for category in data.get("categories", [])}
            if reference is None:
                reference = mapping
                reference_split = split
                continue
            if mapping != reference:
                issues.append(f"{split}: categories differ from '{reference_split}'")
        return issues

    def _check_missing_image_files(self) -> list[str]:
        issues = []
        for split, data in self._iter_loaded_splits():
            split_dir = self.dataset_root / split
            for image in data.get("images", []):
                file_name = image.get("file_name")
                if file_name is None:
                    issues.append(f"{split}: image entry {image.get('id')} has no file_name")
                elif not (split_dir / file_name).exists():
                    issues.append(f"{split}: listed image missing on disk: {file_name}")
        return issues

    def _check_orphan_image_files(self) -> list[str]:
        issues = []
        for split, data in self._iter_loaded_splits():
            split_dir = self.dataset_root / split
            referenced = {image.get("file_name") for image in data.get("images", [])}
            for path in sorted(split_dir.iterdir()):
                if path.is_file() and is_image_file(path) and path.name not in referenced:
                    issues.append(f"{split}: image on disk not referenced in annotations: {path.name}")
        return issues

    def _check_annotation_references(self) -> list[str]:
        issues = []
        for split, data in self._iter_loaded_splits():
            image_ids = {image.get("id") for image in data.get("images", [])}
            category_ids = {category.get("id") for category in data.get("categories", [])}
            for annotation in data.get("annotations", []):
                image_id = annotation.get("image_id")
                category_id = annotation.get("category_id")
                if image_id is None or category_id is None:
                    issues.append(f"{split}: annotation {annotation.get('id')} missing image_id/category_id")
                    continue
                if image_id not in image_ids:
                    issues.append(f"{split}: annotation {annotation.get('id')} references missing image_id {image_id}")
                if category_id not in category_ids:
                    issues.append(
                        f"{split}: annotation {annotation.get('id')} references missing category_id {category_id}"
                    )
        return issues

    def _check_invalid_bboxes(self) -> list[str]:
        issues = []
        for split, data in self._iter_loaded_splits():
            dimensions = {
                image.get("id"): (image.get("width"), image.get("height"))
                for image in data.get("images", [])
            }
            for annotation in data.get("annotations", []):
                bbox = annotation.get("bbox")
                annotation_id = annotation.get("id")
                if not bbox or len(bbox) != 4:
                    issues.append(f"{split}: annotation {annotation_id} has invalid bbox {bbox}")
                    continue
                x, y, width, height = bbox
                if width <= 0 or height <= 0:
                    issues.append(f"{split}: annotation {annotation_id} has non-positive width/height")
                    continue
                if x < 0 or y < 0:
                    issues.append(f"{split}: annotation {annotation_id} has negative x/y")
                    continue
                image_width, image_height = dimensions.get(annotation.get("image_id"), (None, None))
                if image_width and image_height and (x + width > image_width or y + height > image_height):
                    issues.append(f"{split}: annotation {annotation_id} bbox exceeds image bounds")
        return issues

    def _check_duplicate_ids(self) -> list[str]:
        issues = []
        for split, data in self._iter_loaded_splits():
            image_ids = [image.get("id") for image in data.get("images", [])]
            for image_id, count in Counter(image_ids).items():
                if count > 1:
                    issues.append(f"{split}: duplicate image id {image_id} ({count}x)")
            annotation_ids = [annotation.get("id") for annotation in data.get("annotations", [])]
            for annotation_id, count in Counter(annotation_ids).items():
                if annotation_id is not None and count > 1:
                    issues.append(f"{split}: duplicate annotation id {annotation_id} ({count}x)")
        return issues

    def _check_duplicate_filenames_across_splits(self) -> list[str]:
        name_to_splits = {}
        for split, data in self._iter_loaded_splits():
            for image in data.get("images", []):
                file_name = image.get("file_name")
                if file_name is not None:
                    name_to_splits.setdefault(file_name, set()).add(split)
        return [
            f"'{file_name}' appears in splits: {', '.join(sorted(splits))}"
            for file_name, splits in sorted(name_to_splits.items())
            if len(splits) > 1
        ]

    def _check_unused_categories(self) -> list[str]:
        category_names = {}
        usage = Counter()
        for _, data in self._iter_loaded_splits():
            for category in data.get("categories", []):
                category_names[category.get("id")] = category.get("name")
            for annotation in data.get("annotations", []):
                usage[annotation.get("category_id")] += 1
        return [
            f"category {category_id} '{name}' has 0 annotations"
            for category_id, name in sorted(category_names.items(), key=lambda item: (item[0] is None, item[0]))
            if usage.get(category_id, 0) == 0
        ]

    def _check_corrupt_images(self) -> list[str]:
        issues = []
        for split in self.split_dirs:
            split_dir = self.dataset_root / split
            for path in sorted(split_dir.iterdir()):
                if not (path.is_file() and is_image_file(path)):
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:
                    issues.append(f"{split}/{path.name}: {error}")
        return issues

    def _report_section(self, title: str, items: list[str]) -> int:
        if not items:
            print(f"  [ok] {title}: no issues")
            return 0
        print(f"  [warning] {title}: {len(items)} issue(s)")
        for item in items[:MAX_REPORTED_ITEMS]:
            print(f"      - {item}")
        remaining = len(items) - MAX_REPORTED_ITEMS
        if remaining > 0:
            print(f"      ... and {remaining} more")
        return len(items)

    def _print_summary(self, counts: dict[str, int]):
        total = sum(counts.values())
        rows = [[title, count] for title, count in counts.items()]
        rows.append(["TOTAL", total])

        print("\n  SUMMARY:\n")
        print(tabulate(rows, headers=["Check", "Issues"], tablefmt="simple"))

        if total == 0:
            print("\n  Verdict: healthy - no issues found.")
        else:
            failed_checks = sum(1 for count in counts.values() if count)
            print(f"\n  Verdict: {total} issue(s) found across {failed_checks} check(s).")

    def diagnose(self):
        print(f"\n  Dataset Doctor (COCO): {self.dataset_root.name}")
        print(f"  Path:   {self.dataset_root}")
        print(f"  Splits: {', '.join(self.split_dirs) if self.split_dirs else '(none found)'}\n")

        sections = [
            ("Config", self._check_config()),
            ("Category consistency across splits", self._check_category_consistency()),
            ("Listed images missing on disk", self._check_missing_image_files()),
            ("Orphan images (not in annotations)", self._check_orphan_image_files()),
            ("Broken annotation references", self._check_annotation_references()),
            ("Invalid bounding boxes", self._check_invalid_bboxes()),
            ("Duplicate ids", self._check_duplicate_ids()),
            ("Duplicate image names across splits", self._check_duplicate_filenames_across_splits()),
            ("Unused categories", self._check_unused_categories()),
            ("Corrupt/unreadable images", self._check_corrupt_images()),
        ]

        print("  DETAILS:\n")
        counts = {}
        for title, items in sections:
            counts[title] = self._report_section(title, items)

        self._print_summary(counts)
