from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image
from tabulate import tabulate

from app.utils.image_files import find_matching_image
from app.utils.image_files import is_image_file
from app.utils.yaml_config import find_yaml_file
from app.utils.yaml_config import load_class_names
from app.utils.yaml_config import load_yaml_config

SPLIT_NAMES = ["train", "valid", "val", "test"]
YOLO_DETECTION_FIELDS_COUNT = 5
MAX_REPORTED_ITEMS = 20


class YoloDatasetDoctor:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.yaml_path = find_yaml_file(self.dataset_root)
        self.class_names = load_class_names(self.yaml_path) if self.yaml_path else {}
        self.split_dirs = self._discover_splits()

    def _discover_splits(self) -> list[str]:
        return [split for split in SPLIT_NAMES if (self.dataset_root / split).is_dir()]

    def _iter_split_layout(self):
        for split in self.split_dirs:
            images_dir = self.dataset_root / split / "images"
            labels_dir = self.dataset_root / split / "labels"
            yield split, images_dir, labels_dir

    def _label_files(self, labels_dir: Path) -> list[Path]:
        return [path for path in sorted(labels_dir.glob("*.txt")) if path.name != "classes.txt"]

    def _label_line_issue(self, parts: list[str]) -> str | None:
        if len(parts) != YOLO_DETECTION_FIELDS_COUNT:
            return "wrong field count"
        try:
            class_id = int(parts[0])
        except ValueError:
            return "non-integer class id"
        try:
            center_x, center_y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            return "non-float coordinate"
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            return "center out of [0, 1]"
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            return "width/height out of (0, 1]"
        if self.class_names and class_id not in self.class_names:
            return f"unknown class id {class_id}"
        return None

    def _check_config(self) -> list[str]:
        if self.yaml_path is None:
            return ["no *.yaml file found in dataset root"]
        if not self.class_names:
            return [f"could not parse class names from {self.yaml_path.name}"]

        issues = []
        declared_nc = load_yaml_config(self.yaml_path).get("nc")
        if declared_nc is not None and declared_nc != len(self.class_names):
            issues.append(f"nc={declared_nc} but names has {len(self.class_names)} entries")
        return issues

    def _check_structure(self) -> list[str]:
        issues = []
        for split, images_dir, labels_dir in self._iter_split_layout():
            if not images_dir.is_dir():
                issues.append(f"{split}: missing images/ directory")
            if not labels_dir.is_dir():
                issues.append(f"{split}: missing labels/ directory")
        return issues

    def _check_orphans(self) -> tuple[list[str], list[str]]:
        orphan_images = []
        orphan_labels = []
        for split, images_dir, labels_dir in self._iter_split_layout():
            if not images_dir.is_dir() or not labels_dir.is_dir():
                continue
            for image_path in sorted(images_dir.iterdir()):
                if image_path.is_file() and is_image_file(image_path):
                    if not (labels_dir / (image_path.stem + ".txt")).exists():
                        orphan_images.append(f"{split}/images/{image_path.name}")
            for label_file in self._label_files(labels_dir):
                if find_matching_image(images_dir, label_file.stem) is None:
                    orphan_labels.append(f"{split}/labels/{label_file.name}")
        return orphan_images, orphan_labels

    def _check_empty_labels(self) -> list[str]:
        issues = []
        for split, _, labels_dir in self._iter_split_layout():
            if not labels_dir.is_dir():
                continue
            for label_file in self._label_files(labels_dir):
                if not label_file.read_text().strip():
                    issues.append(f"{split}/labels/{label_file.name}")
        return issues

    def _check_corrupt_images(self) -> list[str]:
        issues = []
        for split, images_dir, _ in self._iter_split_layout():
            if not images_dir.is_dir():
                continue
            for image_path in sorted(images_dir.iterdir()):
                if not (image_path.is_file() and is_image_file(image_path)):
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    issues.append(f"{split}/images/{image_path.name}: {error}")
        return issues

    def _scan_labels(self) -> tuple[list[str], Counter]:
        issues = []
        class_usage = Counter()
        for split, _, labels_dir in self._iter_split_layout():
            if not labels_dir.is_dir():
                continue
            for label_file in self._label_files(labels_dir):
                for line_number, raw_line in enumerate(label_file.read_text().splitlines(), start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    issue = self._label_line_issue(parts)
                    if issue:
                        issues.append(f"{split}/labels/{label_file.name}:{line_number} -> {issue}")
                    try:
                        class_usage[int(parts[0])] += 1
                    except ValueError:
                        pass
        return issues, class_usage

    def _check_duplicate_stems(self) -> list[str]:
        stem_to_splits = {}
        for split, images_dir, _ in self._iter_split_layout():
            if not images_dir.is_dir():
                continue
            for image_path in images_dir.iterdir():
                if image_path.is_file() and is_image_file(image_path):
                    stem_to_splits.setdefault(image_path.stem, set()).add(split)
        return [
            f"'{stem}' appears in splits: {', '.join(sorted(splits))}"
            for stem, splits in sorted(stem_to_splits.items())
            if len(splits) > 1
        ]

    def _check_unused_classes(self, class_usage: Counter) -> list[str]:
        return [
            f"class {class_id} '{self.class_names[class_id]}' has 0 annotations"
            for class_id in sorted(self.class_names)
            if class_usage.get(class_id, 0) == 0
        ]

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
        print(f"\n  Dataset Doctor (YOLO): {self.dataset_root.name}")
        print(f"  Path:   {self.dataset_root}")
        print(f"  Splits: {', '.join(self.split_dirs) if self.split_dirs else '(none found)'}\n")

        label_issues, class_usage = self._scan_labels()
        orphan_images, orphan_labels = self._check_orphans()

        sections = [
            ("Config", self._check_config()),
            ("Structure", self._check_structure()),
            ("Orphan images (no label)", orphan_images),
            ("Orphan labels (no image)", orphan_labels),
            ("Empty label files", self._check_empty_labels()),
            ("Corrupt/unreadable images", self._check_corrupt_images()),
            ("Malformed annotations", label_issues),
            ("Duplicate image names across splits", self._check_duplicate_stems()),
            ("Unused classes", self._check_unused_classes(class_usage)),
        ]

        print("  DETAILS:\n")
        counts = {}
        for title, items in sections:
            counts[title] = self._report_section(title, items)

        self._print_summary(counts)
