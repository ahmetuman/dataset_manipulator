from __future__ import annotations

from collections import Counter
from pathlib import Path

from tabulate import tabulate

from app.utils.yaml_config import find_yaml_file
from app.utils.yaml_config import load_class_names

SPLIT_NAMES = ["train", "valid", "val", "test"]
YOLO_DETECTION_FIELDS_COUNT = 5
MIN_NORMALIZED_SIDE = 1e-6


class YoloDatasetSanitizer:
    def __init__(self, dataset_root: str, test_run: bool = True):
        self.dataset_root = Path(dataset_root)
        self.test_run = test_run
        self.yaml_path = find_yaml_file(self.dataset_root)
        self.class_names = load_class_names(self.yaml_path) if self.yaml_path else {}
        self.split_dirs = self._discover_splits()

    def _discover_splits(self) -> list[str]:
        found_splits = [split for split in SPLIT_NAMES if (self.dataset_root / split / "labels").is_dir()]
        if not found_splits:
            raise FileNotFoundError(f"No splits with a labels/ directory found in {self.dataset_root}")
        return found_splits

    def _label_files(self, labels_dir: Path) -> list[Path]:
        return [path for path in sorted(labels_dir.glob("*.txt")) if path.name != "classes.txt"]

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _sanitize_line(self, line: str) -> tuple[str | None, str]:
        parts = line.split()
        if len(parts) != YOLO_DETECTION_FIELDS_COUNT:
            return None, "malformed"
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            return None, "malformed"

        if self.class_names and class_id not in self.class_names:
            return None, "unknown_class"

        left = self._clamp01(center_x - width / 2)
        top = self._clamp01(center_y - height / 2)
        right = self._clamp01(center_x + width / 2)
        bottom = self._clamp01(center_y + height / 2)

        new_width = right - left
        new_height = bottom - top
        if new_width <= MIN_NORMALIZED_SIDE or new_height <= MIN_NORMALIZED_SIDE:
            return None, "degenerate"

        new_center_x = (left + right) / 2
        new_center_y = (top + bottom) / 2
        clamped = any(
            abs(new - old) > 1e-9
            for new, old in (
                (new_center_x, center_x),
                (new_center_y, center_y),
                (new_width, width),
                (new_height, height),
            )
        )

        if not clamped:
            return line, "ok"

        new_line = (
            f"{class_id} {new_center_x:.6f} {new_center_y:.6f} "
            f"{new_width:.6f} {new_height:.6f}"
        )
        return new_line, "clamped"

    def _sanitize_file(self, label_file: Path, stats: Counter):
        original_content = label_file.read_text()

        kept_lines = []
        seen = set()
        for raw_line in original_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            new_line, reason = self._sanitize_line(line)
            if new_line is None:
                stats[reason] += 1
                continue
            if new_line in seen:
                stats["duplicate"] += 1
                continue

            seen.add(new_line)
            if reason == "clamped":
                stats["clamped"] += 1
            kept_lines.append(new_line)

        new_content = ("\n".join(kept_lines) + "\n") if kept_lines else ""
        if new_content != original_content:
            stats["files_changed"] += 1
            if not self.test_run:
                label_file.write_text(new_content)

    def _print_report(self, stats: Counter):
        removed = stats["malformed"] + stats["unknown_class"] + stats["degenerate"] + stats["duplicate"]
        rows = [
            ["Malformed lines removed", stats["malformed"]],
            ["Unknown-class annotations removed", stats["unknown_class"]],
            ["Zero-area/degenerate boxes removed", stats["degenerate"]],
            ["Duplicate boxes removed", stats["duplicate"]],
            ["Boxes clamped to [0, 1]", stats["clamped"]],
            ["Total annotations removed", removed],
            ["Label files changed", f"{stats['files_changed']}/{stats['files_total']}"],
        ]

        print(tabulate(rows, headers=["Fix", "Count"], tablefmt="simple"))

        if self.test_run:
            print("\n  Dry run complete. Re-run with --test_run False to apply changes.")
        else:
            print("\n  Done. Changes written.")

    def sanitize(self):
        mode = "DRY RUN (no files will be modified)" if self.test_run else "APPLYING CHANGES"
        print(f"\n  Sanitize (YOLO): {self.dataset_root.name}")
        print(f"  Splits: {', '.join(self.split_dirs)}")
        print(f"  Mode:   {mode}\n")

        stats = Counter()
        for split in self.split_dirs:
            labels_dir = self.dataset_root / split / "labels"
            for label_file in self._label_files(labels_dir):
                stats["files_total"] += 1
                self._sanitize_file(label_file, stats)

        self._print_report(stats)
