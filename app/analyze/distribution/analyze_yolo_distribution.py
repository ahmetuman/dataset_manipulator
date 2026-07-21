from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from tabulate import tabulate

from app.utils.yaml_config import find_yaml_file
from app.utils.yaml_config import load_class_names

SPLIT_NAMES = ["train", "valid", "val", "test"]
BAR_COLOR = "#636EFA"


class YoloDistributionAnalyzer:
    def __init__(self, dataset_root: str, output_directory: str = "distribution_analysis"):
        self.dataset_root = Path(dataset_root)
        self.output_directory = Path(output_directory)
        self.class_names = self._load_class_names()
        self.split_dirs = self._discover_splits()

    def _load_class_names(self) -> dict[int, str]:
        yaml_path = find_yaml_file(self.dataset_root)
        if yaml_path is None:
            raise FileNotFoundError(f"No YAML file found in {self.dataset_root}")
        class_names = load_class_names(yaml_path)
        if not class_names:
            raise ValueError(f"Could not parse class names from {yaml_path}")
        return class_names

    def _discover_splits(self) -> list[str]:
        found_splits = []
        for candidate in SPLIT_NAMES:
            if (self.dataset_root / candidate / "labels").is_dir():
                found_splits.append(candidate)
        if not found_splits:
            raise FileNotFoundError(f"No splits with a labels/ directory found in {self.dataset_root}")
        return found_splits

    def _count_split(self, split: str) -> tuple[Counter, int]:
        counts = Counter()
        image_count = 0
        for label_file in (self.dataset_root / split / "labels").glob("*.txt"):
            image_count += 1
            for line in label_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
        return counts, image_count

    def _class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"class_{class_id}")

    def _all_class_ids(self, counts_per_split: dict[str, Counter]) -> list[int]:
        ids = set(self.class_names.keys())
        for counts in counts_per_split.values():
            ids.update(counts.keys())
        return sorted(ids)

    def _print_distribution_table(self, counts_per_split: dict[str, Counter]):
        split_totals = {split: sum(counts.values()) for split, counts in counts_per_split.items()}

        print(f"\n  Dataset: {self.dataset_root.name}")
        print(f"  Splits:  {', '.join(self.split_dirs)}\n")

        headers = ["ID", "Class"] + self.split_dirs + ["TOTAL"]
        rows = []
        for class_id in self._all_class_ids(counts_per_split):
            row = [class_id, self._class_name(class_id)]
            class_total = 0
            for split in self.split_dirs:
                count = counts_per_split[split].get(class_id, 0)
                class_total += count
                split_total = split_totals[split]
                percentage = (count / split_total * 100) if split_total else 0.0
                row.append(f"{count} ({percentage:.1f}%)")
            row.append(class_total)
            rows.append(row)

        total_row = ["", "TOTAL"] + [split_totals[split] for split in self.split_dirs]
        total_row.append(sum(split_totals.values()))
        rows.append(total_row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))

    def _print_split_summary(self, counts_per_split: dict[str, Counter], images_per_split: dict[str, int]):
        headers = ["Split", "Images", "Annotations", "Avg/Image", "Classes"]
        rows = []
        for split in self.split_dirs:
            counts = counts_per_split[split]
            images = images_per_split[split]
            annotations = sum(counts.values())
            average = (annotations / images) if images else 0.0
            rows.append([split, images, annotations, f"{average:.2f}", len(counts)])

        print("\n")
        print(tabulate(rows, headers=headers, tablefmt="simple"))

    def _save_split_chart(self, split: str, counts: Counter):
        if not counts:
            print(f"  [{split}] no annotations, skipping chart.")
            return

        total = sum(counts.values())
        sorted_classes = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        labels = [self._class_name(class_id) for class_id, _ in sorted_classes]
        values = [count for _, count in sorted_classes]
        percentages = [count / total * 100 for count in values]

        figure, axis = plt.subplots(figsize=(12, 6))
        bar_positions = range(len(labels))
        bars = axis.bar(bar_positions, values, color=BAR_COLOR)

        for bar, percentage in zip(bars, percentages):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        axis.set_xticks(bar_positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_title(f"[{split}] Label Distribution (Total: {total} annotations)")
        axis.set_xlabel("Class")
        axis.set_ylabel("Count")
        figure.tight_layout()
        figure.savefig(self.output_directory / f"label_distribution_{split}.png", dpi=150)
        plt.close(figure)

    def analyze(self):
        counts_per_split = {}
        images_per_split = {}
        for split in self.split_dirs:
            counts, image_count = self._count_split(split)
            counts_per_split[split] = counts
            images_per_split[split] = image_count

        self._print_distribution_table(counts_per_split)
        self._print_split_summary(counts_per_split, images_per_split)

        self.output_directory.mkdir(parents=True, exist_ok=True)
        for split in self.split_dirs:
            self._save_split_chart(split, counts_per_split[split])

        print(f"\n  Per-split charts saved to: {self.output_directory}/")
