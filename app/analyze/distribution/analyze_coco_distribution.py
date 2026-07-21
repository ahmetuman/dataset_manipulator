from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from tabulate import tabulate

from app.utils.coco_files import find_annotation_file
from app.utils.coco_files import load_coco

SPLIT_NAMES = ["train", "valid", "val", "test"]
BAR_COLOR = "#636EFA"


class CocoDistributionAnalyzer:
    def __init__(self, dataset_root: str, output_directory: str = "distribution_analysis"):
        self.dataset_root = Path(dataset_root)
        self.output_directory = Path(output_directory)
        self.split_dirs = self._discover_splits()
        self.categories = self._load_categories()

    def _discover_splits(self) -> list[str]:
        found_splits = []
        for candidate in SPLIT_NAMES:
            split_dir = self.dataset_root / candidate
            if split_dir.is_dir() and find_annotation_file(split_dir):
                found_splits.append(candidate)
        if not found_splits:
            raise FileNotFoundError(f"No splits with annotation files found in {self.dataset_root}")
        return found_splits

    def _load_categories(self) -> dict[int, str]:
        data = load_coco(find_annotation_file(self.dataset_root / self.split_dirs[0]))
        return {category["id"]: category["name"] for category in data["categories"]}

    def _count_split(self, split: str) -> tuple[Counter, int]:
        data = load_coco(find_annotation_file(self.dataset_root / split))
        counts = Counter()
        for annotation in data.get("annotations", []):
            counts[annotation["category_id"]] += 1
        return counts, len(data.get("images", []))

    def _category_name(self, category_id: int) -> str:
        return self.categories.get(category_id, f"category_{category_id}")

    def _all_category_ids(self, counts_per_split: dict[str, Counter]) -> list[int]:
        ids = set(self.categories.keys())
        for counts in counts_per_split.values():
            ids.update(counts.keys())
        return sorted(ids)

    def _print_distribution_table(self, counts_per_split: dict[str, Counter]):
        split_totals = {split: sum(counts.values()) for split, counts in counts_per_split.items()}

        print(f"\n  Dataset: {self.dataset_root.name}")
        print(f"  Splits:  {', '.join(self.split_dirs)}\n")

        headers = ["ID", "Category"] + self.split_dirs + ["TOTAL"]
        rows = []
        for category_id in self._all_category_ids(counts_per_split):
            row = [category_id, self._category_name(category_id)]
            category_total = 0
            for split in self.split_dirs:
                count = counts_per_split[split].get(category_id, 0)
                category_total += count
                split_total = split_totals[split]
                percentage = (count / split_total * 100) if split_total else 0.0
                row.append(f"{count} ({percentage:.1f}%)")
            row.append(category_total)
            rows.append(row)

        total_row = ["", "TOTAL"] + [split_totals[split] for split in self.split_dirs]
        total_row.append(sum(split_totals.values()))
        rows.append(total_row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))

    def _print_split_summary(self, counts_per_split: dict[str, Counter], images_per_split: dict[str, int]):
        headers = ["Split", "Images", "Annotations", "Avg/Image", "Categories"]
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
        sorted_categories = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        labels = [self._category_name(category_id) for category_id, _ in sorted_categories]
        values = [count for _, count in sorted_categories]
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
        axis.set_xlabel("Category")
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
