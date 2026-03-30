from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class YOLOGraphDrawer:
    def __init__(self, dataset_path, output_directory="dataset_analysis"):
        self.dataset_path = Path(dataset_path)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _parse_label_file(self, filepath):
        annotations = []
        with open(filepath) as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue
                if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0
                        and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    print(f"Warning: skipping out-of-range annotation in {filepath}: {line.strip()}")
                    continue
                annotations.append({
                    "class_id": class_id,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                })
        return annotations

    def _load_class_names(self):
        yaml_candidates = list(self.dataset_path.glob("*.yaml")) + list(self.dataset_path.glob("*.yml"))
        for yaml_path in yaml_candidates:
            try:
                import yaml
                with open(yaml_path) as file:
                    data = yaml.safe_load(file)
                if "names" in data:
                    if isinstance(data["names"], dict):
                        return data["names"]
                    elif isinstance(data["names"], list):
                        return {i: name for i, name in enumerate(data["names"])}
            except Exception:
                continue
        return None

    def _collect_all_annotations(self):
        label_directories = []
        for dirpath, _, filenames in os.walk(self.dataset_path):
            if any(filename.endswith(".txt") for filename in filenames):
                path = Path(dirpath)
                if "labels" in path.parts:
                    label_directories.append(path)

        if not label_directories:
            for dirpath, _, filenames in os.walk(self.dataset_path):
                if any(filename.endswith(".txt") for filename in filenames):
                    label_directories.append(Path(dirpath))

        all_annotations = []
        boxes_per_image = []

        for label_directory in label_directories:
            for label_file in sorted(label_directory.glob("*.txt")):
                if label_file.name == "classes.txt":
                    continue
                annotations = self._parse_label_file(label_file)
                all_annotations.extend(annotations)
                boxes_per_image.append(len(annotations))

        return all_annotations, boxes_per_image

    def _create_label_distribution_chart(self, class_counts, class_names):
        total = sum(class_counts.values())
        sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

        labels = [class_names.get(class_id, f"class_{class_id}") for class_id, _ in sorted_classes]
        counts = [count for _, count in sorted_classes]
        percentages = [count / total * 100 for count in counts]

        figure, axis = plt.subplots(figsize=(12, 6))
        bar_positions = range(len(labels))
        bars = axis.bar(bar_positions, counts, color="#636EFA")

        for bar, percentage in zip(bars, percentages):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{percentage: .1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        axis.set_xticks(bar_positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_title(f"Label Distribution (Total: {total} annotations)")
        axis.set_xlabel("Class")
        axis.set_ylabel("Count")
        figure.tight_layout()
        return figure

    def _create_bbox_area_histogram(self, areas):
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.hist(areas, bins=50, color="#EF553B", edgecolor="white")
        axis.set_title(f"Bounding Box Area Distribution (normalized w x h, n={len(areas)})")
        axis.set_xlabel("Bounding Box Area (width x height, normalized)")
        axis.set_ylabel("Frequency")
        figure.tight_layout()
        return figure

    def _create_boxes_per_image_histogram(self, boxes_per_image):
        average_boxes = sum(boxes_per_image) / len(boxes_per_image)
        bin_count = min(50, max(boxes_per_image) - min(boxes_per_image) + 1)

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.hist(boxes_per_image, bins=bin_count, color="#00CC96", edgecolor="white")
        axis.set_title(f"Boxes Per Image (n={len(boxes_per_image)} images, avg={average_boxes: .1f})")
        axis.set_xlabel("Number of Boxes")
        axis.set_ylabel("Number of Images")
        figure.tight_layout()
        return figure

    def _create_bbox_position_heatmap(self, all_annotations, grid_resolution=100):
        heatmap_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)

        for annotation in all_annotations:
            center_x = annotation["center_x"]
            center_y = annotation["center_y"]
            half_width = annotation["width"] / 2
            half_height = annotation["height"] / 2

            x_min = max(0.0, center_x - half_width)
            y_min = max(0.0, center_y - half_height)
            x_max = min(1.0, center_x + half_width)
            y_max = min(1.0, center_y + half_height)

            column_start = min(int(x_min * grid_resolution), grid_resolution - 1)
            column_end = min(int(x_max * grid_resolution), grid_resolution - 1)
            row_start = min(int(y_min * grid_resolution), grid_resolution - 1)
            row_end = min(int(y_max * grid_resolution), grid_resolution - 1)

            heatmap_grid[row_start:row_end + 1, column_start:column_end + 1] += 1

        figure, axis = plt.subplots(figsize=(7, 7))
        image = axis.imshow(
            heatmap_grid,
            cmap="hot_r",
            extent=[0, 1, 1, 0],
            aspect="equal",
        )
        figure.colorbar(image, ax=axis, label="Box Count")
        axis.set_title(f"Bounding Box Position Heatmap (n={len(all_annotations)} boxes)")
        axis.set_xlabel("Image X (normalized)")
        axis.set_ylabel("Image Y (normalized)")
        figure.tight_layout()
        return figure

    def draw(self):
        if not self.dataset_path.exists():
            print(f"Error: Path '{self.dataset_path}' does not exist.")
            sys.exit(1)

        print(f"Scanning dataset at: {self.dataset_path}")

        class_names = self._load_class_names()
        if class_names:
            print(f"Found class names: {class_names}")
        else:
            print("No YAML with class names found. Using class IDs as labels.")
            class_names = {}

        all_annotations, boxes_per_image = self._collect_all_annotations()

        if not all_annotations:
            print("No annotations found. Check that your dataset has .txt label files.")
            sys.exit(1)

        print(f"Found {len(all_annotations)} annotations across {len(boxes_per_image)} images.")

        class_counts = Counter(annotation["class_id"] for annotation in all_annotations)
        areas = [annotation["width"] * annotation["height"] for annotation in all_annotations]
        # print(sorted(areas)[0])

        label_figure = self._create_label_distribution_chart(class_counts, class_names)
        label_figure.savefig(self.output_directory / "label_distribution.png", dpi=150)
        plt.close(label_figure)

        area_figure = self._create_bbox_area_histogram(areas)
        area_figure.savefig(self.output_directory / "bbox_area_distribution.png", dpi=150)
        plt.close(area_figure)

        if boxes_per_image:
            boxes_figure = self._create_boxes_per_image_histogram(boxes_per_image)
            boxes_figure.savefig(self.output_directory / "boxes_per_image.png", dpi=150)
            plt.close(boxes_figure)

        heatmap_figure = self._create_bbox_position_heatmap(all_annotations)
        heatmap_figure.savefig(self.output_directory / "bbox_position_heatmap.png", dpi=150)
        plt.close(heatmap_figure)
