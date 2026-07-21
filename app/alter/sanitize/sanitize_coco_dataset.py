from __future__ import annotations

from collections import Counter
from pathlib import Path

from tabulate import tabulate

from app.utils.coco_files import find_annotation_file
from app.utils.coco_files import load_coco
from app.utils.coco_files import save_coco

SPLIT_NAMES = ["train", "valid", "val", "test"]


class CocoDatasetSanitizer:
    def __init__(self, dataset_root: str, test_run: bool = True):
        self.dataset_root = Path(dataset_root)
        self.test_run = test_run
        self.split_dirs = self._discover_splits()

    def _discover_splits(self) -> list[str]:
        found_splits = []
        for split in SPLIT_NAMES:
            split_dir = self.dataset_root / split
            if split_dir.is_dir() and find_annotation_file(split_dir):
                found_splits.append(split)
        if not found_splits:
            raise FileNotFoundError(f"No splits with annotation files found in {self.dataset_root}")
        return found_splits

    def _dedup_images(self, images: list[dict], stats: Counter) -> list[dict]:
        unique_images = []
        seen_image_ids = set()
        for image in images:
            image_id = image.get("id")
            if image_id in seen_image_ids:
                stats["duplicate_images"] += 1
                continue
            seen_image_ids.add(image_id)
            unique_images.append(image)
        return unique_images

    def _sanitize_annotations(self, data: dict, images: list[dict], stats: Counter) -> list[dict]:
        image_ids = {image.get("id") for image in images}
        category_ids = {category.get("id") for category in data.get("categories", [])}
        dimensions = {
            image.get("id"): (image.get("width"), image.get("height"))
            for image in images
        }

        kept_annotations = []
        seen_boxes = set()
        for annotation in data.get("annotations", []):
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            if image_id not in image_ids or category_id not in category_ids:
                stats["broken_refs"] += 1
                continue

            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                stats["degenerate"] += 1
                continue
            try:
                x, y, width, height = (float(value) for value in bbox)
            except (TypeError, ValueError):
                stats["degenerate"] += 1
                continue

            new_x = max(0.0, x)
            new_y = max(0.0, y)
            new_width = width
            new_height = height
            image_width, image_height = dimensions.get(image_id, (None, None))
            if image_width:
                new_width = min(new_width, image_width - new_x)
            if image_height:
                new_height = min(new_height, image_height - new_y)

            if new_width <= 0 or new_height <= 0:
                stats["degenerate"] += 1
                continue

            box_key = (image_id, category_id,
                       round(new_x, 3), round(new_y, 3), round(new_width, 3), round(new_height, 3))
            if box_key in seen_boxes:
                stats["duplicate"] += 1
                continue
            seen_boxes.add(box_key)

            new_annotation = dict(annotation)
            if (new_x, new_y, new_width, new_height) != (x, y, width, height):
                new_annotation["bbox"] = [round(new_x, 2), round(new_y, 2), round(new_width, 2), round(new_height, 2)]
                new_annotation["area"] = round(new_width * new_height, 2)
                stats["clamped"] += 1
            kept_annotations.append(new_annotation)

        return kept_annotations

    def _reassign_ids_if_needed(self, annotations: list[dict], stats: Counter):
        ids = [annotation.get("id") for annotation in annotations]
        if len(ids) != len(set(ids)):
            for new_id, annotation in enumerate(annotations, start=1):
                annotation["id"] = new_id
            stats["reassigned_ids"] += 1

    def _sanitize_split(self, split: str, stats: Counter):
        annotation_file = find_annotation_file(self.dataset_root / split)
        data = load_coco(annotation_file)

        original_images = data.get("images", [])
        original_annotations = data.get("annotations", [])

        images = self._dedup_images(original_images, stats)
        annotations = self._sanitize_annotations(data, images, stats)
        self._reassign_ids_if_needed(annotations, stats)

        if images != original_images or annotations != original_annotations:
            stats["files_changed"] += 1
            if not self.test_run:
                data["images"] = images
                data["annotations"] = annotations
                save_coco(annotation_file, data)

    def _print_report(self, stats: Counter):
        rows = [
            ["Duplicate image entries removed", stats["duplicate_images"]],
            ["Annotations with broken refs removed", stats["broken_refs"]],
            ["Invalid/degenerate bboxes removed", stats["degenerate"]],
            ["Duplicate annotations removed", stats["duplicate"]],
            ["Bboxes clamped to image bounds", stats["clamped"]],
            ["Splits with reassigned annotation ids", stats["reassigned_ids"]],
            ["Annotation files changed", f"{stats['files_changed']}/{stats['files_total']}"],
        ]

        print(tabulate(rows, headers=["Fix", "Count"], tablefmt="simple"))

        if self.test_run:
            print("\n  Dry run complete. Re-run with --test_run False to apply changes.")
        else:
            print("\n  Done. Changes written.")

    def sanitize(self):
        mode = "DRY RUN (no files will be modified)" if self.test_run else "APPLYING CHANGES"
        print(f"\n  Sanitize (COCO): {self.dataset_root.name}")
        print(f"  Splits: {', '.join(self.split_dirs)}")
        print(f"  Mode:   {mode}\n")

        stats = Counter()
        for split in self.split_dirs:
            stats["files_total"] += 1
            self._sanitize_split(split, stats)

        self._print_report(stats)
