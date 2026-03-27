from __future__ import annotations

import sys
import fire

from app.analyze.visualize.visualize_yolo_dataset import YoloDatasetVisualizer
from app.analyze.visualize.visualize_coco_dataset import CocoDatasetVisualizer

from app.alter.merge.merge_yolo_dataset import YoloDatasetMerger
from app.alter.merge.merge_coco_dataset import CocoDatasetMerger

from app.alter.remove.remove_from_yolo_dataset import YoloLabelRemover
from app.alter.remove.remove_from_coco_dataset import CocoLabelRemover

from app.alter.edit.edit_yolo_dataset import YoloLabelEditor


VERSION = "0.0.1"

class YOLO:
    def __init__(self):
        pass

    def visualize(self, dataset_directory_path):
        yolo_dataset_visualizer = YoloDatasetVisualizer(dataset_directory_path) 
        yolo_dataset_visualizer.visualize_all_splits()

    def merge(self, datasets_root_directory, output_directory = "merged_dataset"):
        yolo_dataset_merger = YoloDatasetMerger(datasets_root_directory, output_directory)
        yolo_dataset_merger.merge()

    def remove(self, dataset_directory_path):
        yolo_dataset_remover = YoloLabelRemover(dataset_directory_path)
        yolo_dataset_remover.remove()

    def edit(self, dataset_directory_path):
        yolo_label_editor = YoloLabelEditor(dataset_directory_path)
        yolo_label_editor.edit()

class COCO:
    def __init__(self):
        pass

    def visualize(self, dataset_directory_path):
        coco_dataset_visualizer = CocoDatasetVisualizer(dataset_directory_path)
        coco_dataset_visualizer.visualize_all_splits()

    def merge(self, datasets_root_directory, output_directory = "merged_dataset"):
        yolo_dataset_merger = CocoDatasetMerger(datasets_root_directory, output_directory)
        yolo_dataset_merger.merge()

    def remove(self, dataset_directory_path):
        yolo_dataset_remover = CocoLabelRemover(dataset_directory_path)
        yolo_dataset_remover.remove()

class app:
    def __init__(self):
        self.yolo = YOLO()
        self.coco = COCO()

    def version(self):
        print(VERSION)

    def help(self):
        print("\n--- HOW TO USE THIS APP ---\n")

        print("$ app {dataset format (either YOLO or COCO)} {desired function} {parameters of function if needed} {input dataset(s) path}\n")

        print(" -> Feature 1: Visualize")
        print("... \n" \
        "...\n" \
        "...")


def main(args=None):
    fire.Fire(app)


if __name__ == "__main__":
    sys.exit(main())