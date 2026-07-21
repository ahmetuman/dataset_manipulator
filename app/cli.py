from __future__ import annotations

import sys

import fire

from app.alter.edit.edit_coco_dataset import CocoLabelEditor
from app.alter.edit.edit_yolo_dataset import YoloLabelEditor
from app.alter.merge.merge_coco_dataset import CocoDatasetMerger
from app.alter.merge.merge_yolo_dataset import YoloDatasetMerger
from app.alter.remove.remove_from_coco_dataset import CocoLabelRemover
from app.alter.remove.remove_from_yolo_dataset import YoloLabelRemover
from app.analyze.distribution.analyze_coco_distribution import CocoDistributionAnalyzer
from app.analyze.distribution.analyze_yolo_distribution import YoloDistributionAnalyzer
from app.analyze.doctor.doctor_coco_dataset import CocoDatasetDoctor
from app.analyze.doctor.doctor_yolo_dataset import YoloDatasetDoctor
from app.analyze.validate.validate_dataset import DatasetValidator
from app.analyze.visualize.draw_graphs.yolo_graphs import YOLOGraphDrawer
from app.analyze.visualize.visualize_coco_dataset import CocoDatasetVisualizer
from app.analyze.visualize.visualize_yolo_dataset import YoloDatasetVisualizer
from app.convert.convert_yolo_to_coco import YOLOtoCOCOConverter

VERSION = "0.0.2"


class YOLO:
    def __init__(self):
        pass

    def visualize(self, dataset_directory_path, detailed_mode: bool = False):
        yolo_dataset_visualizer = YoloDatasetVisualizer(dataset_directory_path, detailed_mode)
        yolo_dataset_visualizer.visualize_all_splits()

    def draw(self, dataset_directory_path):
        yolo_grap_drawer = YOLOGraphDrawer(dataset_directory_path)
        yolo_grap_drawer.draw()

    def distribution(self, dataset_directory_path):
        yolo_distribution_analyzer = YoloDistributionAnalyzer(dataset_directory_path)
        yolo_distribution_analyzer.analyze()

    def doctor(self, dataset_directory_path):
        yolo_dataset_doctor = YoloDatasetDoctor(dataset_directory_path)
        yolo_dataset_doctor.diagnose()

    def merge(self, datasets_root_directory, output_directory="merged_dataset"):
        yolo_dataset_merger = YoloDatasetMerger(datasets_root_directory, output_directory)
        yolo_dataset_merger.merge()

    def remove(self, dataset_directory_path):
        yolo_dataset_remover = YoloLabelRemover(dataset_directory_path)
        yolo_dataset_remover.remove()

    def edit(self, dataset_directory_path):
        yolo_label_editor = YoloLabelEditor(dataset_directory_path)
        yolo_label_editor.edit()

    def convert(self, dataset_directory_path):
        yolo_to_coco_converter = YOLOtoCOCOConverter(dataset_directory_path)
        yolo_to_coco_converter.convert()


class COCO:
    def __init__(self):
        pass

    def visualize(self, dataset_directory_path, detailed_mode: bool = False):
        coco_dataset_visualizer = CocoDatasetVisualizer(dataset_directory_path, detailed_mode)
        coco_dataset_visualizer.visualize_all_splits()

    def distribution(self, dataset_directory_path):
        coco_distribution_analyzer = CocoDistributionAnalyzer(dataset_directory_path)
        coco_distribution_analyzer.analyze()

    def doctor(self, dataset_directory_path):
        coco_dataset_doctor = CocoDatasetDoctor(dataset_directory_path)
        coco_dataset_doctor.diagnose()

    def merge(self, datasets_root_directory, output_directory="merged_dataset"):
        yolo_dataset_merger = CocoDatasetMerger(datasets_root_directory, output_directory)
        yolo_dataset_merger.merge()

    def remove(self, dataset_directory_path):
        coco_dataset_remover = CocoLabelRemover(dataset_directory_path)
        coco_dataset_remover.remove()

    def edit(self, dataset_directory_path):
        coco_label_editor = CocoLabelEditor(dataset_directory_path)
        coco_label_editor.edit()


class app:
    def __init__(self):
        self.yolo = YOLO()
        self.coco = COCO()

    def version(self):
        print(VERSION)

    def validate(self, images_directory_path, test_run: bool = True):
        dataset_validator = DatasetValidator(images_directory_path, test_run)
        dataset_validator.validate()

    def help(self):
        print("\n--- HOW TO USE THIS APP ---\n")

        print("$ app {dataset format (either YOLO or COCO or nothing)} "
              "{desired function} {parameters of function if needed} {input dataset(s) path}")

        print('e.g. $ app yolo visualize "datasets/colored_yolo_dataset"\n')

        print("\nCHECK README.md FOR ALL EXISTING FUNCTIONS AND THEIR EXPLANATIONS\n")

def main(args=None):
    fire.Fire(app)


if __name__ == "__main__":
    sys.exit(main())
