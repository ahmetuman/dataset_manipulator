from __future__ import annotations

from app.analyze.validate.find_similar_images import SimilarityCalculator


class DatasetValidator:
    def __init__(self, root_directory, test_run):
        self.similarity_calculator = SimilarityCalculator(root_directory, test_run)

    def validate(self):
        self.similarity_calculator.calculate_similarity()
