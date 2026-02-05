"""Model training pipeline"""
from src.model_building import ModelBuilder
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator


class TrainingPipeline:
    """Model training pipeline"""

    def __init__(self):
        self.builder = ModelBuilder()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def run(self):
        """Execute training pipeline"""
        print("Running training pipeline...")
        # Add training logic
