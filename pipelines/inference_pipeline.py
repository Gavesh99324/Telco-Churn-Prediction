"""Batch inference pipeline"""
from src.model_inference import ModelInference


class InferencePipeline:
    """Batch inference pipeline"""

    def __init__(self):
        self.inference = ModelInference()

    def run(self):
        """Execute inference pipeline"""
        print("Running inference pipeline...")
        # Add inference logic
