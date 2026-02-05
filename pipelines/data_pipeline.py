"""Data preprocessing pipeline"""
from src.data_ingestion import DataIngestion
from src.handle_missing_values import MissingValueHandler
from src.outlier_detection import OutlierDetector
from src.feature_scaling import FeatureScaler
from src.feature_encoding import FeatureEncoder
from src.feature_binning import FeatureBinner
from src.data_splitter import DataSplitter


class DataPipeline:
    """Data preprocessing pipeline"""

    def __init__(self):
        self.ingestion = DataIngestion()
        self.missing_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector()
        self.scaler = FeatureScaler()
        self.encoder = FeatureEncoder()
        self.binner = FeatureBinner()
        self.splitter = DataSplitter()

    def run(self):
        """Execute data pipeline"""
        print("Running data pipeline...")
        # Add pipeline execution logic
