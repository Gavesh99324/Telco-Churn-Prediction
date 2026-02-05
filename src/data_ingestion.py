"""Data ingestion module"""
import pandas as pd


class DataIngestion:
    """Handle data loading from various sources"""

    def __init__(self, data_path='data/ChurnModelling.csv'):
        self.data_path = data_path

    def load_data(self):
        """Load data from CSV"""
        return pd.read_csv(self.data_path)
