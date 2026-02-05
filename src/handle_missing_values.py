"""Missing value imputation"""
import pandas as pd


class MissingValueHandler:
    """Handle missing values in datasets"""

    def handle_missing(self, df):
        """Impute or remove missing values"""
        return df.dropna()
