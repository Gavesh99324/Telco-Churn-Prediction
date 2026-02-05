"""Outlier detection and handling"""
import pandas as pd


class OutlierDetector:
    """Detect and handle outliers"""

    def detect_outliers(self, df, column):
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
