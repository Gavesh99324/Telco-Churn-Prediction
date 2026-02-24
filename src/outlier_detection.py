"""Outlier detection and handling for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutlierDetector:
    """Detect and handle outliers using IQR method"""

    def __init__(self, multiplier: float = 1.5):
        """
        Initialize OutlierDetector

        Args:
            multiplier: IQR multiplier for outlier boundaries (default: 1.5)
        """
        self.multiplier = multiplier
        self.outlier_summary = {}

    def detect_outliers_column(self, df: pd.DataFrame, column: str) -> Tuple[pd.Series, Dict]:
        """
        Detect outliers in a single column using IQR method

        Args:
            df: Input DataFrame
            column: Column name to analyze

        Returns:
            Tuple of (boolean mask for outliers, statistics dictionary)
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.multiplier * IQR
        upper_bound = Q3 + self.multiplier * IQR

        # Create boolean mask for outliers
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outlier_mask.sum()

        stats = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': (outlier_count / len(df)) * 100
        }

        return outlier_mask, stats

    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict]:
        """
        Detect outliers in multiple numerical columns

        Args:
            df: Input DataFrame
            columns: List of column names (if None, detect all numerical)

        Returns:
            Dictionary with outlier statistics per column
        """
        logger.info("Detecting outliers using IQR method...")
        logger.info(f"   IQR multiplier: {self.multiplier}")

        if columns is None:
            columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

        outlier_report = {}

        for col in columns:
            if col not in df.columns:
                logger.warning(f"⚠️  Column '{col}' not found, skipping...")
                continue

            outlier_mask, stats = self.detect_outliers_column(df, col)
            outlier_report[col] = stats

            if stats['outlier_count'] > 0:
                logger.info(f"   {col}: {stats['outlier_count']} outliers "
                            f"({stats['outlier_percentage']:.2f}%) "
                            f"[{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
            else:
                logger.info(f"   {col}: No outliers detected")

        self.outlier_summary = outlier_report
        return outlier_report

    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Remove outliers from DataFrame

        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers

        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

        initial_rows = len(df)
        logger.info(f"Checking outliers in {len(columns)} columns...")

        # Create combined mask for all outliers
        combined_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            if col in df.columns:
                outlier_mask, _ = self.detect_outliers_column(df, col)
                combined_mask = combined_mask | outlier_mask

        # Remove rows with any outliers
        df_clean = df[~combined_mask].copy()
        removed_rows = initial_rows - len(df_clean)

        if removed_rows > 0:
            logger.warning(f"⚠️  Removed {removed_rows} rows with outliers "
                           f"({(removed_rows/initial_rows)*100:.2f}%)")
        else:
            logger.info("   No outliers to remove")

        return df_clean

    def handle_outliers(self, df: pd.DataFrame,
                        columns: List[str] = None,
                        remove: bool = False) -> pd.DataFrame:
        """
        Complete outlier handling pipeline
        Note: In Telco Churn analysis, outliers are detected but NOT removed

        Args:
            df: Input DataFrame
            columns: Columns to analyze
            remove: If True, remove outliers (default: False, as per notebook)

        Returns:
            DataFrame (unchanged if remove=False)
        """
        logger.info("="*70)
        logger.info("OUTLIER DETECTION")
        logger.info("="*70)

        # Detect outliers
        outlier_report = self.detect_outliers(df, columns)

        # Remove if requested (not done in notebook analysis)
        if remove:
            logger.warning(
                "⚠️  Removing outliers (not standard for this dataset)")
            df = self.remove_outliers(df, columns)
        else:
            logger.info(
                "   ℹ️  Outliers detected but NOT removed (analysis only)")

        logger.info("="*70)
        logger.info("✅ OUTLIER DETECTION COMPLETE")
        logger.info(f"   Final shape: {df.shape}")
        logger.info("="*70)

        return df
