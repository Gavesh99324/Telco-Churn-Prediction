"""Feature binning for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBinner:
    """Create bins for continuous features"""

    def __init__(self):
        """Initialize FeatureBinner"""
        self.binning_config = {}

    def create_tenure_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure group bins as per notebook 03
        Bins: [0-12, 12-24, 24-48, 48-72] months

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with tenure_group column added
        """
        logger.info("Creating tenure groups...")

        bins = [0, 12, 24, 48, 72]
        labels = ['0-1 year', '1-2 years', '2-4 years', '4+ years']

        df['tenure_group'] = pd.cut(df['tenure'],
                                    bins=bins,
                                    labels=labels,
                                    include_lowest=True)

        # Log distribution
        distribution = df['tenure_group'].value_counts().sort_index()
        logger.info("   Tenure group distribution:")
        for label, count in distribution.items():
            logger.info(f"      {label}: {count} ({count/len(df)*100:.1f}%)")

        self.binning_config['tenure_group'] = {
            'bins': bins,
            'labels': labels
        }

        return df

    def bin_feature(self, df: pd.DataFrame, column: str,
                    bins: List[float], labels: Optional[List[str]] = None,
                    new_column: str = None) -> pd.DataFrame:
        """
        Bin a feature into categories

        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Bin edges
            labels: Bin labels (optional)
            new_column: Name for new column (default: {column}_binned)

        Returns:
            DataFrame with binned feature
        """
        if new_column is None:
            new_column = f"{column}_binned"

        logger.info(f"Binning {column} into {len(bins)-1} groups...")

        df[new_column] = pd.cut(df[column], bins=bins,
                                labels=labels, include_lowest=True)

        # Log distribution
        distribution = df[new_column].value_counts()
        logger.info(f"   Distribution:")
        for label, count in distribution.items():
            logger.info(f"      {label}: {count}")

        self.binning_config[new_column] = {
            'source_column': column,
            'bins': bins,
            'labels': labels
        }

        return df

    def create_all_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all binned features as per notebook

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all binned features
        """
        logger.info("="*70)
        logger.info("CREATING BINNED FEATURES")
        logger.info("="*70)

        # Create tenure groups
        df = self.create_tenure_groups(df)

        logger.info("="*70)
        logger.info("✅ BINNING COMPLETE")
        logger.info(f"   Features created: {list(self.binning_config.keys())}")
        logger.info("="*70)

        return df
