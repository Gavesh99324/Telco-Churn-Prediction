"""Train/test split for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train and test sets with stratification"""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataSplitter

        Args:
            test_size: Proportion of dataset for test set (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.Series, np.ndarray],
              stratify: bool = True) -> Tuple:
        """
        Split data into train and test sets

        Args:
            X: Features
            y: Target variable
            stratify: If True, stratify split based on y (maintain class distribution)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("="*70)
        logger.info("SPLITTING DATA INTO TRAIN/TEST SETS")
        logger.info("="*70)
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Test size: {self.test_size} ({self.test_size*100:.0f}%)")
        logger.info(f"Random state: {self.random_state}")

        # Determine stratification
        stratify_param = y if stratify else None
        if stratify:
            logger.info(
                "Stratification: ENABLED (maintains class distribution)")
            # Log class distribution
            if isinstance(y, pd.Series):
                class_counts = y.value_counts().to_dict()
            else:
                unique, counts = np.unique(y, return_counts=True)
                class_counts = dict(zip(unique, counts))
            logger.info(f"   Class distribution: {class_counts}")
        else:
            logger.info("Stratification: DISABLED")

        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        # Log results
        logger.info("="*70)
        logger.info("✅ DATA SPLIT COMPLETE")
        logger.info(
            f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(
            f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Log class distribution in splits
        if isinstance(y_train, pd.Series):
            train_dist = y_train.value_counts().to_dict()
            test_dist = y_test.value_counts().to_dict()
        else:
            train_unique, train_counts = np.unique(y_train, return_counts=True)
            train_dist = dict(zip(train_unique, train_counts))
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            test_dist = dict(zip(test_unique, test_counts))

        logger.info(f"   Train class distribution: {train_dist}")
        logger.info(f"   Test class distribution: {test_dist}")
        logger.info("="*70)

        return X_train, X_test, y_train, y_test

    def get_split_info(self) -> dict:
        """
        Get information about split configuration

        Returns:
            Dictionary with split parameters
        """
        return {
            'test_size': self.test_size,
            'train_size': 1 - self.test_size,
            'random_state': self.random_state
        }
