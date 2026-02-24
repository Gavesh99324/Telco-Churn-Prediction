"""Feature normalization for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureScaler:
    """Scale features using StandardScaler (z-score normalization)"""

    def __init__(self):
        """Initialize FeatureScaler"""
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Fit scaler and transform features

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Scaled features as DataFrame
        """
        logger.info("="*70)
        logger.info("STARTING FEATURE SCALING")
        logger.info("="*70)
        logger.info(f"Input shape: {X.shape}")

        # Preserve column names and index if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            index = X.index
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            index = None

        # Fit and transform
        logger.info("Applying StandardScaler (z-score normalization)...")
        X_scaled_array = self.scaler.fit_transform(X)
        self.is_fitted = True

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(
            X_scaled_array,
            columns=self.feature_names,
            index=index
        )

        # Log scaling statistics
        logger.info(f"   Scaled {len(self.feature_names)} features")
        logger.info(f"   Mean (per feature): ~0.0")
        logger.info(f"   Std (per feature): ~1.0")
        logger.info(f"   Formula: (X - mean) / std_dev")

        logger.info("="*70)
        logger.info("✅ FEATURE SCALING COMPLETE")
        logger.info(f"   Output shape: {X_scaled.shape}")
        logger.info("="*70)

        return X_scaled

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform features using fitted scaler

        Args:
            X: Input features

        Returns:
            Scaled features as DataFrame

        Raises:
            RuntimeError: If scaler not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")

        logger.info("Transforming features with fitted scaler...")

        # Preserve column names and index if DataFrame
        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = None

        # Transform
        X_scaled_array = self.scaler.transform(X)

        # Convert back to DataFrame
        X_scaled = pd.DataFrame(
            X_scaled_array,
            columns=self.feature_names,
            index=index
        )

        logger.info(f"   ✅ Transformed {X_scaled.shape[0]} samples")

        return X_scaled

    def inverse_transform(self, X_scaled: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale

        Args:
            X_scaled: Scaled features

        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")

        X_original = self.scaler.inverse_transform(X_scaled)

        if isinstance(X_scaled, pd.DataFrame):
            return pd.DataFrame(
                X_original,
                columns=self.feature_names,
                index=X_scaled.index
            )

        return X_original
