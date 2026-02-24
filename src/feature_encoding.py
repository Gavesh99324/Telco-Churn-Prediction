"""Categorical encoding for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEncoder:
    """Encode categorical features using Label Encoding and One-Hot Encoding"""

    def __init__(self):
        """Initialize FeatureEncoder"""
        self.label_encoders = {}
        self.target_encoder = None
        self.encoded_columns = []
        self.onehot_columns = []

    def identify_binary_columns(self, df: pd.DataFrame,
                                categorical_cols: List[str]) -> List[str]:
        """
        Identify binary categorical columns (exactly 2 unique values)

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names

        Returns:
            List of binary column names
        """
        binary_cols = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count == 2:
                binary_cols.append(col)

        logger.info(
            f"Identified {len(binary_cols)} binary columns for Label Encoding")
        return binary_cols

    def label_encode_binary(self, df: pd.DataFrame,
                            binary_cols: List[str]) -> pd.DataFrame:
        """
        Apply Label Encoding to binary categorical features

        Args:
            df: Input DataFrame
            binary_cols: List of binary column names

        Returns:
            DataFrame with binary columns label encoded
        """
        if not binary_cols:
            logger.info("   No binary columns to encode")
            return df

        logger.info(
            f"Applying Label Encoding to {len(binary_cols)} binary columns...")

        for col in binary_cols:
            le = LabelEncoder()
            original_values = df[col].unique()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

            # Log mapping
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"   {col}: {mapping}")

        self.encoded_columns = binary_cols
        return df

    def onehot_encode_multiclass(self, df: pd.DataFrame,
                                 categorical_cols: List[str]) -> pd.DataFrame:
        """
        Apply One-Hot Encoding to multi-class categorical features

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names

        Returns:
            DataFrame with one-hot encoded features
        """
        if not categorical_cols:
            logger.info("   No multi-class columns to encode")
            return df

        logger.info(
            f"Applying One-Hot Encoding to {len(categorical_cols)} multi-class columns...")

        # Get original column count
        original_cols = df.shape[1]

        # Apply one-hot encoding with drop_first=True to avoid multicollinearity
        df_encoded = pd.get_dummies(
            df, columns=categorical_cols, drop_first=True)

        # Track which columns were created
        new_cols = [col for col in df_encoded.columns if col not in df.columns]
        self.onehot_columns = new_cols

        logger.info(f"   Created {len(new_cols)} new dummy variables")
        logger.info(
            f"   Total features: {original_cols} → {df_encoded.shape[1]}")

        return df_encoded

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Encode target variable (Churn: 'No' → 0, 'Yes' → 1)

        Args:
            y: Target variable Series

        Returns:
            Encoded target as numpy array
        """
        logger.info("Encoding target variable (Churn)...")

        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)

        # Log mapping
        mapping = dict(
            zip(le_target.classes_, le_target.transform(le_target.classes_)))
        logger.info(f"   Mapping: {mapping}")
        logger.info(f"   Class distribution: {np.bincount(y_encoded)}")

        self.target_encoder = le_target
        return y_encoded

    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> tuple:
        """
        Complete encoding pipeline: label encode binary, one-hot encode multi-class

        Args:
            df: Input DataFrame
            target_col: Name of target column (if None, returns only X)

        Returns:
            Tuple of (X_encoded, y_encoded) if target_col provided, else X_encoded
        """
        logger.info("="*70)
        logger.info("STARTING FEATURE ENCODING")
        logger.info("="*70)

        df = df.copy()

        # Separate target if provided
        if target_col:
            y = df[target_col].copy()
            X = df.drop(target_col, axis=1)
        else:
            X = df.copy()
            y = None

        # Identify categorical columns
        categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns.tolist()
        logger.info(f"Found {len(categorical_cols)} categorical columns")

        if not categorical_cols:
            logger.info("   No categorical columns to encode")
            if target_col:
                y_encoded = self.encode_target(y)
                return X, y_encoded
            return X

        # Step 1: Label encode binary columns
        binary_cols = self.identify_binary_columns(X, categorical_cols)
        X = self.label_encode_binary(X, binary_cols)

        # Step 2: One-hot encode remaining categorical columns
        remaining_categorical = X.select_dtypes(
            include=['object', 'category']).columns.tolist()
        X_encoded = self.onehot_encode_multiclass(X, remaining_categorical)

        logger.info("="*70)
        logger.info("✅ FEATURE ENCODING COMPLETE")
        logger.info(f"   Input shape: {df.shape}")
        logger.info(f"   Output shape: {X_encoded.shape}")
        logger.info(f"   Label encoded: {len(self.encoded_columns)} columns")
        logger.info(
            f"   One-hot encoded: {len(remaining_categorical)} columns → {len(self.onehot_columns)} features")
        logger.info("="*70)

        # Encode target if provided
        if target_col:
            y_encoded = self.encode_target(y)
            return X_encoded, y_encoded

        return X_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders

        Args:
            df: Input DataFrame

        Returns:
            Encoded DataFrame
        """
        df = df.copy()

        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Apply one-hot encoding (get_dummies will create same columns if data is consistent)
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df
