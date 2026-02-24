"""Missing value imputation for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Handle missing values in Telco Churn dataset"""

    def __init__(self):
        """Initialize MissingValueHandler"""
        self.imputation_strategy = {}

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze missing values in the dataset

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with missing value counts per column
        """
        logger.info("Analyzing missing values...")
        missing_counts = df.isnull().sum()
        missing_dict = missing_counts[missing_counts > 0].to_dict()

        if missing_dict:
            logger.warning(f"⚠️  Missing values found:")
            for col, count in missing_dict.items():
                percentage = (count / len(df)) * 100
                logger.warning(f"   {col}: {count} ({percentage:.2f}%)")
        else:
            logger.info("   ✅ No missing values found")

        return missing_dict

    def impute_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing TotalCharges using business logic
        Strategy: TotalCharges = MonthlyCharges * tenure

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with TotalCharges imputed
        """
        missing_count = df['TotalCharges'].isna().sum()

        if missing_count == 0:
            logger.info("   No missing TotalCharges values")
            return df

        logger.info(f"Imputing {missing_count} missing TotalCharges values...")

        # Create a mask for missing values
        mask = df['TotalCharges'].isna()

        # Impute: TotalCharges = MonthlyCharges * tenure
        # This handles tenure=0 cases automatically (result is 0)
        df.loc[mask, 'TotalCharges'] = df.loc[mask,
                                              'MonthlyCharges'] * df.loc[mask, 'tenure']

        # Log imputation details
        imputed_values = df.loc[mask, 'TotalCharges']
        logger.info(f"   ✅ Imputed {missing_count} values")
        logger.info(
            f"   Imputed range: [{imputed_values.min():.2f}, {imputed_values.max():.2f}]")

        # Record strategy
        self.imputation_strategy['TotalCharges'] = 'MonthlyCharges * tenure'

        return df

    def handle_remaining_missing(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """
        Handle any remaining missing values after imputation

        Args:
            df: Input DataFrame
            drop: If True, drop rows with remaining missing values

        Returns:
            DataFrame with remaining missing values handled
        """
        remaining_missing = df.isnull().sum().sum()

        if remaining_missing == 0:
            logger.info("   ✅ No remaining missing values")
            return df

        logger.warning(
            f"⚠️  {remaining_missing} missing values remain after imputation")

        if drop:
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            logger.info(f"   Dropped {dropped_rows} rows with missing values")
        else:
            logger.warning(
                "   Missing values not dropped - may cause issues downstream")

        return df

    def validate_no_missing(self, df: pd.DataFrame) -> bool:
        """
        Validate that no missing values remain

        Args:
            df: DataFrame to validate

        Returns:
            True if no missing values

        Raises:
            ValueError: If missing values are found
        """
        missing_count = df.isnull().sum().sum()

        if missing_count > 0:
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            logger.error(
                f"❌ Validation failed: {missing_count} missing values found")
            logger.error(f"   Columns: {missing_cols.to_dict()}")
            raise ValueError(f"Missing values found: {missing_cols.to_dict()}")

        logger.info("✅ Validation passed: No missing values")
        return True

    def handle_missing(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Complete missing value handling pipeline

        Args:
            df: Input DataFrame
            validate: If True, validate no missing values remain

        Returns:
            DataFrame with all missing values handled
        """
        logger.info("="*70)
        logger.info("STARTING MISSING VALUE HANDLING")
        logger.info("="*70)

        # Analyze missing values
        missing_dict = self.analyze_missing_values(df)

        if not missing_dict:
            logger.info("="*70)
            logger.info("✅ NO MISSING VALUES TO HANDLE")
            logger.info("="*70)
            return df

        # Impute TotalCharges
        if 'TotalCharges' in missing_dict:
            df = self.impute_total_charges(df)

        # Handle any remaining missing values
        df = self.handle_remaining_missing(df, drop=True)

        # Validate
        if validate:
            self.validate_no_missing(df)

        logger.info("="*70)
        logger.info("✅ MISSING VALUE HANDLING COMPLETE")
        logger.info(f"   Final shape: {df.shape}")
        if self.imputation_strategy:
            logger.info(f"   Strategies used: {self.imputation_strategy}")
        logger.info("="*70)

        return df
