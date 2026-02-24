"""Data ingestion module for Telco Customer Churn"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handle data loading and schema validation for Telco Churn dataset"""

    def __init__(self, data_path: str = 'data/raw/telco_churn.csv'):
        """
        Initialize DataIngestion

        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        self.expected_schema = self._define_schema()

    def _define_schema(self) -> Dict[str, str]:
        """
        Define expected schema for Telco Churn dataset

        Returns:
            Dictionary mapping column names to expected data types
        """
        return {
            'customerID': 'object',
            'gender': 'object',
            'SeniorCitizen': 'int64',
            'Partner': 'object',
            'Dependents': 'object',
            'tenure': 'int64',
            'PhoneService': 'object',
            'MultipleLines': 'object',
            'InternetService': 'object',
            'OnlineSecurity': 'object',
            'OnlineBackup': 'object',
            'DeviceProtection': 'object',
            'TechSupport': 'object',
            'StreamingTV': 'object',
            'StreamingMovies': 'object',
            'Contract': 'object',
            'PaperlessBilling': 'object',
            'PaymentMethod': 'object',
            'MonthlyCharges': 'float64',
            'TotalCharges': 'object',  # Will be converted later
            'Churn': 'object'
        }

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If CSV file is empty
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(
                f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"❌ Empty CSV file: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has expected columns

        Args:
            df: DataFrame to validate

        Returns:
            True if schema is valid

        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Validating schema...")

        # Check for missing columns
        expected_cols = set(self.expected_schema.keys())
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            logger.error(f"❌ Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        if extra_cols:
            logger.warning(
                f"⚠️  Extra columns found (will be ignored): {extra_cols}")

        logger.info("✅ Schema validation passed")
        return True

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform data quality checks

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary containing quality metrics
        """
        logger.info("Performing data quality checks...")

        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_customer_ids': df['customerID'].duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': df.isnull().sum().sum()
        }

        # Check for negative values in numerical columns
        numerical_cols = ['tenure', 'MonthlyCharges']
        quality_report['negative_values'] = {}
        for col in numerical_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                quality_report['negative_values'][col] = negative_count

        # Log summary
        logger.info(f"   Total rows: {quality_report['total_rows']}")
        logger.info(f"   Duplicate rows: {quality_report['duplicate_rows']}")
        logger.info(
            f"   Duplicate customer IDs: {quality_report['duplicate_customer_ids']}")
        logger.info(
            f"   Total missing values: {quality_report['total_missing']}")

        if quality_report['total_missing'] > 0:
            logger.warning(
                f"⚠️  Found {quality_report['total_missing']} missing values")

        return quality_report

    def convert_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert TotalCharges from object to numeric, handling whitespace

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with TotalCharges converted to numeric
        """
        logger.info("Converting TotalCharges to numeric...")

        if df['TotalCharges'].dtype == 'object':
            # Check for problematic values
            empty_count = ((df['TotalCharges'] == ' ') |
                           (df['TotalCharges'] == '')).sum()
            if empty_count > 0:
                logger.warning(
                    f"⚠️  Found {empty_count} empty/whitespace TotalCharges values")

            # Convert to numeric, coercing errors to NaN
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'], errors='coerce')

            new_missing = df['TotalCharges'].isna().sum()
            logger.info(
                f"   Conversion complete. New missing values: {new_missing}")
        else:
            logger.info("   TotalCharges already numeric")

        return df

    def convert_senior_citizen(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert SeniorCitizen from binary (0/1) to categorical ('No'/'Yes')

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with SeniorCitizen converted
        """
        logger.info("Converting SeniorCitizen to categorical...")
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        logger.info("   ✅ SeniorCitizen conversion complete")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            logger.warning(
                f"⚠️  Found {duplicates} duplicate rows - removing...")
            df = df.drop_duplicates()
            logger.info(f"   Removed {initial_count - len(df)} duplicates")
        else:
            logger.info("   No duplicates found")

        return df

    def ingest(self) -> pd.DataFrame:
        """
        Complete ingestion pipeline: load, validate, and preprocess

        Returns:
            Processed DataFrame ready for missing value handling
        """
        logger.info("="*70)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("="*70)

        # Load data
        df = self.load_data()

        # Validate schema
        self.validate_schema(df)

        # Data quality checks
        quality_report = self.validate_data_quality(df)

        # Convert TotalCharges to numeric
        df = self.convert_total_charges(df)

        # Convert SeniorCitizen to categorical
        df = self.convert_senior_citizen(df)

        # Remove duplicates
        df = self.remove_duplicates(df)

        logger.info("="*70)
        logger.info("✅ DATA INGESTION COMPLETE")
        logger.info(f"   Final shape: {df.shape}")
        logger.info("="*70)

        return df
