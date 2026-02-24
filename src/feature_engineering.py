"""Feature engineering - create new features from existing ones"""
import pandas as pd
import numpy as np
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create new features for Telco Churn dataset as per notebook 03"""

    def __init__(self):
        """Initialize FeatureEngineer"""
        self.created_features = []

    def create_avg_charge_per_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average charge per tenure month
        Formula: TotalCharges / (tenure + 1) to avoid division by zero
        """
        df['avg_charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        self.created_features.append('avg_charge_per_tenure')
        return df

    def create_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count total number of services subscribed
        """
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']

        df['service_count'] = 0
        for col in service_cols:
            if col in df.columns:
                df['service_count'] += (df[col] == 'Yes').astype(int)

        self.created_features.append('service_count')
        return df

    def create_has_phone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary flag for phone service"""
        if 'PhoneService' in df.columns:
            df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
            self.created_features.append('has_phone')
        return df

    def create_has_internet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary flag for internet service"""
        if 'InternetService' in df.columns:
            df['has_internet'] = (df['InternetService'] != 'No').astype(int)
            self.created_features.append('has_internet')
        return df

    def create_premium_services_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count of premium services (security, backup, protection)"""
        premium_services = ['OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection']
        df['premium_services_count'] = 0
        for col in premium_services:
            if col in df.columns:
                df['premium_services_count'] += (df[col] == 'Yes').astype(int)

        self.created_features.append('premium_services_count')
        return df

    def create_streaming_services_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count of streaming services"""
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        df['streaming_services_count'] = 0
        for col in streaming_cols:
            if col in df.columns:
                df['streaming_services_count'] += (df[col]
                                                   == 'Yes').astype(int)

        self.created_features.append('streaming_services_count')
        return df

    def create_senior_with_dependents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction: Senior citizen with dependents"""
        if 'SeniorCitizen' in df.columns and 'Dependents' in df.columns:
            df['senior_with_dependents'] = (
                (df['SeniorCitizen'] == 'Yes') & (df['Dependents'] == 'Yes')
            ).astype(int)
            self.created_features.append('senior_with_dependents')
        return df

    def create_monthly_to_total_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratio of monthly charges to total charges"""
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / \
            (df['TotalCharges'] + 1)
        self.created_features.append('monthly_to_total_ratio')
        return df

    def create_contract_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal encoding for contract type"""
        if 'Contract' in df.columns:
            contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
            df['contract_encoded'] = df['Contract'].map(contract_map)
            self.created_features.append('contract_encoded')
        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all new features added
        """
        logger.info("="*70)
        logger.info("CREATING ENGINEERED FEATURES")
        logger.info("="*70)
        logger.info(f"Input shape: {df.shape}")

        df = df.copy()
        self.created_features = []

        # Create all features
        df = self.create_avg_charge_per_tenure(df)
        df = self.create_service_count(df)
        df = self.create_has_phone(df)
        df = self.create_has_internet(df)
        df = self.create_premium_services_count(df)
        df = self.create_streaming_services_count(df)
        df = self.create_senior_with_dependents(df)
        df = self.create_monthly_to_total_ratio(df)
        df = self.create_contract_encoding(df)

        logger.info(f"✅ Created {len(self.created_features)} new features:")
        for i, feat in enumerate(self.created_features, 1):
            logger.info(f"   {i}. {feat}")

        logger.info("="*70)
        logger.info("✅ FEATURE ENGINEERING COMPLETE")
        logger.info(f"   Output shape: {df.shape}")
        logger.info("="*70)

        return df
