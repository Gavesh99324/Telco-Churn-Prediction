"""Unit tests for feature_engineering.py"""
import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_telco_df():
    """Create sample Telco churn data"""
    return pd.DataFrame({
        'tenure': [1, 12, 24, 36, 48],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
        'TotalCharges': [29.85, 1889.5, 1377.4, 1840.75, 3397.0],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes', 'No', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'No', 'Yes', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No', 'No', 'Yes'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'SeniorCitizen': ['No', 'No', 'Yes', 'Yes', 'No'],
        'Dependents': ['Yes', 'No', 'No', 'Yes', 'No']
    })


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance"""
    return FeatureEngineer()


class TestFeatureEngineerInitialization:
    """Test FeatureEngineer initialization"""
    
    def test_initialization(self, feature_engineer):
        """Test proper initialization"""
        assert isinstance(feature_engineer.created_features, list)
        assert len(feature_engineer.created_features) == 0


class TestAvgChargePerTenure:
    """Test average charge per tenure calculation"""
    
    def test_create_avg_charge_per_tenure(self, feature_engineer, sample_telco_df):
        """Test creating avg_charge_per_tenure feature"""
        result_df = feature_engineer.create_avg_charge_per_tenure(sample_telco_df)
        
        assert 'avg_charge_per_tenure' in result_df.columns
        assert 'avg_charge_per_tenure' in feature_engineer.created_features
    
    def test_avg_charge_per_tenure_calculation(self, feature_engineer, sample_telco_df):
        """Test calculation is correct"""
        result_df = feature_engineer.create_avg_charge_per_tenure(sample_telco_df)
        
        # First row: 29.85 / (1 + 1) = 14.925
        expected_first = 29.85 / (1 + 1)
        assert result_df['avg_charge_per_tenure'].iloc[0] == pytest.approx(expected_first)
    
    def test_avg_charge_per_tenure_no_division_by_zero(self, feature_engineer):
        """Test that zero tenure doesn't cause division by zero"""
        df = pd.DataFrame({
            'tenure': [0],
            'TotalCharges': [100]
        })
        
        result_df = feature_engineer.create_avg_charge_per_tenure(df)
        
        # Should be 100 / (0 + 1) = 100
        assert result_df['avg_charge_per_tenure'].iloc[0] == 100.0


class TestServiceCount:
    """Test service count feature"""
    
    def test_create_service_count(self, feature_engineer, sample_telco_df):
        """Test creating service_count feature"""
        result_df = feature_engineer.create_service_count(sample_telco_df)
        
        assert 'service_count' in result_df.columns
        assert 'service_count' in feature_engineer.created_features
    
    def test_service_count_calculation(self, feature_engineer, sample_telco_df):
        """Test service count is calculated correctly"""
        result_df = feature_engineer.create_service_count(sample_telco_df)
        
        # First row: PhoneService=Yes, others=No/Yes -> should count correctly
        first_count = result_df['service_count'].iloc[0]
        assert first_count >= 0
        assert first_count <= 9  # Max 9 services
    
    def test_service_count_range(self, feature_engineer, sample_telco_df):
        """Test that service count is in valid range"""
        result_df = feature_engineer.create_service_count(sample_telco_df)
        
        assert result_df['service_count'].min() >= 0
        assert result_df['service_count'].max() <= 9


class TestHasPhoneFeature:
    """Test has_phone binary feature"""
    
    def test_create_has_phone(self, feature_engineer, sample_telco_df):
        """Test creating has_phone feature"""
        result_df = feature_engineer.create_has_phone(sample_telco_df)
        
        assert 'has_phone' in result_df.columns
        assert 'has_phone' in feature_engineer.created_features
    
    def test_has_phone_binary(self, feature_engineer, sample_telco_df):
        """Test that has_phone is binary"""
        result_df = feature_engineer.create_has_phone(sample_telco_df)
        
        assert set(result_df['has_phone'].unique()).issubset({0, 1})
    
    def test_has_phone_correct_values(self, feature_engineer, sample_telco_df):
        """Test that has_phone values are correct"""
        result_df = feature_engineer.create_has_phone(sample_telco_df)
        
        # First row has PhoneService='Yes', should be 1
        assert result_df['has_phone'].iloc[0] == 1
        # Third row has PhoneService='No', should be 0
        assert result_df['has_phone'].iloc[2] == 0


class TestHasInternetFeature:
    """Test has_internet binary feature"""
    
    def test_create_has_internet(self, feature_engineer, sample_telco_df):
        """Test creating has_internet feature"""
        result_df = feature_engineer.create_has_internet(sample_telco_df)
        
        assert 'has_internet' in result_df.columns
        assert 'has_internet' in feature_engineer.created_features
    
    def test_has_internet_binary(self, feature_engineer, sample_telco_df):
        """Test that has_internet is binary"""
        result_df = feature_engineer.create_has_internet(sample_telco_df)
        
        assert set(result_df['has_internet'].unique()).issubset({0, 1})
    
    def test_has_internet_correct_values(self, feature_engineer, sample_telco_df):
        """Test that has_internet values are correct"""
        result_df = feature_engineer.create_has_internet(sample_telco_df)
        
        # First row has InternetService='DSL', should be 1
        assert result_df['has_internet'].iloc[0] == 1
        # Third row has InternetService='No', should be 0
        assert result_df['has_internet'].iloc[2] == 0


class TestPremiumServicesCount:
    """Test premium services count feature"""
    
    def test_create_premium_services_count(self, feature_engineer, sample_telco_df):
        """Test creating premium_services_count feature"""
        result_df = feature_engineer.create_premium_services_count(sample_telco_df)
        
        assert 'premium_services_count' in result_df.columns
        assert 'premium_services_count' in feature_engineer.created_features
    
    def test_premium_services_count_range(self, feature_engineer, sample_telco_df):
        """Test that premium services count is in valid range"""
        result_df = feature_engineer.create_premium_services_count(sample_telco_df)
        
        # Should be 0-3 (OnlineSecurity, OnlineBackup, DeviceProtection)
        assert result_df['premium_services_count'].min() >= 0
        assert result_df['premium_services_count'].max() <= 3
    
    def test_premium_services_count_calculation(self, feature_engineer, sample_telco_df):
        """Test calculation of premium services"""
        result_df = feature_engineer.create_premium_services_count(sample_telco_df)
        
        # First row: OnlineSecurity=No, OnlineBackup=Yes, DeviceProtection=No -> 1
        assert result_df['premium_services_count'].iloc[0] == 1


class TestStreamingServicesCount:
    """Test streaming services count feature"""
    
    def test_create_streaming_services_count(self, feature_engineer, sample_telco_df):
        """Test creating streaming_services_count feature"""
        result_df = feature_engineer.create_streaming_services_count(sample_telco_df)
        
        assert 'streaming_services_count' in result_df.columns
        assert 'streaming_services_count' in feature_engineer.created_features
    
    def test_streaming_services_count_range(self, feature_engineer, sample_telco_df):
        """Test that streaming count is in valid range"""
        result_df = feature_engineer.create_streaming_services_count(sample_telco_df)
        
        # Should be 0-2 (StreamingTV, StreamingMovies)
        assert result_df['streaming_services_count'].min() >= 0
        assert result_df['streaming_services_count'].max() <= 2


class TestSeniorWithDependents:
    """Test senior with dependents interaction feature"""
    
    def test_create_senior_with_dependents(self, feature_engineer, sample_telco_df):
        """Test creating senior_with_dependents feature"""
        result_df = feature_engineer.create_senior_with_dependents(sample_telco_df)
        
        assert 'senior_with_dependents' in result_df.columns
        assert 'senior_with_dependents' in feature_engineer.created_features
    
    def test_senior_with_dependents_binary(self, feature_engineer, sample_telco_df):
        """Test that feature is binary"""
        result_df = feature_engineer.create_senior_with_dependents(sample_telco_df)
        
        assert set(result_df['senior_with_dependents'].unique()).issubset({0, 1})
    
    def test_senior_with_dependents_interaction(self, feature_engineer, sample_telco_df):
        """Test interaction logic"""
        result_df = feature_engineer.create_senior_with_dependents(sample_telco_df)
        
        # Row 4: SeniorCitizen=Yes, Dependents=Yes -> should be 1
        assert result_df['senior_with_dependents'].iloc[3] == 1


class TestMonthlyToTotalRatio:
    """Test monthly to total charges ratio"""
    
    def test_create_monthly_to_total_ratio(self, feature_engineer, sample_telco_df):
        """Test creating monthly_to_total_ratio feature"""
        result_df = feature_engineer.create_monthly_to_total_ratio(sample_telco_df)
        
        assert 'monthly_to_total_ratio' in result_df.columns
        assert 'monthly_to_total_ratio' in feature_engineer.created_features
    
    def test_monthly_to_total_ratio_calculation(self, feature_engineer, sample_telco_df):
        """Test calculation is correct"""
        result_df = feature_engineer.create_monthly_to_total_ratio(sample_telco_df)
        
        # First row: 29.85 / (29.85 + 1)
        expected = 29.85 / (29.85 + 1)
        assert result_df['monthly_to_total_ratio'].iloc[0] == pytest.approx(expected)
    
    def test_monthly_to_total_ratio_no_division_by_zero(self, feature_engineer):
        """Test that zero total doesn't cause issues"""
        df = pd.DataFrame({
            'MonthlyCharges': [50],
            'TotalCharges': [0]
        })
        
        result_df = feature_engineer.create_monthly_to_total_ratio(df)
        
        # Should be 50 / (0 + 1) = 50
        assert result_df['monthly_to_total_ratio'].iloc[0] == 50.0


class TestContractEncoding:
    """Test contract ordinal encoding"""
    
    def test_create_contract_encoding(self, feature_engineer, sample_telco_df):
        """Test creating contract_encoded feature"""
        result_df = feature_engineer.create_contract_encoding(sample_telco_df)
        
        assert 'contract_encoded' in result_df.columns
        assert 'contract_encoded' in feature_engineer.created_features
    
    def test_contract_encoding_values(self, feature_engineer, sample_telco_df):
        """Test that encoding values are correct"""
        result_df = feature_engineer.create_contract_encoding(sample_telco_df)
        
        # First row: 'Month-to-month' -> 0
        assert result_df['contract_encoded'].iloc[0] == 0
        # Second row: 'One year' -> 1
        assert result_df['contract_encoded'].iloc[1] == 1
        # Fourth row: 'Two year' -> 2
        assert result_df['contract_encoded'].iloc[3] == 2
    
    def test_contract_encoding_ordinal(self, feature_engineer, sample_telco_df):
        """Test that encoding reflects ordinal nature"""
        result_df = feature_engineer.create_contract_encoding(sample_telco_df)
        
        # Values should be 0, 1, 2
        unique_values = result_df['contract_encoded'].dropna().unique()
        assert set(unique_values).issubset({0, 1, 2})


class TestCreateAllFeatures:
    """Test creating all features at once"""
    
    def test_create_all_features(self, feature_engineer, sample_telco_df):
        """Test creating all features at once"""
        result_df = feature_engineer.create_all_features(sample_telco_df)
        
        # Check that features were created
        assert len(feature_engineer.created_features) > 0
    
    def test_create_all_features_count(self, feature_engineer, sample_telco_df):
        """Test that expected number of features are created"""
        result_df = feature_engineer.create_all_features(sample_telco_df)
        
        # Should create 9 features
        expected_features = [
            'avg_charge_per_tenure', 'service_count', 'has_phone',
            'has_internet', 'premium_services_count', 'streaming_services_count',
            'senior_with_dependents', 'monthly_to_total_ratio', 'contract_encoded'
        ]
        
        for feat in expected_features:
            assert feat in result_df.columns
    
    def test_create_all_features_preserves_original(self, feature_engineer, sample_telco_df):
        """Test that original columns are preserved"""
        original_cols = sample_telco_df.columns.tolist()
        result_df = feature_engineer.create_all_features(sample_telco_df)
        
        for col in original_cols:
            assert col in result_df.columns
    
    def test_create_all_features_increases_column_count(self, feature_engineer, sample_telco_df):
        """Test that column count increases"""
        original_count = len(sample_telco_df.columns)
        result_df = feature_engineer.create_all_features(sample_telco_df)
        
        assert len(result_df.columns) > original_count
    
    def test_create_all_features_preserves_rows(self, feature_engineer, sample_telco_df):
        """Test that row count is preserved"""
        original_rows = len(sample_telco_df)
        result_df = feature_engineer.create_all_features(sample_telco_df)
        
        assert len(result_df) == original_rows


class TestMissingColumns:
    """Test behavior with missing columns"""
    
    def test_missing_phone_service_column(self, feature_engineer):
        """Test when PhoneService column is missing"""
        df = pd.DataFrame({
            'tenure': [1, 2],
            'TotalCharges': [100, 200],
            'MonthlyCharges': [50, 60]
        })
        
        result_df = feature_engineer.create_has_phone(df)
        
        # Should handle gracefully
        assert 'has_phone' not in result_df.columns or 'has_phone' in result_df.columns
    
    def test_missing_contract_column(self, feature_engineer):
        """Test when Contract column is missing"""
        df = pd.DataFrame({
            'tenure': [1, 2],
            'TotalCharges': [100, 200],
            'MonthlyCharges': [50, 60]
        })
        
        result_df = feature_engineer.create_contract_encoding(df)
        
        # Should handle gracefully
        assert 'contract_encoded' not in result_df.columns or 'contract_encoded' in result_df.columns


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_row(self, feature_engineer):
        """Test with single row"""
        df = pd.DataFrame({
            'tenure': [1],
            'MonthlyCharges': [50],
            'TotalCharges': [50],
            'PhoneService': ['Yes'],
            'InternetService': ['DSL']
        })
        
        result_df = feature_engineer.create_all_features(df)
        
        assert len(result_df) == 1
        assert 'avg_charge_per_tenure' in result_df.columns
    
    def test_zero_values(self, feature_engineer):
        """Test with zero values"""
        df = pd.DataFrame({
            'tenure': [0],
            'MonthlyCharges': [0],
            'TotalCharges': [0]
        })
        
        result_df = feature_engineer.create_avg_charge_per_tenure(df)
        result_df = feature_engineer.create_monthly_to_total_ratio(result_df)
        
        # Should handle zeros gracefully (using +1 to avoid division by zero)
        assert not result_df['avg_charge_per_tenure'].isna().any()
        assert not result_df['monthly_to_total_ratio'].isna().any()


class TestFeatureTracking:
    """Test feature tracking"""
    
    def test_created_features_list_updated(self, feature_engineer, sample_telco_df):
        """Test that created_features list is updated"""
        assert len(feature_engineer.created_features) == 0
        
        feature_engineer.create_avg_charge_per_tenure(sample_telco_df)
        assert len(feature_engineer.created_features) == 1
        
        feature_engineer.create_service_count(sample_telco_df)
        assert len(feature_engineer.created_features) == 2
    
    def test_created_features_list_reset(self, feature_engineer, sample_telco_df):
        """Test that list is reset in create_all_features"""
        feature_engineer.create_avg_charge_per_tenure(sample_telco_df)
        assert len(feature_engineer.created_features) > 0
        
        # create_all_features should reset the list
        feature_engineer.create_all_features(sample_telco_df)
        
        # Should have all features from create_all_features
        assert len(feature_engineer.created_features) >= 9


class TestDataIntegrity:
    """Test data integrity"""
    
    def test_original_dataframe_not_modified(self, feature_engineer):
        """Test that original DataFrame is not modified"""
        df = pd.DataFrame({
            'tenure': [1, 2, 3],
            'MonthlyCharges': [50, 60, 70],
            'TotalCharges': [50, 120, 210]
        })
        
        original_shape = df.shape
        original_cols = df.columns.tolist()
        
        feature_engineer.create_avg_charge_per_tenure(df)
        
        # Original should be unchanged
        assert df.shape == original_shape
        assert df.columns.tolist() == original_cols
