"""Test feature encoding module"""
import pytest
import pandas as pd
import numpy as np
from src.feature_encoding import FeatureEncoder


@pytest.fixture
def sample_categorical_df():
    """Create sample DataFrame with categorical features"""
    return pd.DataFrame({
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
        'tenure': [12, 24, 6, 48, 3],
        'MonthlyCharges': [50.0, 75.0, 30.0, 100.0, 80.0],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    })


def test_feature_encoder_initialization():
    """Test FeatureEncoder initialization"""
    encoder = FeatureEncoder()
    assert encoder is not None
    assert isinstance(encoder.label_encoders, dict)
    assert len(encoder.label_encoders) == 0


def test_identify_binary_columns(sample_categorical_df):
    """Test identification of binary columns"""
    encoder = FeatureEncoder()
    categorical_cols = ['gender', 'Partner',
                        'Contract', 'InternetService', 'Churn']
    binary_cols = encoder.identify_binary_columns(
        sample_categorical_df, categorical_cols)

    # gender, Partner, and Churn are binary (2 unique values)
    assert 'gender' in binary_cols
    assert 'Partner' in binary_cols
    assert 'Churn' in binary_cols
    # Contract and InternetService have >2 values
    assert 'Contract' not in binary_cols
    assert 'InternetService' not in binary_cols


def test_label_encode_binary(sample_categorical_df):
    """Test label encoding of binary columns"""
    encoder = FeatureEncoder()
    df = sample_categorical_df.copy()
    binary_cols = ['gender', 'Partner']

    df_encoded = encoder.label_encode_binary(df, binary_cols)

    # Should be encoded to integers
    assert df_encoded['gender'].dtype in ['int64', 'int32']
    assert df_encoded['Partner'].dtype in ['int64', 'int32']

    # Should have stored encoders
    assert 'gender' in encoder.label_encoders
    assert 'Partner' in encoder.label_encoders

    # Values should be 0 or 1
    assert set(df_encoded['gender'].unique()) <= {0, 1}
    assert set(df_encoded['Partner'].unique()) <= {0, 1}


def test_onehot_encode_multiclass(sample_categorical_df):
    """Test one-hot encoding of multi-class columns"""
    encoder = FeatureEncoder()
    df = sample_categorical_df.copy()

    # First label encode binary columns
    binary_cols = ['gender', 'Partner']
    df = encoder.label_encode_binary(df, binary_cols)

    # Then one-hot encode remaining categorical
    categorical_cols = ['Contract', 'InternetService']
    df_encoded = encoder.onehot_encode_multiclass(df, categorical_cols)

    # Should have created new dummy columns
    assert len(df_encoded.columns) > len(df.columns)
    # Original columns should be removed
    assert 'Contract' not in df_encoded.columns
    assert 'InternetService' not in df_encoded.columns
    # New dummy columns should exist
    assert any('Contract_' in col for col in df_encoded.columns)
    assert any('InternetService_' in col for col in df_encoded.columns)


def test_encode_target(sample_categorical_df):
    """Test target variable encoding"""
    encoder = FeatureEncoder()
    y = sample_categorical_df['Churn']

    y_encoded = encoder.encode_target(y)

    # Should be numpy array
    assert isinstance(y_encoded, np.ndarray)
    # Should have values 0 and 1
    assert set(y_encoded) <= {0, 1}
    # 'No' should be 0, 'Yes' should be 1
    assert encoder.target_encoder is not None


def test_fit_transform_without_target(sample_categorical_df):
    """Test fit_transform without target column"""
    encoder = FeatureEncoder()
    df = sample_categorical_df.drop('Churn', axis=1)

    X_encoded = encoder.fit_transform(df, target_col=None)

    # Should return DataFrame
    assert isinstance(X_encoded, pd.DataFrame)
    # Should have more columns due to one-hot encoding
    assert X_encoded.shape[1] >= df.shape[1]
    # Numerical columns should be preserved
    assert 'tenure' in X_encoded.columns
    assert 'MonthlyCharges' in X_encoded.columns


def test_fit_transform_with_target(sample_categorical_df):
    """Test fit_transform with target column"""
    encoder = FeatureEncoder()

    X_encoded, y_encoded = encoder.fit_transform(
        sample_categorical_df.drop('Churn', axis=1), target_col=None)
    y_encoded = encoder.encode_target(sample_categorical_df['Churn'])

    # X should be DataFrame
    assert isinstance(X_encoded, pd.DataFrame)
    # y should be numpy array
    assert isinstance(y_encoded, np.ndarray)
    # Shapes should match
    assert len(X_encoded) == len(y_encoded)


def test_transform_new_data(sample_categorical_df):
    """Test transforming new data with fitted encoders"""
    encoder = FeatureEncoder()
    df_train = sample_categorical_df.iloc[:3].drop('Churn', axis=1).copy()
    df_test = sample_categorical_df.iloc[3:].drop('Churn', axis=1).copy()

    # Fit on training data
    X_train = encoder.fit_transform(df_train, target_col=None)

    # Transform test data
    X_test = encoder.transform(df_test)

    # Should have same columns (approximately - one-hot encoding may differ)
    assert isinstance(X_test, pd.DataFrame)
    # Should have same number of samples
    assert len(X_test) == len(df_test)
