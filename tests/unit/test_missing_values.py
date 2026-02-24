"""Test missing value handling module"""
import pytest
import pandas as pd
import numpy as np
from src.handle_missing_values import MissingValueHandler


@pytest.fixture
def sample_df_with_missing():
    """Create sample DataFrame with missing values"""
    return pd.DataFrame({
        'customerID': ['001', '002', '003', '004', '005'],
        'tenure': [12, 0, 6, 48, 3],
        'MonthlyCharges': [50.0, 75.0, 30.0, 100.0, 80.0],
        'TotalCharges': [600.0, np.nan, 180.0, 4800.0, np.nan],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    })


def test_missing_value_handler_initialization():
    """Test MissingValueHandler initialization"""
    handler = MissingValueHandler()
    assert handler is not None
    assert isinstance(handler.imputation_strategy, dict)


def test_analyze_missing_values(sample_df_with_missing):
    """Test missing value analysis"""
    handler = MissingValueHandler()
    missing_dict = handler.analyze_missing_values(sample_df_with_missing)

    assert isinstance(missing_dict, dict)
    assert 'TotalCharges' in missing_dict
    assert missing_dict['TotalCharges'] == 2


def test_impute_total_charges(sample_df_with_missing):
    """Test TotalCharges imputation"""
    handler = MissingValueHandler()
    df = handler.impute_total_charges(sample_df_with_missing)

    # Should have no missing TotalCharges
    assert df['TotalCharges'].isna().sum() == 0

    # Check imputation logic: For row 1 (tenure=0), TotalCharges should be 0
    assert df.loc[1, 'TotalCharges'] == 0.0

    # For row 4 (tenure=3, MonthlyCharges=80), TotalCharges should be 240
    assert df.loc[4, 'TotalCharges'] == 240.0

    # Check strategy was recorded
    assert 'TotalCharges' in handler.imputation_strategy


def test_handle_remaining_missing():
    """Test handling remaining missing values"""
    handler = MissingValueHandler()
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': [5, np.nan, 7, 8]
    })

    df_clean = handler.handle_remaining_missing(df, drop=True)

    # Should have dropped rows with missing values
    assert len(df_clean) == 2
    assert df_clean.isnull().sum().sum() == 0


def test_validate_no_missing():
    """Test validation of no missing values"""
    handler = MissingValueHandler()

    # DataFrame with no missing values
    df_clean = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    result = handler.validate_no_missing(df_clean)
    assert result == True

    # DataFrame with missing values
    df_missing = pd.DataFrame({'col1': [1, np.nan, 3]})
    with pytest.raises(ValueError, match="Missing values found"):
        handler.validate_no_missing(df_missing)


def test_complete_missing_handling_pipeline(sample_df_with_missing):
    """Test complete missing value handling pipeline"""
    handler = MissingValueHandler()
    df = handler.handle_missing(sample_df_with_missing)

    # Should have no missing values
    assert df.isnull().sum().sum() == 0

    # Should have imputed TotalCharges correctly
    assert df.loc[1, 'TotalCharges'] == 0.0  # tenure=0
    assert df.loc[4, 'TotalCharges'] == 240.0  # MonthlyCharges * tenure

    # Shape should be preserved (no rows dropped if only TotalCharges missing)
    assert len(df) == 5
