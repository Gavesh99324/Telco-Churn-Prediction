"""Test data ingestion module"""
import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.data_ingestion import DataIngestion


@pytest.fixture
def sample_data_path(tmp_path):
    """Create a sample CSV file for testing"""
    data = {
        'customerID': ['001', '002', '003', '004', '005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No', 'No', 'Yes'],
        'tenure': [12, 24, 6, 48, 3],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'No', 'Yes', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'No', 'No', 'Yes', 'No'],
        'TechSupport': ['Yes', 'No', 'No', 'Yes', 'No'],
        'StreamingTV': ['No', 'Yes', 'No', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Bank transfer', 'Credit card', 'Electronic check', 'Bank transfer', 'Electronic check'],
        'MonthlyCharges': [50.0, 75.0, 30.0, 100.0, 80.0],
        # Note: string with whitespace
        'TotalCharges': ['600', ' ', '180', '4800', '240'],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_data_ingestion_initialization():
    """Test DataIngestion initialization"""
    ingestion = DataIngestion()
    assert ingestion is not None
    assert ingestion.data_path == 'data/raw/telco_churn.csv'
    assert isinstance(ingestion.expected_schema, dict)
    assert len(ingestion.expected_schema) > 0


def test_load_data(sample_data_path):
    """Test loading data from CSV"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert 'customerID' in df.columns
    assert 'Churn' in df.columns


def test_validate_schema(sample_data_path):
    """Test schema validation"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    # Should pass validation
    result = ingestion.validate_schema(df)
    assert result == True


def test_validate_schema_missing_columns(sample_data_path):
    """Test schema validation with missing columns"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    # Drop a required column
    df = df.drop('Churn', axis=1)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Missing required columns"):
        ingestion.validate_schema(df)


def test_validate_data_quality(sample_data_path):
    """Test data quality validation"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    quality_report = ingestion.validate_data_quality(df)

    assert isinstance(quality_report, dict)
    assert 'total_rows' in quality_report
    assert 'total_columns' in quality_report
    assert 'duplicate_rows' in quality_report
    assert 'missing_values' in quality_report
    assert quality_report['total_rows'] == 5


def test_convert_total_charges(sample_data_path):
    """Test TotalCharges conversion to numeric"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    # Before conversion, TotalCharges is object type
    assert df['TotalCharges'].dtype == 'object'

    df = ingestion.convert_total_charges(df)

    # After conversion, should be numeric
    assert df['TotalCharges'].dtype in ['float64', 'int64']
    # Should have NaN for whitespace values
    assert df['TotalCharges'].isna().sum() == 1


def test_convert_senior_citizen(sample_data_path):
    """Test SeniorCitizen conversion to categorical"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    # Before conversion
    assert df['SeniorCitizen'].dtype in ['int64', 'object']

    df = ingestion.convert_senior_citizen(df)

    # After conversion
    assert df['SeniorCitizen'].dtype == 'object'
    assert set(df['SeniorCitizen'].unique()) <= {'Yes', 'No'}


def test_remove_duplicates(sample_data_path):
    """Test duplicate removal"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.load_data()

    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(df) == 6

    df = ingestion.remove_duplicates(df)

    # Should have removed 1 duplicate
    assert len(df) == 5


def test_complete_ingestion_pipeline(sample_data_path):
    """Test complete ingestion pipeline"""
    ingestion = DataIngestion(data_path=sample_data_path)
    df = ingestion.ingest()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # TotalCharges should be converted to numeric
    assert df['TotalCharges'].dtype in ['float64', 'int64']
    # SeniorCitizen should be categorical
    assert df['SeniorCitizen'].dtype == 'object'
    # No duplicates
    assert df.duplicated().sum() == 0
