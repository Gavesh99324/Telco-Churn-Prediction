"""Test complete data pipeline"""
import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from pipelines.data_pipeline import DataPipeline


@pytest.fixture
def sample_telco_data(tmp_path):
    """Create a complete sample Telco dataset"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customerID': [f'C{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'], n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
        'TotalCharges': '',  # Will be filled below
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Create TotalCharges (some with whitespace to test conversion)
    total_charges = []
    for i in range(n_samples):
        if df.loc[i, 'tenure'] == 0:
            total_charges.append(' ')  # Whitespace for tenure=0
        else:
            total_charges.append(str(df.loc[i, 'MonthlyCharges'] * df.loc[i, 'tenure']))
    
    df['TotalCharges'] = total_charges
    
    # Save to CSV
    csv_path = tmp_path / "telco_test.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path), str(tmp_path / "artifacts")


def test_pipeline_initialization(sample_telco_data):
    """Test DataPipeline initialization"""
    data_path, output_dir = sample_telco_data
    pipeline = DataPipeline(
        data_path=data_path,
        output_dir=output_dir,
        save_intermediates=False
    )
    
    assert pipeline is not None
    assert pipeline.data_path == data_path
    assert pipeline.output_dir == output_dir


def test_pipeline_run_creates_artifacts(sample_telco_data):
    """Test that pipeline creates all required artifacts"""
    data_path, output_dir = sample_telco_data
    pipeline = DataPipeline(
        data_path=data_path,
        output_dir=output_dir,
        save_intermediates=False
    )
    
    X_train, X_test, y_train, y_test = pipeline.run(save_artifacts=True)
    
    # Check return values
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    
    # Check shapes
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    
    # Check that artifacts were saved
    assert os.path.exists(f'{output_dir}/X_train.csv')
    assert os.path.exists(f'{output_dir}/X_test.csv')
    assert os.path.exists(f'{output_dir}/y_train.csv')
    assert os.path.exists(f'{output_dir}/y_test.csv')
    assert os.path.exists(f'{output_dir}/scaler.pkl')
    assert os.path.exists(f'{output_dir}/label_encoders.pkl')
    assert os.path.exists(f'{output_dir}/target_encoder.pkl')
    assert os.path.exists(f'{output_dir}/feature_names.pkl')


def test_pipeline_train_test_split_ratio(sample_telco_data):
    """Test that train/test split maintains correct ratio"""
    data_path, output_dir = sample_telco_data
    pipeline = DataPipeline(
        data_path=data_path,
        output_dir=output_dir,
        save_intermediates=False
    )
    
    X_train, X_test, y_train, y_test = pipeline.run(save_artifacts=False)
    
    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples
    
    # Should be approximately 0.2 (20%)
    assert 0.15 < test_ratio < 0.25


def test_pipeline_no_missing_values(sample_telco_data):
    """Test that pipeline output has no missing values"""
    data_path, output_dir = sample_telco_data
    pipeline = DataPipeline(
        data_path=data_path,
        output_dir=output_dir,
        save_intermediates=False
    )
    
    X_train, X_test, y_train, y_test = pipeline.run(save_artifacts=False)
    
    # No missing values in features
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    
    # No missing values in target
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()


def test_pipeline_feature_scaling(sample_telco_data):
    """Test that features are properly scaled"""
    data_path, output_dir = sample_telco_data
    pipeline = DataPipeline(
        data_path=data_path,
        output_dir=output_dir,
        save_intermediates=False
    )
    
    X_train, X_test, y_train, y_test = pipeline.run(save_artifacts=False)
    
    # Check that features are scaled (mean ≈ 0, std ≈ 1)
    # Allow some tolerance for small datasets
    train_means = X_train.mean()
    train_stds = X_train.std()
    
    # Most features should have mean close to 0
    assert (train_means.abs() < 0.5).sum() > X_train.shape[1] * 0.8
    # Most features should have std close to 1
    assert ((train_stds > 0.5) & (train_stds < 1.5)).sum() > X_train.shape[1] * 0.8


def test_pipeline_target_encoding(sample_telco_data):
    """Test that target is properly encoded"""
    data_path, output_dir = sample_telco_data
    pipeline = DataPipeline(
        data_path=data_path,
        output_dir=output_dir,
        save_intermediates=False
    )
    
    X_train, X_test, y_train, y_test = pipeline.run(save_artifacts=False)
    
    # Target should be binary (0 and 1)
    assert set(y_train) <= {0, 1}
    assert set(y_test) <= {0, 1}
    
    # Should have both classes
    assert len(np.unique(y_train)) == 2
