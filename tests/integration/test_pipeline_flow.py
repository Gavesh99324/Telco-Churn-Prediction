"""Integration tests for contract-safe end-to-end pipeline execution."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.data_pipeline import DataPipeline
from pipelines.training_pipeline import TrainingPipeline
from pipelines.inference_pipeline import InferencePipeline


@pytest.fixture
def sample_telco_dataframe() -> pd.DataFrame:
    """Generate a deterministic Telco-like dataset for end-to-end checks."""
    rng = np.random.default_rng(42)
    n = 240

    tenure = rng.integers(1, 73, size=n)
    monthly = rng.uniform(25.0, 120.0, size=n).round(2)

    df = pd.DataFrame({
        'customerID': [f'C{i:05d}' for i in range(n)],
        'gender': rng.choice(['Male', 'Female'], size=n),
        'SeniorCitizen': rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        'Partner': rng.choice(['Yes', 'No'], size=n),
        'Dependents': rng.choice(['Yes', 'No'], size=n, p=[0.3, 0.7]),
        'tenure': tenure,
        'PhoneService': rng.choice(['Yes', 'No'], size=n, p=[0.9, 0.1]),
        'MultipleLines': rng.choice(
            ['Yes', 'No', 'No phone service'], size=n, p=[0.45, 0.45, 0.10]
        ),
        'InternetService': rng.choice(['DSL', 'Fiber optic', 'No'], size=n),
        'OnlineSecurity': rng.choice(
            ['Yes', 'No', 'No internet service'], size=n
        ),
        'OnlineBackup': rng.choice(
            ['Yes', 'No', 'No internet service'], size=n
        ),
        'DeviceProtection': rng.choice(
            ['Yes', 'No', 'No internet service'], size=n
        ),
        'TechSupport': rng.choice(
            ['Yes', 'No', 'No internet service'], size=n
        ),
        'StreamingTV': rng.choice(
            ['Yes', 'No', 'No internet service'], size=n
        ),
        'StreamingMovies': rng.choice(
            ['Yes', 'No', 'No internet service'], size=n
        ),
        'Contract': rng.choice(
            ['Month-to-month', 'One year', 'Two year'],
            size=n,
            p=[0.6, 0.25, 0.15]
        ),
        'PaperlessBilling': rng.choice(['Yes', 'No'], size=n),
        'PaymentMethod': rng.choice(
            [
                'Electronic check',
                'Mailed check',
                'Bank transfer (automatic)',
                'Credit card (automatic)',
            ],
            size=n,
        ),
        'MonthlyCharges': monthly,
        'TotalCharges': (monthly * tenure).round(2).astype(str),
    })

    # Build a churn label with meaningful but not extreme class imbalance.
    churn_score = (
        (df['Contract'] == 'Month-to-month').astype(int)
        + (df['InternetService'] == 'Fiber optic').astype(int)
        + (df['TechSupport'] == 'No').astype(int)
        + (df['tenure'] < 18).astype(int)
    )
    threshold = np.quantile(churn_score, 0.70)
    df['Churn'] = np.where(churn_score >= threshold, 'Yes', 'No')

    return df


def test_end_to_end_pipeline_contract_and_manifest(tmp_path, sample_telco_dataframe):
    """Run data->train->infer and validate quality and artifact contracts."""
    raw_data_path = tmp_path / 'telco_churn.csv'
    data_dir = tmp_path / 'artifacts' / 'data'
    model_dir = tmp_path / 'artifacts' / 'models'
    output_dir = tmp_path / 'artifacts' / 'predictions'

    sample_telco_dataframe.to_csv(raw_data_path, index=False)

    data_pipeline = DataPipeline(
        data_path=str(raw_data_path),
        output_dir=str(data_dir),
        save_intermediates=False,
    )
    X_train, X_test, y_train, y_test = data_pipeline.run(save_artifacts=True)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    schema_contract_path = data_dir / 'schema_contract.json'
    quality_report_path = data_dir / 'data_quality_report.json'
    assert schema_contract_path.exists()
    assert quality_report_path.exists()

    with open(quality_report_path, 'r', encoding='utf-8') as f:
        quality_report = json.load(f)
    assert quality_report['null_ratio'] <= quality_report['thresholds']['max_null_ratio']

    training_pipeline = TrainingPipeline(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        enable_mlflow=False,
        environment='development',
    )
    training_results = training_pipeline.run()

    assert training_results['best_model_name']
    assert training_results['best_f1_score'] > 0

    training_quality_path = model_dir / 'training_quality_report.json'
    manifest_path = model_dir / 'model_manifest.json'
    best_model_path = model_dir / 'best_model.pkl'

    assert training_quality_path.exists()
    assert manifest_path.exists()
    assert best_model_path.exists()

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    assert manifest['model_name'] == training_results['best_model_name']
    assert 'artifact_hashes' in manifest
    assert set(manifest['artifact_hashes'].keys()) == {
        'best_model.pkl',
        'feature_names.pkl',
        'label_encoders.pkl',
        'scaler.pkl',
    }

    inference_pipeline = InferencePipeline(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        output_dir=str(output_dir),
    )
    inference_results = inference_pipeline.run('best_model.pkl')

    assert len(inference_results) == len(pd.read_csv(data_dir / 'X_test.csv'))
    assert {'actual_churn', 'predicted_churn', 'churn_probability'}.issubset(
        inference_results.columns
    )


def test_manifest_compatibility_guard_detects_tampering(tmp_path, sample_telco_dataframe):
    """Tamper with feature artifacts and verify manifest guard fails fast."""
    raw_data_path = tmp_path / 'telco_churn.csv'
    data_dir = tmp_path / 'artifacts' / 'data'
    model_dir = tmp_path / 'artifacts' / 'models'

    sample_telco_dataframe.to_csv(raw_data_path, index=False)

    DataPipeline(
        data_path=str(raw_data_path),
        output_dir=str(data_dir),
    ).run(save_artifacts=True)

    TrainingPipeline(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        enable_mlflow=False,
        environment='development',
    ).run()

    # Tamper with feature_names artifact after manifest generation.
    feature_names_path = data_dir / 'feature_names.pkl'
    with open(feature_names_path, 'rb') as f:
        feature_blob = pickle.load(f)
    feature_blob['feature_names'] = list(feature_blob['feature_names'])[:-1]
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_blob, f)

    pipeline = InferencePipeline(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        output_dir=str(tmp_path / 'artifacts' / 'predictions'),
    )

    pipeline.load_preprocessors()
    pipeline.load_model('best_model.pkl')
    with pytest.raises(ValueError, match='Artifact compatibility validation failed'):
        pipeline.validate_manifest_compatibility()
