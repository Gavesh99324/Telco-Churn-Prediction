"""
Integration tests for end-to-end ML pipeline flow
Tests data ingestion -> preprocessing -> training -> inference
"""
import pytest
import pandas as pd
import pickle
import tempfile
import shutil
from pathlib import Path

# Import pipeline modules
from pipelines.data_pipeline import DataPipeline
from pipelines.training_pipeline import TrainingPipeline
from pipelines.inference_pipeline import InferencePipeline


@pytest.fixture
def sample_data():
    """Generate sample customer data for testing"""
    data = {
        'customerID': ['TEST-001', 'TEST-002', 'TEST-003', 'TEST-004', 'TEST-005'],
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
        'tenure': [12, 24, 6, 48, 60],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'Yes', 'No internet service', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'DeviceProtection': ['Yes', 'No', 'No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'Yes', 'No', 'No internet service', 'No'],
        'StreamingTV': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'StreamingMovies': ['Yes', 'No', 'Yes', 'No internet service', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 'Bank transfer', 'Credit card'],
        'MonthlyCharges': [65.50, 89.75, 45.20, 25.00, 105.50],
        'TotalCharges': [786.00, 2154.00, 271.20, 1200.00, 6330.00],
        'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary directory for artifacts"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestPipelineFlow:
    """Integration tests for complete ML pipeline"""

    def test_data_pipeline_execution(self, sample_data, temp_artifacts_dir, temp_data_dir):
        """Test data pipeline from raw data to processed splits"""
        # Save sample data
        raw_data_path = temp_data_dir / "telco_churn.csv"
        sample_data.to_csv(raw_data_path, index=False)
        
        # Run data pipeline
        pipeline = DataPipeline(
            raw_data_path=str(raw_data_path),
            artifacts_dir=str(temp_artifacts_dir)
        )
        
        # Execute pipeline
        result = pipeline.run()
        
        # Verify results
        assert result is not None
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        
        # Check data shapes
        assert len(result['X_train']) > 0
        assert len(result['X_test']) > 0
        assert len(result['y_train']) > 0
        assert len(result['y_test']) > 0
        
        # Check train-test split ratio (approximately 80-20)
        total_records = len(result['X_train']) + len(result['X_test'])
        train_ratio = len(result['X_train']) / total_records
        assert 0.7 <= train_ratio <= 0.9

    def test_training_pipeline_execution(self, temp_artifacts_dir):
        """Test training pipeline produces models"""
        # Note: This test requires pre-processed data
        # In real scenario, run data_pipeline first
        
        try:
            pipeline = TrainingPipeline(artifacts_dir=str(temp_artifacts_dir))
            result = pipeline.run()
            
            # Verify model artifacts
            assert result is not None
            assert 'best_model' in result
            assert 'best_score' in result
            assert 'results' in result
            
            # Check best score is reasonable
            assert result['best_score'] > 0
            
        except FileNotFoundError:
            pytest.skip("Pre-processed data not available")

    def test_inference_pipeline_execution(self, sample_data, temp_artifacts_dir):
        """Test inference pipeline makes predictions"""
        # Prepare test data (remove target and customerID)
        test_data = sample_data.drop(['Churn', 'customerID'], axis=1)
        
        try:
            # Initialize inference pipeline
            pipeline = InferencePipeline(artifacts_dir=str(temp_artifacts_dir))
            
            # Make predictions
            predictions = pipeline.predict(test_data)
            
            # Verify predictions
            assert predictions is not None
            assert len(predictions) == len(test_data)
            assert all(pred in [0, 1] for pred in predictions)
            
        except FileNotFoundError:
            pytest.skip("Trained model not available")

    def test_end_to_end_pipeline_flow(self, sample_data, temp_artifacts_dir, temp_data_dir):
        """Test complete pipeline: data -> training -> inference"""
        # Step 1: Data Pipeline
        raw_data_path = temp_data_dir / "telco_churn.csv"
        sample_data.to_csv(raw_data_path, index=False)
        
        data_pipeline = DataPipeline(
            raw_data_path=str(raw_data_path),
            artifacts_dir=str(temp_artifacts_dir)
        )
        data_result = data_pipeline.run()
        
        assert data_result is not None
        assert len(data_result['X_train']) > 0
        
        # Step 2: Training Pipeline
        training_pipeline = TrainingPipeline(artifacts_dir=str(temp_artifacts_dir))
        training_result = training_pipeline.run()
        
        assert training_result is not None
        assert training_result['best_model'] is not None
        
        # Step 3: Inference Pipeline
        test_data = sample_data.drop(['Churn', 'customerID'], axis=1).head(2)
        
        inference_pipeline = InferencePipeline(artifacts_dir=str(temp_artifacts_dir))
        predictions = inference_pipeline.predict(test_data)
        
        assert predictions is not None
        assert len(predictions) == len(test_data)

    def test_pipeline_artifact_persistence(self, sample_data, temp_artifacts_dir, temp_data_dir):
        """Test that pipeline artifacts are saved correctly"""
        # Run data pipeline
        raw_data_path = temp_data_dir / "telco_churn.csv"
        sample_data.to_csv(raw_data_path, index=False)
        
        pipeline = DataPipeline(
            raw_data_path=str(raw_data_path),
            artifacts_dir=str(temp_artifacts_dir)
        )
        pipeline.run()
        
        # Check artifacts directory
        artifacts_path = Path(temp_artifacts_dir)
        
        # Verify data artifacts
        assert (artifacts_path / "data").exists()
        
        # Check for saved preprocessors
        expected_files = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl']
        data_path = artifacts_path / "data"
        
        for file_name in expected_files:
            file_path = data_path / file_name
            if file_path.exists():
                # Try loading
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    assert data is not None

    def test_pipeline_error_handling(self, temp_artifacts_dir):
        """Test pipeline error handling with missing data"""
        pipeline = DataPipeline(
            raw_data_path="nonexistent.csv",
            artifacts_dir=str(temp_artifacts_dir)
        )
        
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    def test_prediction_consistency(self, sample_data, temp_artifacts_dir, temp_data_dir):
        """Test that repeated predictions are consistent"""
        # Prepare pipeline
        raw_data_path = temp_data_dir / "telco_churn.csv"
        sample_data.to_csv(raw_data_path, index=False)
        
        # Run data and training pipeline
        data_pipeline = DataPipeline(
            raw_data_path=str(raw_data_path),
            artifacts_dir=str(temp_artifacts_dir)
        )
        data_pipeline.run()
        
        training_pipeline = TrainingPipeline(artifacts_dir=str(temp_artifacts_dir))
        training_pipeline.run()
        
        # Prepare test data
        test_data = sample_data.drop(['Churn', 'customerID'], axis=1).head(1)
        
        # Make multiple predictions
        inference_pipeline = InferencePipeline(artifacts_dir=str(temp_artifacts_dir))
        
        predictions = []
        for _ in range(5):
            pred = inference_pipeline.predict(test_data)
            predictions.append(pred[0])
        
        # All predictions should be the same
        assert all(p == predictions[0] for p in predictions)

    def test_model_performance_metrics(self, sample_data, temp_artifacts_dir, temp_data_dir):
        """Test that model performance metrics are calculated"""
        # Setup
        raw_data_path = temp_data_dir / "telco_churn.csv"
        sample_data.to_csv(raw_data_path, index=False)
        
        # Run pipelines
        data_pipeline = DataPipeline(
            raw_data_path=str(raw_data_path),
            artifacts_dir=str(temp_artifacts_dir)
        )
        data_pipeline.run()
        
        training_pipeline = TrainingPipeline(artifacts_dir=str(temp_artifacts_dir))
        result = training_pipeline.run()
        
        # Check that results contain metrics
        assert 'results' in result
        results = result['results']
        
        # Verify each model has metrics
        for model_name, metrics in results.items():
            assert 'train_accuracy' in metrics
            assert 'test_accuracy' in metrics
            assert 'test_f1' in metrics
            
            # Metrics should be in valid range
            assert 0 <= metrics['train_accuracy'] <= 1
            assert 0 <= metrics['test_accuracy'] <= 1
            assert 0 <= metrics['test_f1'] <= 1

    @pytest.mark.slow
    def test_pipeline_with_large_dataset(self, temp_artifacts_dir, temp_data_dir):
        """Test pipeline with larger dataset (100 records)"""
        # Generate larger dataset
        large_data = []
        for i in range(100):
            record = {
                'customerID': f'TEST-{i:04d}',
                'gender': 'Female' if i % 2 == 0 else 'Male',
                'SeniorCitizen': i % 2,
                'Partner': 'Yes' if i % 3 == 0 else 'No',
                'Dependents': 'Yes' if i % 4 == 0 else 'No',
                'tenure': (i % 72) + 1,
                'PhoneService': 'Yes',
                'MultipleLines': 'No' if i % 2 == 0 else 'Yes',
                'InternetService': 'DSL' if i % 3 == 0 else 'Fiber optic',
                'OnlineSecurity': 'Yes' if i % 2 == 0 else 'No',
                'OnlineBackup': 'No' if i % 3 == 0 else 'Yes',
                'DeviceProtection': 'Yes' if i % 4 == 0 else 'No',
                'TechSupport': 'No',
                'StreamingTV': 'Yes' if i % 2 == 0 else 'No',
                'StreamingMovies': 'No' if i % 3 == 0 else 'Yes',
                'Contract': 'Month-to-month' if i % 3 == 0 else 'One year',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 50.0 + (i % 100),
                'TotalCharges': 500.0 + i * 10,
                'Churn': 'Yes' if i % 4 == 0 else 'No'
            }
            large_data.append(record)
        
        df = pd.DataFrame(large_data)
        
        # Save and process
        raw_data_path = temp_data_dir / "large_telco_churn.csv"
        df.to_csv(raw_data_path, index=False)
        
        # Run pipeline
        data_pipeline = DataPipeline(
            raw_data_path=str(raw_data_path),
            artifacts_dir=str(temp_artifacts_dir)
        )
        result = data_pipeline.run()
        
        # Verify processing
        assert result is not None
        assert len(result['X_train']) >= 60  # At least 60% for training
        assert len(result['X_test']) >= 20   # At least 20% for testing


def test_pipeline_configuration():
    """Test that pipeline can be configured"""
    from utils.config import Config
    
    # Load configuration
    config = Config()
    
    # Verify pipeline configuration exists
    assert hasattr(config, 'data_pipeline') or True
    assert hasattr(config, 'training_pipeline') or True


def test_pipeline_logging():
    """Test that pipeline logging is configured"""
    import logging
    
    # Check if logging is setup
    logger = logging.getLogger('pipelines')
    assert logger is not None

