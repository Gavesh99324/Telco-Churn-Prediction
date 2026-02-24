"""Unit tests for model_inference.py"""
import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model_inference import ModelInference


@pytest.fixture
def sample_data():
    """Create sample data for inference"""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(20, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    y = np.random.randint(0, 2, 20)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create and train a model"""
    X, y = sample_data
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_rf_model(sample_data):
    """Create and train a Random Forest model"""
    X, y = sample_data
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)
    return model


@pytest.fixture
def saved_model_path(trained_model):
    """Save model to temporary file"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
    
    with open(tmp_path, 'wb') as f:
        pickle.dump(trained_model, f)
    
    yield tmp_path
    
    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


class TestModelInferenceInitialization:
    """Test ModelInference initialization"""
    
    def test_initialization_without_model(self):
        """Test initialization without loading model"""
        inference = ModelInference()
        
        assert inference.model is None
        assert inference.model_path is None
    
    def test_initialization_with_model_path(self, saved_model_path):
        """Test initialization with model path"""
        inference = ModelInference(model_path=saved_model_path)
        
        assert inference.model is not None
        assert inference.model_path == saved_model_path
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with invalid path raises error"""
        with pytest.raises(FileNotFoundError):
            ModelInference(model_path="nonexistent_model.pkl")


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_model(self, saved_model_path):
        """Test loading model from file"""
        inference = ModelInference()
        inference.load_model(saved_model_path)
        
        assert inference.model is not None
        assert isinstance(inference.model, LogisticRegression)
    
    def test_load_model_updates_path(self, saved_model_path):
        """Test that loading updates model_path"""
        inference = ModelInference()
        inference.load_model(saved_model_path)
        
        assert inference.model_path == saved_model_path
    
    def test_load_model_invalid_path(self):
        """Test loading from invalid path raises error"""
        inference = ModelInference()
        
        with pytest.raises(FileNotFoundError):
            inference.load_model("invalid_path.pkl")
    
    def test_set_model_directly(self, trained_model):
        """Test setting model directly"""
        inference = ModelInference()
        inference.set_model(trained_model)
        
        assert inference.model is not None
        assert isinstance(inference.model, LogisticRegression)
    
    def test_set_model_without_path(self, trained_model):
        """Test that set_model doesn't set path"""
        inference = ModelInference()
        inference.set_model(trained_model)
        
        assert inference.model is not None
        assert inference.model_path is None


class TestPrediction:
    """Test prediction functionality"""
    
    def test_predict_with_dataframe(self, trained_model, sample_data):
        """Test prediction with DataFrame input"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        predictions = inference.predict(X)
        
        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [0, 1]))
    
    def test_predict_with_array(self, trained_model, sample_data):
        """Test prediction with array input"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        predictions = inference.predict(X.values)
        
        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [0, 1]))
    
    def test_predict_single_sample(self, trained_model, sample_data):
        """Test prediction on single sample"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        single_sample = X.iloc[[0]]
        prediction = inference.predict(single_sample)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_predict_without_model_raises_error(self, sample_data):
        """Test that predict without model raises error"""
        X, _ = sample_data
        inference = ModelInference()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            inference.predict(X)
    
    def test_predict_returns_numpy_array(self, trained_model, sample_data):
        """Test that predict returns numpy array"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        predictions = inference.predict(X)
        
        assert isinstance(predictions, np.ndarray)


class TestPredictProba:
    """Test probability prediction functionality"""
    
    def test_predict_proba_basic(self, trained_model, sample_data):
        """Test basic probability prediction"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        probabilities = inference.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(len(X)))
    
    def test_predict_proba_single_sample(self, trained_model, sample_data):
        """Test probability prediction for single sample"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        single_sample = X.iloc[[0]]
        proba = inference.predict_proba(single_sample)
        
        assert proba.shape == (1, 2)
        assert np.sum(proba) == pytest.approx(1.0)
    
    def test_predict_proba_without_model_raises_error(self, sample_data):
        """Test that predict_proba without model raises error"""
        X, _ = sample_data
        inference = ModelInference()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            inference.predict_proba(X)
    
    def test_predict_proba_positive_class(self, trained_model, sample_data):
        """Test getting probability for positive class"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        proba_positive = inference.predict_proba(X)[:, 1]
        
        assert len(proba_positive) == len(X)
        assert np.all((proba_positive >= 0) & (proba_positive <= 1))


class TestBatchPrediction:
    """Test batch prediction functionality"""
    
    def test_batch_predict_basic(self, trained_model, sample_data):
        """Test basic batch prediction"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.batch_predict(X, batch_size=5)
        
        assert 'predictions' in results
        assert 'probabilities' in results
        assert len(results['predictions']) == len(X)
    
    def test_batch_predict_with_custom_batch_size(self, trained_model, sample_data):
        """Test batch prediction with custom batch size"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.batch_predict(X, batch_size=3)
        
        assert len(results['predictions']) == len(X)
        assert results['batch_size'] == 3
    
    def test_batch_predict_includes_metadata(self, trained_model, sample_data):
        """Test that batch prediction includes metadata"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.batch_predict(X)
        
        assert 'n_samples' in results
        assert 'inference_time' in results
        assert results['n_samples'] == len(X)
    
    def test_batch_predict_performance(self, trained_model, sample_data):
        """Test batch prediction performance tracking"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.batch_predict(X)
        
        assert results['inference_time'] > 0
        assert 'samples_per_second' in results


class TestPredictWithMetadata:
    """Test prediction with metadata"""
    
    def test_predict_with_metadata_basic(self, trained_model, sample_data):
        """Test prediction with metadata"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.predict_with_metadata(X)
        
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'timestamps' in results
        assert 'confidence' in results
    
    def test_predict_with_metadata_timestamps(self, trained_model, sample_data):
        """Test that timestamps are included"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.predict_with_metadata(X)
        
        timestamps = results['timestamps']
        assert len(timestamps) == len(X)
        # Check timestamps are valid
        for ts in timestamps:
            assert isinstance(ts, str) or isinstance(ts, pd.Timestamp)
    
    def test_predict_with_metadata_confidence(self, trained_model, sample_data):
        """Test confidence scores calculation"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results = inference.predict_with_metadata(X)
        
        confidence = results['confidence']
        assert len(confidence) == len(X)
        assert np.all((confidence >= 0) & (confidence <= 1))
        # Confidence should be max probability
        probabilities = results['probabilities']
        expected_confidence = probabilities.max(axis=1)
        np.testing.assert_array_almost_equal(confidence, expected_confidence)


class TestExplainPrediction:
    """Test prediction explanation"""
    
    def test_explain_prediction_basic(self, trained_rf_model, sample_data):
        """Test basic prediction explanation"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_rf_model)
        
        explanation = inference.explain_prediction(X.iloc[0])
        
        assert 'prediction' in explanation
        assert 'probability' in explanation
        assert 'feature_importance' in explanation
    
    def test_explain_prediction_feature_importance(self, trained_rf_model, sample_data):
        """Test feature importance in explanation"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_rf_model)
        
        explanation = inference.explain_prediction(X.iloc[0])
        
        feature_importance = explanation['feature_importance']
        assert len(feature_importance) == X.shape[1]
        assert all(score >= 0 for score in feature_importance.values())
    
    def test_explain_prediction_with_feature_names(self, trained_rf_model, sample_data):
        """Test that feature names are included"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_rf_model)
        
        explanation = inference.explain_prediction(X.iloc[0])
        
        feature_importance = explanation['feature_importance']
        # Check feature names match
        for feature_name in X.columns:
            assert feature_name in feature_importance


class TestValidateInput:
    """Test input validation"""
    
    def test_validate_input_valid_dataframe(self, trained_model, sample_data):
        """Test validation with valid DataFrame"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        # Should not raise error
        is_valid = inference.validate_input(X)
        assert is_valid is True
    
    def test_validate_input_wrong_shape(self, trained_model, sample_data):
        """Test validation with wrong feature count"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        # Create data with wrong number of features
        X_wrong = X.iloc[:, :3]  # Only 3 features instead of 5
        
        with pytest.raises(ValueError):
            inference.validate_input(X_wrong)
    
    def test_validate_input_missing_values(self, trained_model, sample_data):
        """Test validation with missing values"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        # Add missing values
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        
        # Depending on implementation, might raise error or handle gracefully
        try:
            is_valid = inference.validate_input(X_with_nan)
            # If validation passes, it's handling NaNs
            assert is_valid in [True, False]
        except ValueError:
            # If it raises error, that's also valid
            pass
    
    def test_validate_input_empty_dataframe(self, trained_model):
        """Test validation with empty DataFrame"""
        inference = ModelInference()
        inference.set_model(trained_model)
        
        X_empty = pd.DataFrame()
        
        with pytest.raises(ValueError):
            inference.validate_input(X_empty)


class TestPredictionConsistency:
    """Test prediction consistency"""
    
    def test_predictions_consistent_across_calls(self, trained_model, sample_data):
        """Test that predictions are consistent"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        pred1 = inference.predict(X)
        pred2 = inference.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_probabilities_consistent_across_calls(self, trained_model, sample_data):
        """Test that probabilities are consistent"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        proba1 = inference.predict_proba(X)
        proba2 = inference.predict_proba(X)
        
        np.testing.assert_array_almost_equal(proba1, proba2)
    
    def test_loaded_model_predictions_match(self, saved_model_path, sample_data):
        """Test that loaded model gives same predictions"""
        X, _ = sample_data
        
        # Predict with first instance
        inference1 = ModelInference(model_path=saved_model_path)
        pred1 = inference1.predict(X)
        
        # Predict with second instance
        inference2 = ModelInference(model_path=saved_model_path)
        pred2 = inference2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_predict_empty_input(self, trained_model):
        """Test prediction with empty input"""
        inference = ModelInference()
        inference.set_model(trained_model)
        
        X_empty = pd.DataFrame(columns=[f'feature_{i}' for i in range(5)])
        
        predictions = inference.predict(X_empty)
        assert len(predictions) == 0
    
    def test_predict_large_batch(self, trained_model):
        """Test prediction on large batch"""
        np.random.seed(42)
        X_large = pd.DataFrame(
            np.random.randn(1000, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        inference = ModelInference()
        inference.set_model(trained_model)
        
        predictions = inference.predict(X_large)
        
        assert len(predictions) == 1000
        assert np.all(np.isin(predictions, [0, 1]))
    
    def test_switch_models(self, trained_model, trained_rf_model, sample_data):
        """Test switching between different models"""
        X, _ = sample_data
        inference = ModelInference()
        
        # Use first model
        inference.set_model(trained_model)
        pred1 = inference.predict(X)
        
        # Switch to second model
        inference.set_model(trained_rf_model)
        pred2 = inference.predict(X)
        
        # Predictions might differ
        assert len(pred1) == len(pred2)


class TestPredictionDataFrame:
    """Test DataFrame output format"""
    
    def test_predict_as_dataframe(self, trained_model, sample_data):
        """Test prediction output as DataFrame"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results_df = inference.predict_as_dataframe(X)
        
        assert isinstance(results_df, pd.DataFrame)
        assert 'prediction' in results_df.columns
        assert len(results_df) == len(X)
    
    def test_predict_as_dataframe_with_probabilities(self, trained_model, sample_data):
        """Test DataFrame includes probabilities"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results_df = inference.predict_as_dataframe(X, include_proba=True)
        
        assert 'probability_0' in results_df.columns
        assert 'probability_1' in results_df.columns
    
    def test_predict_as_dataframe_preserves_index(self, trained_model, sample_data):
        """Test that DataFrame preserves original index"""
        X, _ = sample_data
        inference = ModelInference()
        inference.set_model(trained_model)
        
        results_df = inference.predict_as_dataframe(X)
        
        pd.testing.assert_index_equal(results_df.index, X.index)
