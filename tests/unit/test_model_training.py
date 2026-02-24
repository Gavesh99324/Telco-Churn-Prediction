"""Unit tests for model_training.py"""
import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model_training import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    # Create imbalanced target (80-20 split)
    y_train = np.array([0] * 80 + [1] * 20)
    np.random.shuffle(y_train)

    return X_train, y_train


@pytest.fixture
def balanced_data():
    """Create balanced training data"""
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    y_train = np.array([0] * 50 + [1] * 50)
    np.random.shuffle(y_train)

    return X_train, y_train


@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance"""
    return ModelTrainer(random_state=42)


@pytest.fixture
def simple_model():
    """Create simple model for testing"""
    return LogisticRegression(random_state=42, max_iter=1000)


class TestModelTrainerInitialization:
    """Test ModelTrainer initialization"""

    def test_initialization(self, model_trainer):
        """Test proper initialization"""
        assert model_trainer.random_state == 42
        assert isinstance(model_trainer.training_history, list)
        assert len(model_trainer.training_history) == 0

    def test_custom_random_state(self):
        """Test initialization with custom random state"""
        trainer = ModelTrainer(random_state=123)
        assert trainer.random_state == 123


class TestBasicTraining:
    """Test basic model training"""

    def test_train_simple_model(self, model_trainer, simple_model, sample_data):
        """Test training a simple model"""
        X_train, y_train = sample_data

        trained_model, info = model_trainer.train(
            simple_model, X_train, y_train, model_name="TestModel"
        )

        # Check model is trained
        assert hasattr(trained_model, 'coef_')
        assert trained_model.classes_ is not None

        # Check training info
        assert info['model_name'] == "TestModel"
        assert 'training_time' in info
        assert info['training_time'] > 0
        assert info['n_samples'] == 100
        assert info['n_features'] == 5

    def test_train_updates_history(self, model_trainer, simple_model, sample_data):
        """Test that training updates history"""
        X_train, y_train = sample_data

        assert len(model_trainer.training_history) == 0

        model_trainer.train(simple_model, X_train, y_train)
        assert len(model_trainer.training_history) == 1

        # Train another model
        model_trainer.train(LogisticRegression(
            random_state=42), X_train, y_train)
        assert len(model_trainer.training_history) == 2

    def test_train_different_model_types(self, model_trainer, sample_data):
        """Test training different model types"""
        X_train, y_train = sample_data

        # Train Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        trained_lr, info_lr = model_trainer.train(
            lr_model, X_train, y_train, "LR")
        assert isinstance(trained_lr, LogisticRegression)

        # Train Random Forest
        rf_model = RandomForestClassifier(random_state=42, n_estimators=10)
        trained_rf, info_rf = model_trainer.train(
            rf_model, X_train, y_train, "RF")
        assert isinstance(trained_rf, RandomForestClassifier)

    def test_train_records_time(self, model_trainer, simple_model, sample_data):
        """Test that training time is recorded"""
        X_train, y_train = sample_data

        _, info = model_trainer.train(simple_model, X_train, y_train)

        assert 'training_time' in info
        assert isinstance(info['training_time'], float)
        assert info['training_time'] > 0
        assert info['training_time'] < 60  # Should be fast for toy data


class TestSMOTETraining:
    """Test training with SMOTE"""

    def test_train_with_smote_basic(self, model_trainer, simple_model, sample_data):
        """Test basic SMOTE training"""
        X_train, y_train = sample_data

        trained_model, info = model_trainer.train_with_smote(
            simple_model, X_train, y_train, model_name="SMOTE_Model"
        )

        # Check model is trained
        assert hasattr(trained_model, 'coef_')

        # Check info contains SMOTE details
        assert info['model_name'] == "SMOTE_Model"
        assert 'n_samples_original' in info
        assert 'n_samples_smote' in info
        assert 'smote_applied' in info
        assert info['smote_applied'] is True

    def test_train_with_smote_balances_classes(self, model_trainer, simple_model, sample_data):
        """Test that SMOTE balances classes"""
        X_train, y_train = sample_data

        # Check original imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        original_ratio = counts.min() / counts.max()
        assert original_ratio < 0.5  # Imbalanced

        _, info = model_trainer.train_with_smote(
            simple_model, X_train, y_train)

        # After SMOTE, samples should increase
        assert info['n_samples_smote'] > info['n_samples_original']

    def test_train_with_smote_custom_k_neighbors(self, model_trainer, simple_model, sample_data):
        """Test SMOTE with custom k_neighbors"""
        X_train, y_train = sample_data

        trained_model, info = model_trainer.train_with_smote(
            simple_model, X_train, y_train, k_neighbors=3
        )

        assert info['smote_applied'] is True
        assert hasattr(trained_model, 'coef_')

    def test_train_with_smote_records_distribution(self, model_trainer, simple_model, sample_data):
        """Test that original and SMOTE distributions are recorded"""
        X_train, y_train = sample_data

        _, info = model_trainer.train_with_smote(
            simple_model, X_train, y_train)

        assert 'class_distribution_original' in info
        assert 'class_distribution_smote' in info

        # Check distributions are dictionaries
        assert isinstance(info['class_distribution_original'], dict)
        assert isinstance(info['class_distribution_smote'], dict)

    def test_train_with_smote_on_balanced_data(self, model_trainer, simple_model, balanced_data):
        """Test SMOTE on already balanced data"""
        X_train, y_train = balanced_data

        trained_model, info = model_trainer.train_with_smote(
            simple_model, X_train, y_train
        )

        # SMOTE should still work but not change much
        assert info['smote_applied'] is True
        assert hasattr(trained_model, 'coef_')


class TestHyperparameterTuning:
    """Test hyperparameter tuning methods"""

    def test_hyperparameter_tuning_grid_basic(self, model_trainer, sample_data):
        """Test basic grid search"""
        X_train, y_train = sample_data

        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l2']
        }

        best_model, info = model_trainer.hyperparameter_tuning_grid(
            model, param_grid, X_train, y_train, cv=2
        )

        # Check best model is returned
        assert hasattr(best_model, 'best_estimator_')
        assert hasattr(best_model, 'best_params_')
        assert hasattr(best_model, 'best_score_')

        # Check info
        assert info['method'] == 'GridSearchCV'
        assert 'best_params' in info
        assert 'best_score' in info
        assert info['cv_folds'] == 2

    def test_hyperparameter_tuning_grid_best_params(self, model_trainer, sample_data):
        """Test that best parameters are found"""
        X_train, y_train = sample_data

        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0]
        }

        best_model, info = model_trainer.hyperparameter_tuning_grid(
            model, param_grid, X_train, y_train, cv=3
        )

        # Best C should be in param_grid
        assert info['best_params']['C'] in param_grid['C']
        assert info['best_score'] > 0

    def test_hyperparameter_tuning_random_basic(self, model_trainer, sample_data):
        """Test basic random search"""
        X_train, y_train = sample_data

        model = RandomForestClassifier(random_state=42)
        param_distributions = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }

        best_model, info = model_trainer.hyperparameter_tuning_random(
            model, param_distributions, X_train, y_train, n_iter=2, cv=2
        )

        # Check best model is returned
        assert hasattr(best_model, 'best_estimator_')

        # Check info
        assert info['method'] == 'RandomizedSearchCV'
        assert 'best_params' in info
        assert info['n_iterations'] == 2

    def test_hyperparameter_tuning_custom_scoring(self, model_trainer, sample_data):
        """Test tuning with custom scoring metric"""
        X_train, y_train = sample_data

        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {'C': [0.1, 1.0]}

        best_model, info = model_trainer.hyperparameter_tuning_grid(
            model, param_grid, X_train, y_train, scoring='f1', cv=2
        )

        assert info['scoring'] == 'f1'
        assert 'best_score' in info

    def test_hyperparameter_tuning_updates_history(self, model_trainer, sample_data):
        """Test that tuning updates training history"""
        X_train, y_train = sample_data

        initial_history_len = len(model_trainer.training_history)

        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {'C': [1.0]}

        model_trainer.hyperparameter_tuning_grid(
            model, param_grid, X_train, y_train, cv=2
        )

        assert len(model_trainer.training_history) > initial_history_len


class TestModelPersistence:
    """Test model saving and loading"""

    def test_save_model(self, model_trainer, simple_model, sample_data):
        """Test saving trained model"""
        X_train, y_train = sample_data

        # Train model
        trained_model, _ = model_trainer.train(simple_model, X_train, y_train)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model_trainer.save_model(trained_model, tmp_path)

            # Check file exists
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_model(self, model_trainer, simple_model, sample_data):
        """Test loading saved model"""
        X_train, y_train = sample_data

        # Train and save
        trained_model, _ = model_trainer.train(simple_model, X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model_trainer.save_model(trained_model, tmp_path)

            # Load model
            loaded_model = model_trainer.load_model(tmp_path)

            # Check predictions match
            original_pred = trained_model.predict(X_train)
            loaded_pred = loaded_model.predict(X_train)

            np.testing.assert_array_equal(original_pred, loaded_pred)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_and_load_preserves_attributes(self, model_trainer, simple_model, sample_data):
        """Test that model attributes are preserved after save/load"""
        X_train, y_train = sample_data

        trained_model, _ = model_trainer.train(simple_model, X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model_trainer.save_model(trained_model, tmp_path)
            loaded_model = model_trainer.load_model(tmp_path)

            # Check key attributes
            assert loaded_model.random_state == trained_model.random_state
            np.testing.assert_array_equal(
                loaded_model.coef_, trained_model.coef_)
            np.testing.assert_array_equal(
                loaded_model.classes_, trained_model.classes_)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestTrainingHistory:
    """Test training history tracking"""

    def test_history_accumulation(self, model_trainer, sample_data):
        """Test that history accumulates across multiple trainings"""
        X_train, y_train = sample_data

        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            RandomForestClassifier(random_state=42, n_estimators=10)
        ]

        for i, model in enumerate(models):
            model_trainer.train(model, X_train, y_train, f"Model_{i}")

        assert len(model_trainer.training_history) == 2
        assert model_trainer.training_history[0]['model_name'] == "Model_0"
        assert model_trainer.training_history[1]['model_name'] == "Model_1"

    def test_history_contains_metadata(self, model_trainer, simple_model, sample_data):
        """Test that history contains all expected metadata"""
        X_train, y_train = sample_data

        model_trainer.train(simple_model, X_train, y_train, "TestModel")

        history_entry = model_trainer.training_history[0]

        required_keys = ['model_name',
                         'training_time', 'n_samples', 'n_features']
        for key in required_keys:
            assert key in history_entry


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_train_with_minimal_data(self, model_trainer, simple_model):
        """Test training with minimal data"""
        X_train = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])

        trained_model, info = model_trainer.train(
            simple_model, X_train, y_train)

        assert hasattr(trained_model, 'coef_')
        assert info['n_samples'] == 3
        assert info['n_features'] == 2

    def test_smote_with_insufficient_neighbors(self, model_trainer, simple_model):
        """Test SMOTE with k_neighbors larger than minority class"""
        X_train = pd.DataFrame(np.random.randn(10, 3))
        y_train = np.array([0] * 8 + [1] * 2)

        # k_neighbors=5 but only 2 minority samples - should handle gracefully
        trained_model, info = model_trainer.train_with_smote(
            simple_model, X_train, y_train, k_neighbors=1
        )

        assert info['smote_applied'] is True
