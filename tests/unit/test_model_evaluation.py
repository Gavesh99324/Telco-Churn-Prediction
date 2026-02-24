"""Unit tests for model_evaluation.py"""
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.model_evaluation import ModelEvaluator


@pytest.fixture
def sample_data():
    """Create sample train/test data"""
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    y_train = np.random.randint(0, 2, 100)
    
    X_test = pd.DataFrame(
        np.random.randn(30, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    y_test = np.random.randint(0, 2, 30)
    
    return X_train, y_train, X_test, y_test


@pytest.fixture
def trained_model(sample_data):
    """Create and train a simple model"""
    X_train, y_train, _, _ = sample_data
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def multiple_trained_models(sample_data):
    """Create and train multiple models"""
    X_train, y_train, _, _ = sample_data
    
    models = {}
    models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000)
    models['RandomForest'] = RandomForestClassifier(random_state=42, n_estimators=10)
    models['DecisionTree'] = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models


@pytest.fixture
def evaluator():
    """Create ModelEvaluator instance"""
    return ModelEvaluator()


class TestModelEvaluatorInitialization:
    """Test ModelEvaluator initialization"""
    
    def test_initialization(self, evaluator):
        """Test proper initialization"""
        assert isinstance(evaluator.evaluation_results, list)
        assert len(evaluator.evaluation_results) == 0


class TestBasicEvaluation:
    """Test basic model evaluation"""
    
    def test_evaluate_model_basic(self, evaluator, trained_model, sample_data):
        """Test basic model evaluation"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test,
            model_name="TestModel"
        )
        
        # Check all required metrics are present
        required_metrics = [
            'model_name', 'train_accuracy', 'test_accuracy',
            'train_precision', 'test_precision',
            'train_recall', 'test_recall',
            'train_f1', 'test_f1',
            'test_roc_auc', 'confusion_matrix'
        ]
        
        for metric in required_metrics:
            assert metric in results, f"Missing metric: {metric}"
    
    def test_evaluate_model_name(self, evaluator, trained_model, sample_data):
        """Test that model name is correctly stored"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test,
            model_name="MyCustomModel"
        )
        
        assert results['model_name'] == "MyCustomModel"
    
    def test_evaluate_model_metrics_range(self, evaluator, trained_model, sample_data):
        """Test that metrics are in valid ranges"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        # All metrics should be between 0 and 1
        metrics_to_check = [
            'train_accuracy', 'test_accuracy',
            'train_precision', 'test_precision',
            'train_recall', 'test_recall',
            'train_f1', 'test_f1',
            'test_roc_auc'
        ]
        
        for metric in metrics_to_check:
            if results[metric] is not None:
                assert 0 <= results[metric] <= 1, f"{metric} out of range"
    
    def test_evaluate_model_confusion_matrix(self, evaluator, trained_model, sample_data):
        """Test confusion matrix generation"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        cm = results['confusion_matrix']
        assert cm.shape == (2, 2)  # Binary classification
        assert cm.sum() == len(y_test)  # Total equals test samples
        assert np.all(cm >= 0)  # No negative values
    
    def test_evaluate_model_predictions(self, evaluator, trained_model, sample_data):
        """Test that predictions are stored"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        assert 'predictions' in results
        assert len(results['predictions']) == len(y_test)
        assert np.all(np.isin(results['predictions'], [0, 1]))
    
    def test_evaluate_model_probabilities(self, evaluator, trained_model, sample_data):
        """Test that probabilities are stored"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        assert 'probabilities' in results
        if results['probabilities'] is not None:
            assert len(results['probabilities']) == len(y_test)
            assert np.all((results['probabilities'] >= 0) & (results['probabilities'] <= 1))


class TestMultipleModelEvaluation:
    """Test evaluation of multiple models"""
    
    def test_evaluate_multiple_models(self, evaluator, multiple_trained_models, sample_data):
        """Test evaluating multiple models"""
        X_train, y_train, X_test, y_test = sample_data
        
        for name, model in multiple_trained_models.items():
            evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name=name
            )
        
        assert len(evaluator.evaluation_results) == 3
        
        # Check all model names are unique
        names = [r['model_name'] for r in evaluator.evaluation_results]
        assert len(names) == len(set(names))
    
    def test_evaluation_results_accumulate(self, evaluator, trained_model, sample_data):
        """Test that evaluation results accumulate"""
        X_train, y_train, X_test, y_test = sample_data
        
        assert len(evaluator.evaluation_results) == 0
        
        evaluator.evaluate_model(trained_model, X_train, y_train, X_test, y_test, "Model1")
        assert len(evaluator.evaluation_results) == 1
        
        evaluator.evaluate_model(trained_model, X_train, y_train, X_test, y_test, "Model2")
        assert len(evaluator.evaluation_results) == 2


class TestModelComparison:
    """Test model comparison functionality"""
    
    def test_compare_models_basic(self, evaluator, multiple_trained_models, sample_data):
        """Test basic model comparison"""
        X_train, y_train, X_test, y_test = sample_data
        
        # Evaluate all models
        for name, model in multiple_trained_models.items():
            evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name=name
            )
        
        # Compare models
        comparison_df = evaluator.compare_models()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert 'Model' in comparison_df.columns
    
    def test_compare_models_metrics(self, evaluator, multiple_trained_models, sample_data):
        """Test that comparison includes all metrics"""
        X_train, y_train, X_test, y_test = sample_data
        
        for name, model in multiple_trained_models.items():
            evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name=name
            )
        
        comparison_df = evaluator.compare_models()
        
        # Check key metrics are present
        expected_columns = ['Model', 'Test_Accuracy', 'Test_F1', 'Test_ROC_AUC']
        for col in expected_columns:
            assert col in comparison_df.columns
    
    def test_compare_models_empty(self, evaluator):
        """Test comparison with no evaluated models"""
        comparison_df = evaluator.compare_models()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 0
    
    def test_compare_models_sorted_by_f1(self, evaluator, multiple_trained_models, sample_data):
        """Test that models are sorted by F1 score"""
        X_train, y_train, X_test, y_test = sample_data
        
        for name, model in multiple_trained_models.items():
            evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name=name
            )
        
        comparison_df = evaluator.compare_models(sort_by='Test_F1')
        
        # Check F1 scores are in descending order
        f1_scores = comparison_df['Test_F1'].values
        assert np.all(f1_scores[:-1] >= f1_scores[1:])


class TestConfusionMatrixPlotting:
    """Test confusion matrix plotting"""
    
    def test_plot_confusion_matrix_basic(self, evaluator, trained_model, sample_data):
        """Test basic confusion matrix plotting"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        # Plot should not raise error
        fig = evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            model_name="TestModel"
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix_labels(self, evaluator, trained_model, sample_data):
        """Test confusion matrix with custom labels"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        fig = evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            model_name="TestModel",
            labels=['No Churn', 'Churn']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix_normalized(self, evaluator, trained_model, sample_data):
        """Test normalized confusion matrix"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        fig = evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            model_name="TestModel",
            normalize=True
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestROCCurvePlotting:
    """Test ROC curve plotting"""
    
    def test_plot_roc_curve_single_model(self, evaluator, trained_model, sample_data):
        """Test plotting ROC curve for single model"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test,
            model_name="TestModel"
        )
        
        fig = evaluator.plot_roc_curves([results])
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_roc_curves_multiple_models(self, evaluator, multiple_trained_models, sample_data):
        """Test plotting ROC curves for multiple models"""
        X_train, y_train, X_test, y_test = sample_data
        
        for name, model in multiple_trained_models.items():
            evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name=name
            )
        
        fig = evaluator.plot_roc_curves(evaluator.evaluation_results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_roc_curves_empty_list(self, evaluator):
        """Test plotting with empty results list"""
        fig = evaluator.plot_roc_curves([])
        
        # Should handle gracefully
        if fig is not None:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestClassificationReport:
    """Test classification report generation"""
    
    def test_get_classification_report(self, evaluator, trained_model, sample_data):
        """Test classification report generation"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        report = evaluator.get_classification_report(
            y_test, 
            results['predictions']
        )
        
        assert isinstance(report, (str, dict))
    
    def test_classification_report_with_labels(self, evaluator, trained_model, sample_data):
        """Test classification report with custom labels"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        report = evaluator.get_classification_report(
            y_test,
            results['predictions'],
            target_names=['No Churn', 'Churn']
        )
        
        assert isinstance(report, (str, dict))


class TestMetricCalculation:
    """Test individual metric calculations"""
    
    def test_calculate_accuracy(self, evaluator):
        """Test accuracy calculation"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.8  # 4 out of 5 correct
    
    def test_calculate_precision_recall_f1(self, evaluator):
        """Test precision, recall, and F1 calculation"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_calculate_roc_auc_with_probabilities(self, evaluator):
        """Test ROC-AUC calculation with probabilities"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] is not None
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_calculate_metrics_without_probabilities(self, evaluator):
        """Test metric calculation without probabilities"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba=None)
        
        assert metrics['roc_auc'] is None


class TestPredictionTiming:
    """Test prediction timing"""
    
    def test_records_prediction_time(self, evaluator, trained_model, sample_data):
        """Test that prediction times are recorded"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        assert 'train_predict_time' in results
        assert 'test_predict_time' in results
        assert results['train_predict_time'] > 0
        assert results['test_predict_time'] > 0
    
    def test_prediction_time_reasonable(self, evaluator, trained_model, sample_data):
        """Test that prediction times are reasonable"""
        X_train, y_train, X_test, y_test = sample_data
        
        results = evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test
        )
        
        # Predictions should be fast for small data
        assert results['train_predict_time'] < 10  # seconds
        assert results['test_predict_time'] < 10


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_evaluate_with_minimal_data(self, evaluator):
        """Test evaluation with minimal data"""
        X_train = pd.DataFrame([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        X_test = pd.DataFrame([[5, 6]])
        y_test = np.array([1])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        results = evaluator.evaluate_model(
            model, X_train, y_train, X_test, y_test
        )
        
        assert results is not None
        assert 'test_accuracy' in results
    
    def test_evaluate_perfect_predictions(self, evaluator):
        """Test evaluation with perfect predictions"""
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = np.array([0] * 25 + [1] * 25)
        X_test = pd.DataFrame(np.random.randn(20, 3))
        y_test = np.array([0] * 10 + [1] * 10)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        results = evaluator.evaluate_model(
            model, X_train, y_train, X_test, y_test
        )
        
        assert 0 <= results['test_accuracy'] <= 1
    
    def test_compare_models_with_single_model(self, evaluator, trained_model, sample_data):
        """Test comparison with only one model"""
        X_train, y_train, X_test, y_test = sample_data
        
        evaluator.evaluate_model(
            trained_model, X_train, y_train, X_test, y_test,
            model_name="SingleModel"
        )
        
        comparison_df = evaluator.compare_models()
        
        assert len(comparison_df) == 1
        assert comparison_df.iloc[0]['Model'] == "SingleModel"


class TestSaveComparison:
    """Test saving comparison results"""
    
    def test_save_comparison_to_csv(self, evaluator, multiple_trained_models, sample_data):
        """Test saving comparison to CSV"""
        X_train, y_train, X_test, y_test = sample_data
        
        for name, model in multiple_trained_models.items():
            evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name=name
            )
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            comparison_df = evaluator.compare_models()
            evaluator.save_comparison(comparison_df, tmp_path)
            
            # Read back and verify
            loaded_df = pd.read_csv(tmp_path)
            assert len(loaded_df) == 3
            assert 'Model' in loaded_df.columns
        finally:
            import os
            os.unlink(tmp_path)
