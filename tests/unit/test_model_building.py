"""Unit tests for model_building.py"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from src.model_building import ModelBuilder


@pytest.fixture
def model_builder():
    """Create ModelBuilder instance"""
    return ModelBuilder(random_state=42)


class TestModelBuilderInitialization:
    """Test ModelBuilder initialization"""
    
    def test_initialization(self, model_builder):
        """Test proper initialization"""
        assert model_builder.random_state == 42
        assert isinstance(model_builder.model_configs, dict)
        assert len(model_builder.model_configs) == 5
    
    def test_model_configs_keys(self, model_builder):
        """Test model configuration keys"""
        expected_keys = {
            'logistic_regression', 
            'decision_tree', 
            'random_forest',
            'gradient_boosting', 
            'svm'
        }
        assert set(model_builder.model_configs.keys()) == expected_keys
    
    def test_random_state_in_configs(self, model_builder):
        """Test random_state is set in all configs"""
        for config_name, config in model_builder.model_configs.items():
            assert 'random_state' in config
            assert config['random_state'] == 42


class TestBaselineModelBuilders:
    """Test individual baseline model builders"""
    
    def test_build_logistic_regression(self, model_builder):
        """Test Logistic Regression builder"""
        model = model_builder.build_logistic_regression()
        
        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 1000
        assert model.random_state == 42
    
    def test_build_logistic_regression_custom_params(self, model_builder):
        """Test Logistic Regression with custom parameters"""
        model = model_builder.build_logistic_regression(max_iter=2000, C=0.5)
        
        assert model.max_iter == 2000
        assert model.C == 0.5
        assert model.random_state == 42
    
    def test_build_decision_tree(self, model_builder):
        """Test Decision Tree builder"""
        model = model_builder.build_decision_tree()
        
        assert isinstance(model, DecisionTreeClassifier)
        assert model.max_depth == 10
        assert model.min_samples_split == 20
        assert model.random_state == 42
    
    def test_build_decision_tree_custom_params(self, model_builder):
        """Test Decision Tree with custom parameters"""
        model = model_builder.build_decision_tree(max_depth=5, min_samples_leaf=10)
        
        assert model.max_depth == 5
        assert model.min_samples_leaf == 10
        assert model.random_state == 42
    
    def test_build_random_forest(self, model_builder):
        """Test Random Forest builder"""
        model = model_builder.build_random_forest()
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
        assert model.random_state == 42
        assert model.n_jobs == -1
    
    def test_build_random_forest_custom_params(self, model_builder):
        """Test Random Forest with custom parameters"""
        model = model_builder.build_random_forest(n_estimators=200, max_depth=15)
        
        assert model.n_estimators == 200
        assert model.max_depth == 15
        assert model.random_state == 42
    
    def test_build_gradient_boosting(self, model_builder):
        """Test Gradient Boosting builder"""
        model = model_builder.build_gradient_boosting()
        
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.random_state == 42
    
    def test_build_gradient_boosting_custom_params(self, model_builder):
        """Test Gradient Boosting with custom parameters"""
        model = model_builder.build_gradient_boosting(n_estimators=150, learning_rate=0.05)
        
        assert model.n_estimators == 150
        assert model.learning_rate == 0.05
        assert model.random_state == 42
    
    def test_build_svm(self, model_builder):
        """Test SVM builder"""
        model = model_builder.build_svm()
        
        assert isinstance(model, SVC)
        assert model.kernel == 'rbf'
        assert model.probability is True
        assert model.random_state == 42
    
    def test_build_svm_custom_params(self, model_builder):
        """Test SVM with custom parameters"""
        model = model_builder.build_svm(kernel='linear', C=0.8)
        
        assert model.kernel == 'linear'
        assert model.C == 0.8
        assert model.random_state == 42


class TestEnsembleModelBuilders:
    """Test ensemble model builders"""
    
    def test_build_voting_classifier_default(self, model_builder):
        """Test Voting Classifier with default parameters"""
        model = model_builder.build_voting_classifier(use_smote_params=False)
        
        assert isinstance(model, VotingClassifier)
        assert model.voting == 'soft'
        assert len(model.estimators) == 3
        
        # Check estimator names
        estimator_names = [name for name, _ in model.estimators]
        assert estimator_names == ['lr', 'rf', 'gb']
    
    def test_build_voting_classifier_smote_params(self, model_builder):
        """Test Voting Classifier with SMOTE parameters"""
        model = model_builder.build_voting_classifier(use_smote_params=True)
        
        assert isinstance(model, VotingClassifier)
        assert len(model.estimators) == 3
        
        # Check that estimators have specific SMOTE params
        lr, rf, gb = [est for _, est in model.estimators]
        assert isinstance(lr, LogisticRegression)
        assert isinstance(rf, RandomForestClassifier)
        assert isinstance(gb, GradientBoostingClassifier)
        
        # Check SMOTE-specific params
        assert rf.n_estimators == 200
        assert rf.max_depth == 20
        assert gb.n_estimators == 200
    
    def test_build_stacking_classifier_default(self, model_builder):
        """Test Stacking Classifier with default parameters"""
        model = model_builder.build_stacking_classifier(use_smote_params=False)
        
        assert isinstance(model, StackingClassifier)
        assert model.cv == 5
        assert len(model.estimators) == 3
        
        # Check meta learner
        assert isinstance(model.final_estimator, LogisticRegression)
        assert model.final_estimator.max_iter == 1000
    
    def test_build_stacking_classifier_smote_params(self, model_builder):
        """Test Stacking Classifier with SMOTE parameters"""
        model = model_builder.build_stacking_classifier(use_smote_params=True)
        
        assert isinstance(model, StackingClassifier)
        assert len(model.estimators) == 3
        
        # Check that estimators have SMOTE params
        lr, rf, gb = [est for _, est in model.estimators]
        assert rf.n_estimators == 100
        assert rf.max_depth == 20
        assert gb.max_depth == 5
    
    def test_voting_classifier_estimator_types(self, model_builder):
        """Test types of estimators in Voting Classifier"""
        model = model_builder.build_voting_classifier()
        
        lr, rf, gb = [est for _, est in model.estimators]
        assert isinstance(lr, LogisticRegression)
        assert isinstance(rf, RandomForestClassifier)
        assert isinstance(gb, GradientBoostingClassifier)
    
    def test_stacking_classifier_estimator_types(self, model_builder):
        """Test types of estimators in Stacking Classifier"""
        model = model_builder.build_stacking_classifier()
        
        lr, rf, gb = [est for _, est in model.estimators]
        assert isinstance(lr, LogisticRegression)
        assert isinstance(rf, RandomForestClassifier)
        assert isinstance(gb, GradientBoostingClassifier)


class TestBuildAllModels:
    """Test build_all_models method"""
    
    def test_build_all_models(self, model_builder):
        """Test building all models at once"""
        models = model_builder.build_all_models()
        
        assert isinstance(models, dict)
        assert len(models) == 5
        
        # Check all expected models are present
        expected_models = {
            'LogisticRegression',
            'DecisionTree',
            'RandomForest',
            'GradientBoosting',
            'SVM'
        }
        assert set(models.keys()) == expected_models
    
    def test_build_all_models_types(self, model_builder):
        """Test types of all built models"""
        models = model_builder.build_all_models()
        
        assert isinstance(models['LogisticRegression'], LogisticRegression)
        assert isinstance(models['DecisionTree'], DecisionTreeClassifier)
        assert isinstance(models['RandomForest'], RandomForestClassifier)
        assert isinstance(models['GradientBoosting'], GradientBoostingClassifier)
        assert isinstance(models['SVM'], SVC)
    
    def test_build_all_models_random_state(self, model_builder):
        """Test all models have consistent random_state"""
        models = model_builder.build_all_models()
        
        for model_name, model in models.items():
            assert hasattr(model, 'random_state')
            assert model.random_state == 42


class TestModelBuilderEdgeCases:
    """Test edge cases and error handling"""
    
    def test_different_random_states(self):
        """Test ModelBuilder with different random states"""
        builder1 = ModelBuilder(random_state=42)
        builder2 = ModelBuilder(random_state=123)
        
        assert builder1.random_state == 42
        assert builder2.random_state == 123
        
        model1 = builder1.build_random_forest()
        model2 = builder2.build_random_forest()
        
        assert model1.random_state == 42
        assert model2.random_state == 123
    
    def test_override_n_jobs(self, model_builder):
        """Test overriding n_jobs parameter"""
        model = model_builder.build_random_forest(n_jobs=4)
        assert model.n_jobs == 4
    
    def test_ensemble_with_custom_voting(self, model_builder):
        """Test that voting parameter is set correctly"""
        model = model_builder.build_voting_classifier()
        assert model.voting == 'soft'
    
    def test_ensemble_with_custom_cv(self, model_builder):
        """Test that CV parameter is set correctly in stacking"""
        model = model_builder.build_stacking_classifier()
        assert model.cv == 5


class TestParameterValidation:
    """Test parameter validation"""
    
    def test_svm_probability_true(self, model_builder):
        """Test SVM has probability=True for predict_proba"""
        model = model_builder.build_svm()
        assert model.probability is True
    
    def test_logistic_regression_max_iter(self, model_builder):
        """Test Logistic Regression has sufficient max_iter"""
        model = model_builder.build_logistic_regression()
        assert model.max_iter >= 1000
    
    def test_gradient_boosting_learning_rate(self, model_builder):
        """Test Gradient Boosting has reasonable learning rate"""
        model = model_builder.build_gradient_boosting()
        assert 0 < model.learning_rate <= 1
    
    def test_decision_tree_constraints(self, model_builder):
        """Test Decision Tree has depth constraints"""
        model = model_builder.build_decision_tree()
        assert model.max_depth is not None
        assert model.min_samples_split >= 2
