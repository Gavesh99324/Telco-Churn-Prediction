"""Model architecture for Telco Customer Churn"""
import logging
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBuilder:
    """Build ML models for churn prediction"""

    def __init__(self, random_state: int = 42):
        """
        Initialize ModelBuilder
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model_configs = self._define_model_configs()

    def _define_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Define configurations for all models
        
        Returns:
            Dictionary of model configurations
        """
        return {
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': self.random_state
            },
            'decision_tree': {
                'max_depth': 10,
                'min_samples_split': 20,
                'random_state': self.random_state
            },
            'random_forest': {
                'n_estimators': 100,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': self.random_state
            },
            'svm': {
                'kernel': 'rbf',
                'probability': True,
                'random_state': self.random_state
            }
        }

    def build_logistic_regression(self, **kwargs) -> LogisticRegression:
        """Build Logistic Regression model"""
        config = {**self.model_configs['logistic_regression'], **kwargs}
        logger.info(f"Building Logistic Regression with params: {config}")
        return LogisticRegression(**config)

    def build_decision_tree(self, **kwargs) -> DecisionTreeClassifier:
        """Build Decision Tree model"""
        config = {**self.model_configs['decision_tree'], **kwargs}
        logger.info(f"Building Decision Tree with params: {config}")
        return DecisionTreeClassifier(**config)

    def build_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Build Random Forest model"""
        config = {**self.model_configs['random_forest'], **kwargs}
        logger.info(f"Building Random Forest with params: {config}")
        return RandomForestClassifier(**config)

    def build_gradient_boosting(self, **kwargs) -> GradientBoostingClassifier:
        """Build Gradient Boosting model"""
        config = {**self.model_configs['gradient_boosting'], **kwargs}
        logger.info(f"Building Gradient Boosting with params: {config}")
        return GradientBoostingClassifier(**config)

    def build_svm(self, **kwargs) -> SVC:
        """Build SVM model"""
        config = {**self.model_configs['svm'], **kwargs}
        logger.info(f"Building SVM with params: {config}")
        return SVC(**config)

    def build_voting_classifier(self, use_smote_params: bool = False) -> VotingClassifier:
        """
        Build Voting Classifier ensemble
        
        Args:
            use_smote_params: If True, use tuned params for SMOTE data
            
        Returns:
            VotingClassifier
        """
        if use_smote_params:
            estimators = [
                ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state)),
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                             random_state=self.random_state, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                                 max_depth=5, random_state=self.random_state))
            ]
        else:
            estimators = [
                ('lr', self.build_logistic_regression()),
                ('rf', self.build_random_forest()),
                ('gb', self.build_gradient_boosting())
            ]
        
        logger.info("Building Voting Classifier (soft voting)")
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

    def build_stacking_classifier(self, use_smote_params: bool = False) -> StackingClassifier:
        """
        Build Stacking Classifier ensemble
        
        Args:
            use_smote_params: If True, use tuned params for SMOTE data
            
        Returns:
            StackingClassifier
        """
        if use_smote_params:
            estimators = [
                ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, 
                                             random_state=self.random_state, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                                 max_depth=5, random_state=self.random_state))
            ]
        else:
            estimators = [
                ('lr', self.build_logistic_regression()),
                ('rf', self.build_random_forest()),
                ('gb', self.build_gradient_boosting())
            ]
        
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)
        
        logger.info("Building Stacking Classifier (5-fold CV)")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )

    def build_all_models(self) -> Dict[str, Any]:
        """
        Build all baseline models
        
        Returns:
            Dictionary of model name to model instance
        """
        logger.info("="*70)
        logger.info("BUILDING ALL BASELINE MODELS")
        logger.info("="*70)
        
        models = {
            'Logistic Regression': self.build_logistic_regression(),
            'Decision Tree': self.build_decision_tree(),
            'Random Forest': self.build_random_forest(),
            'Gradient Boosting': self.build_gradient_boosting(),
            'SVM': self.build_svm()
        }
        
        logger.info(f"✅ Built {len(models)} models")
        return models
