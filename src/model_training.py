"""Training logic for Telco Customer Churn models"""
import logging
import time
import pickle
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train ML models with various strategies"""

    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.training_history = []

    def train(self, model, X_train, y_train, model_name: str = "Model") -> Tuple[Any, Dict]:
        """
        Train a single model
        
        Args:
            model: Sklearn model instance
            X_train: Training features
            y_train: Training target
            model_name: Name for logging
            
        Returns:
            Tuple of (trained model, training info dict)
        """
        logger.info(f"Training {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        info = {
            'model_name': model_name,
            'training_time': train_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        self.training_history.append(info)
        logger.info(f"✅ {model_name} trained in {train_time:.2f}s")
        
        return model, info

    def train_with_smote(self, model, X_train, y_train, model_name: str = "Model",
                        k_neighbors: int = 5) -> Tuple[Any, Dict]:
        """
        Train model with SMOTE for class imbalance
        
        Args:
            model: Sklearn model instance
            X_train: Training features
            y_train: Training target
            model_name: Name for logging
            k_neighbors: Number of neighbors for SMOTE
            
        Returns:
            Tuple of (trained model, training info dict)
        """
        logger.info(f"Applying SMOTE to training data for {model_name}...")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"   Original distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # New distribution
        unique_smote, counts_smote = np.unique(y_train_smote, return_counts=True)
        logger.info(f"   SMOTE distribution: {dict(zip(unique_smote, counts_smote))}")
        
        # Train with SMOTE data
        start_time = time.time()
        model.fit(X_train_smote, y_train_smote)
        train_time = time.time() - start_time
        
        info = {
            'model_name': model_name,
            'training_time': train_time,
            'n_samples_original': len(X_train),
            'n_samples_smote': len(X_train_smote),
            'n_features': X_train.shape[1],
            'smote_applied': True
        }
        
        self.training_history.append(info)
        logger.info(f"✅ {model_name} trained with SMOTE in {train_time:.2f}s")
        
        return model, info

    def hyperparameter_tuning_grid(
        self,
        model,
        param_grid: Dict,
        X_train,
        y_train,
        model_name: str = "Model",
        cv: int = 5,
        scoring: str = 'f1',
        verbose: int = 2
    ) -> Tuple[Any, Dict]:
        """
        Perform GridSearchCV hyperparameter tuning
        
        Args:
            model: Sklearn model instance
            param_grid: Parameter grid for GridSearchCV
            X_train: Training features
            y_train: Training target
            model_name: Name for logging
            cv: Number of cross-validation folds
            scoring: Scoring metric
            verbose: Verbosity level
            
        Returns:
            Tuple of (best model, tuning info dict)
        """
        logger.info("="*70)
        logger.info(f"GRID SEARCH CV - {model_name}")
        logger.info("="*70)
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"CV folds: {cv}, Scoring: {scoring}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=-1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        info = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'search_time': search_time,
            'n_candidates': len(grid_search.cv_results_['params'])
        }
        
        logger.info("="*70)
        logger.info(f"✅ GRID SEARCH COMPLETE")
        logger.info(f"   Best {scoring}: {grid_search.best_score_:.4f}")
        logger.info(f"   Best params: {grid_search.best_params_}")
        logger.info(f"   Time: {search_time:.2f}s")
        logger.info("="*70)
        
        self.training_history.append(info)
        return grid_search.best_estimator_, info

    def hyperparameter_tuning_random(
        self,
        model,
        param_distributions: Dict,
        X_train,
        y_train,
        model_name: str = "Model",
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = 'f1',
        verbose: int = 2
    ) -> Tuple[Any, Dict]:
        """
        Perform RandomizedSearchCV hyperparameter tuning
        
        Args:
            model: Sklearn model instance
            param_distributions: Parameter distributions for RandomizedSearchCV
            X_train: Training features
            y_train: Training target
            model_name: Name for logging
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            scoring: Scoring metric
            verbose: Verbosity level
            
        Returns:
            Tuple of (best model, tuning info dict)
        """
        logger.info("="*70)
        logger.info(f"RANDOMIZED SEARCH CV - {model_name}")
        logger.info("="*70)
        logger.info(f"Parameter distributions: {param_distributions}")
        logger.info(f"Iterations: {n_iter}, CV folds: {cv}, Scoring: {scoring}")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        info = {
            'model_name': model_name,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'search_time': search_time,
            'n_iterations': n_iter
        }
        
        logger.info("="*70)
        logger.info(f"✅ RANDOM SEARCH COMPLETE")
        logger.info(f"   Best {scoring}: {random_search.best_score_:.4f}")
        logger.info(f"   Best params: {random_search.best_params_}")
        logger.info(f"   Time: {search_time:.2f}s")
        logger.info("="*70)
        
        self.training_history.append(info)
        return random_search.best_estimator_, info

    def train_multiple_models(
        self,
        models_dict: Dict[str, Any],
        X_train,
        y_train
    ) -> Dict[str, Tuple[Any, Dict]]:
        """
        Train multiple models
        
        Args:
            models_dict: Dictionary of model_name: model_instance
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of model_name: (trained_model, info)
        """
        logger.info("="*70)
        logger.info(f"TRAINING {len(models_dict)} MODELS")
        logger.info("="*70)
        
        results = {}
        for name, model in models_dict.items():
            trained_model, info = self.train(model, X_train, y_train, name)
            results[name] = (trained_model, info)
        
        logger.info("="*70)
        logger.info("✅ ALL MODELS TRAINED")
        logger.info("="*70)
        
        return results

    def save_model(self, model, filepath: str):
        """Save trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✅ Model saved to {filepath}")

    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training history"""
        if not self.training_history:
            logger.warning("No training history available")
            return pd.DataFrame()
        
        return pd.DataFrame(self.training_history)
