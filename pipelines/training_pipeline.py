"""Complete model training pipeline for Telco Customer Churn"""
import sys
import os
import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_building import ModelBuilder
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete model training pipeline"""

    def __init__(self,
                 data_dir: str = 'artifacts/data',
                 model_dir: str = 'artifacts/models',
                 random_state: int = 42):
        """
        Initialize TrainingPipeline
        
        Args:
            data_dir: Directory with preprocessed data
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.random_state = random_state
        
        # Initialize components
        self.builder = ModelBuilder(random_state=random_state)
        self.trainer = ModelTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()
        
        # Create output directory
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load preprocessed data from artifacts"""
        logger.info("Loading preprocessed data...")
        
        X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')
        X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        y_train = pd.read_csv(f'{self.data_dir}/y_train.csv')['Churn'].values
        y_test = pd.read_csv(f'{self.data_dir}/y_test.csv')['Churn'].values
        
        logger.info(f"✅ Loaded: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_baseline_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train and evaluate all baseline models"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: BASELINE MODEL TRAINING")
        logger.info("="*70)
        
        # Build all models
        models = self.builder.build_all_models()
        
        # Train and evaluate each
        trained_models = {}
        for name, model in models.items():
            # Train
            trained_model, train_info = self.trainer.train(model, X_train, y_train, name)
            trained_models[name] = trained_model
            
            # Evaluate
            self.evaluator.evaluate_model(
                trained_model, X_train, y_train, X_test, y_test, name
            )
        
        # Compare all models
        comparison_df = self.evaluator.compare_models(
            save_path=f'{self.model_dir}/baseline_comparison.csv'
        )
        
        # Get best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        
        logger.info(f"\n🏆 BEST BASELINE MODEL: {best_model_name}")
        
        return {
            'models': trained_models,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'comparison': comparison_df
        }

    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test,
                             best_model_name: str):
        """Perform hyperparameter tuning on best model"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: HYPERPARAMETER TUNING")
        logger.info("="*70)
        
        # Define param grids based on model type
        if 'Random Forest' in best_model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
            base_model = RandomForestClassifier(random_state=self.random_state)
            
        elif 'Gradient Boosting' in best_model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 1.0]
            }
            base_model = GradientBoostingClassifier(random_state=self.random_state)
        else:
            logger.warning(f"No tuning grid for {best_model_name}, skipping tuning")
            return None
        
        # Perform grid search
        tuned_model, tuning_info = self.trainer.hyperparameter_tuning_grid(
            base_model, param_grid, X_train, y_train,
            model_name=f"{best_model_name}_Tuned", cv=5, scoring='f1'
        )
        
        # Evaluate tuned model
        tuned_results = self.evaluator.evaluate_model(
            tuned_model, X_train, y_train, X_test, y_test,
            f"{best_model_name}_Tuned"
        )
        
        # Save tuned model
        self.trainer.save_model(tuned_model, f'{self.model_dir}/best_tuned_model.pkl')
        
        return {
            'model': tuned_model,
            'results': tuned_results,
            'info': tuning_info
        }

    def train_with_smote(self, X_train, y_train, X_test, y_test, best_model):
        """Train best model with SMOTE"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: TRAINING WITH SMOTE")
        logger.info("="*70)
        
        # Clone model to avoid modifying original
        import copy
        smote_model = copy.deepcopy(best_model)
        
        # Train with SMOTE
        smote_trained, smote_info = self.trainer.train_with_smote(
            smote_model, X_train, y_train,
            model_name="Best_Model_SMOTE", k_neighbors=5
        )
        
        # Evaluate
        smote_results = self.evaluator.evaluate_model(
            smote_trained, X_train, y_train, X_test, y_test,
            "Best_Model_SMOTE"
        )
        
        return {
            'model': smote_trained,
            'results': smote_results
        }

    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble models"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: ENSEMBLE MODELS")
        logger.info("="*70)
        
        # Voting Classifier
        voting_clf = self.builder.build_voting_classifier(use_smote_params=False)
        voting_model, _ = self.trainer.train(
            voting_clf, X_train, y_train, "Voting_Classifier"
        )
        voting_results = self.evaluator.evaluate_model(
            voting_model, X_train, y_train, X_test, y_test, "Voting_Classifier"
        )
        
        # Stacking Classifier
        stacking_clf = self.builder.build_stacking_classifier(use_smote_params=False)
        stacking_model, _ = self.trainer.train(
            stacking_clf, X_train, y_train, "Stacking_Classifier"
        )
        stacking_results = self.evaluator.evaluate_model(
            stacking_model, X_train, y_train, X_test, y_test, "Stacking_Classifier"
        )
        
        return {
            'voting': {'model': voting_model, 'results': voting_results},
            'stacking': {'model': stacking_model, 'results': stacking_results}
        }

    def save_best_model(self):
        """Select and save the overall best model"""
        logger.info("\n" + "="*70)
        logger.info("SELECTING BEST MODEL")
        logger.info("="*70)
        
        # Compare all models
        final_comparison = self.evaluator.compare_models(
            save_path=f'{self.model_dir}/final_comparison.csv'
        )
        
        # Get absolute best
        best_model_name = final_comparison.iloc[0]['Model']
        best_f1 = final_comparison.iloc[0]['Test F1']
        
        logger.info(f"\n🏆 FINAL BEST MODEL: {best_model_name}")
        logger.info(f"   Test F1 Score: {best_f1:.4f}")
        
        # Note: The best model is already in evaluation_results
        # We need to get it from the trained models
        return best_model_name, best_f1

    def run(self):
        """Execute complete training pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Phase 1: Baseline models
        baseline_results = self.train_baseline_models(X_train, y_train, X_test, y_test)
        
        # Save best baseline model
        self.trainer.save_model(
            baseline_results['best_model'],
            f'{self.model_dir}/best_model.pkl'
        )
        
        # Phase 2: Hyperparameter tuning
        tuned_results = self.hyperparameter_tuning(
            X_train, y_train, X_test, y_test,
            baseline_results['best_model_name']
        )
        
        # Phase 3: SMOTE training
        smote_results = self.train_with_smote(
            X_train, y_train, X_test, y_test,
            baseline_results['best_model']
        )
        
        # Phase 4: Ensemble models
        ensemble_results = self.train_ensemble_models(X_train, y_train, X_test, y_test)
        
        # Final comparison and best model selection
        best_name, best_f1 = self.save_best_model()
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"\n📊 Results saved to: {self.model_dir}")
        logger.info(f"🏆 Best model: {best_name}")
        logger.info(f"📈 Best F1 Score: {best_f1:.4f}")
        
        return {
            'best_model_name': best_name,
            'best_f1_score': best_f1,
            'baseline_results': baseline_results,
            'tuned_results': tuned_results,
            'smote_results': smote_results,
            'ensemble_results': ensemble_results
        }


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    results = pipeline.run()
