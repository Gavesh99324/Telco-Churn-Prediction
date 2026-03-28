"""Complete model training pipeline for Telco Customer Churn with MLflow tracking"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils.mlflow_utils import MLflowManager
from src.model_evaluation import ModelEvaluator
from src.model_training import ModelTrainer
from src.model_building import ModelBuilder
import sys
import pickle
import json
import hashlib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
from mlflow.tracking import MlflowClient

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
                 random_state: int = 42,
                 enable_mlflow: bool = True,
                 mlflow_tracking_uri: str = 'http://localhost:5000',
                 environment: str = 'development',
                 require_mlflow_in_production: bool = True):
        """
        Initialize TrainingPipeline

        Args:
            data_dir: Directory with preprocessed data
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
            enable_mlflow: Enable MLflow experiment tracking
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.random_state = random_state
        self.enable_mlflow = enable_mlflow
        self.environment = environment.lower()
        self.require_mlflow_in_production = require_mlflow_in_production
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.quality_thresholds = {
            'max_null_ratio': 0.01,
            'min_minority_class_ratio': 0.10,
            'max_drift_proxy': 0.60,
        }

        # Initialize components
        self.builder = ModelBuilder(random_state=random_state)
        self.trainer = ModelTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()

        # Initialize MLflow manager
        if self.enable_mlflow:
            try:
                self._validate_mlflow_connectivity(mlflow_tracking_uri)
                self.mlflow = MLflowManager(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name='telco-churn-prediction',
                    enable_autolog=True
                )
                logger.info("✅ MLflow tracking enabled")
            except Exception as e:
                if self.environment == 'production' and self.require_mlflow_in_production:
                    raise RuntimeError(
                        "MLflow is mandatory in production mode, but initialization failed"
                    ) from e
                logger.warning(f"⚠️ MLflow initialization failed: {e}")
                logger.warning("Continuing without MLflow tracking...")
                self.enable_mlflow = False
        else:
            if self.environment == 'production' and self.require_mlflow_in_production:
                raise RuntimeError(
                    "MLflow cannot be disabled in production mode"
                )
            logger.info("MLflow tracking disabled")

        # Create output directory
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def _validate_mlflow_connectivity(self, tracking_uri: str):
        """Validate MLflow endpoint reachability for production-safe runs."""
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)

    def _save_json_report(self, filename: str, payload: Dict[str, Any]):
        """Persist training reports to model artifacts."""
        report_path = Path(self.model_dir) / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("✅ Saved report: %s", report_path)
        return str(report_path)

    def _build_training_quality_report(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Build a quality report and drift proxy for train/test inputs."""
        total_cells_train = max(1, X_train.shape[0] * X_train.shape[1])
        total_cells_test = max(1, X_test.shape[0] * X_test.shape[1])
        null_ratio_train = float(
            X_train.isna().sum().sum()) / total_cells_train
        null_ratio_test = float(X_test.isna().sum().sum()) / total_cells_test

        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        minority_train = float(
            y_train_series.value_counts(normalize=True).min())
        minority_test = float(y_test_series.value_counts(normalize=True).min())

        # Drift proxy: mean standardized mean-difference across feature columns.
        mean_diff = (X_train.mean(numeric_only=True) -
                     X_test.mean(numeric_only=True)).abs()
        std_denom = X_train.std(numeric_only=True).replace(0, 1e-8)
        drift_proxy = float((mean_diff / std_denom).mean())

        return {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'train_shape': list(X_train.shape),
            'test_shape': list(X_test.shape),
            'null_ratio_train': null_ratio_train,
            'null_ratio_test': null_ratio_test,
            'class_distribution_train': {
                str(k): int(v)
                for k, v in y_train_series.value_counts().to_dict().items()
            },
            'class_distribution_test': {
                str(k): int(v)
                for k, v in y_test_series.value_counts().to_dict().items()
            },
            'minority_class_ratio_train': minority_train,
            'minority_class_ratio_test': minority_test,
            'drift_proxy_smd_mean': drift_proxy,
            'thresholds': self.quality_thresholds,
        }

    def _enforce_training_quality_gates(self, report: Dict[str, Any]):
        """Fail fast when training quality gates are violated."""
        failures = []

        if report['null_ratio_train'] > self.quality_thresholds['max_null_ratio']:
            failures.append(
                "training null ratio gate failed: "
                f"{report['null_ratio_train']:.4f} > "
                f"{self.quality_thresholds['max_null_ratio']:.4f}"
            )
        if report['null_ratio_test'] > self.quality_thresholds['max_null_ratio']:
            failures.append(
                "test null ratio gate failed: "
                f"{report['null_ratio_test']:.4f} > "
                f"{self.quality_thresholds['max_null_ratio']:.4f}"
            )
        if report['minority_class_ratio_train'] < self.quality_thresholds['min_minority_class_ratio']:
            failures.append(
                "train class imbalance gate failed: "
                f"{report['minority_class_ratio_train']:.4f} < "
                f"{self.quality_thresholds['min_minority_class_ratio']:.4f}"
            )
        if report['drift_proxy_smd_mean'] > self.quality_thresholds['max_drift_proxy']:
            failures.append(
                "train/test drift proxy gate failed: "
                f"{report['drift_proxy_smd_mean']:.4f} > "
                f"{self.quality_thresholds['max_drift_proxy']:.4f}"
            )

        if failures:
            failure_text = "\n - ".join(failures)
            raise ValueError(
                "Training quality gates failed:\n"
                f" - {failure_text}"
            )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute SHA256 digest for an artifact file."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _build_artifact_manifest(
        self,
        best_model_name: str,
        best_model: Any,
        quality_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create manifest used to enforce train/infer artifact compatibility."""
        feature_names_path = Path(self.data_dir) / 'feature_names.pkl'
        label_encoders_path = Path(self.data_dir) / 'label_encoders.pkl'
        scaler_path = Path(self.data_dir) / 'scaler.pkl'
        best_model_path = Path(self.model_dir) / 'best_model.pkl'

        with open(feature_names_path, 'rb') as f:
            feature_data = pickle.load(f)
            feature_names = feature_data.get('feature_names', [])

        return {
            'manifest_version': '1.0.0',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'model_name': best_model_name,
            'model_version': datetime.utcnow().strftime('%Y%m%d%H%M%S'),
            'training_tracking_uri': self.mlflow_tracking_uri,
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'model_has_feature_names_in': bool(
                hasattr(best_model, 'feature_names_in_')
            ),
            'artifact_hashes': {
                'best_model.pkl': self._sha256_file(best_model_path),
                'feature_names.pkl': self._sha256_file(feature_names_path),
                'label_encoders.pkl': self._sha256_file(label_encoders_path),
                'scaler.pkl': self._sha256_file(scaler_path),
            },
            'training_quality': quality_report,
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load preprocessed data from artifacts"""
        logger.info("Loading preprocessed data...")

        X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')
        X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        y_train = pd.read_csv(f'{self.data_dir}/y_train.csv')['Churn'].values
        y_test = pd.read_csv(f'{self.data_dir}/y_test.csv')['Churn'].values

        logger.info(
            f"✅ Loaded: X_train={X_train.shape}, X_test={X_test.shape}")
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
            # Start MLflow run for each baseline model
            if self.enable_mlflow:
                with self.mlflow.start_run(run_name=f"baseline_{name}", nested=True):
                    # Log model parameters
                    self.mlflow.log_params({
                        'model_type': name,
                        'phase': 'baseline',
                        'random_state': self.random_state
                    })

                    # Train
                    trained_model, train_info = self.trainer.train(
                        model, X_train, y_train, name)
                    trained_models[name] = trained_model

                    # Evaluate
                    results = self.evaluator.evaluate_model(
                        trained_model, X_train, y_train, X_test, y_test, name
                    )

                    # Log metrics to MLflow
                    self.mlflow.log_metrics({
                        'train_accuracy': results['train_accuracy'],
                        'train_precision': results['train_precision'],
                        'train_recall': results['train_recall'],
                        'train_f1': results['train_f1'],
                        'test_accuracy': results['test_accuracy'],
                        'test_precision': results['test_precision'],
                        'test_recall': results['test_recall'],
                        'test_f1': results['test_f1'],
                        'test_roc_auc': results['test_roc_auc']
                    })

                    # Log confusion matrix as dict
                    self.mlflow.log_dict(
                        {'confusion_matrix': results['confusion_matrix'].tolist(
                        )},
                        'confusion_matrix.json'
                    )
            else:
                # Train without MLflow
                trained_model, train_info = self.trainer.train(
                    model, X_train, y_train, name)
                trained_models[name] = trained_model
                self.evaluator.evaluate_model(
                    trained_model, X_train, y_train, X_test, y_test, name
                )

        # Compare all models
        comparison_df = self.evaluator.compare_models(
            save_path=f'{self.model_dir}/baseline_comparison.csv'
        )

        # Log comparison to MLflow
        if self.enable_mlflow:
            self.mlflow.log_dataframe(comparison_df, 'baseline_comparison.csv')

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
            base_model = GradientBoostingClassifier(
                random_state=self.random_state)
        else:
            logger.warning(
                f"No tuning grid for {best_model_name}, skipping tuning")
            return None

        # Start MLflow run for hyperparameter tuning
        if self.enable_mlflow:
            with self.mlflow.start_run(run_name=f"tuning_{best_model_name}", nested=True):
                # Log tuning parameters
                self.mlflow.log_params({
                    'model_type': best_model_name,
                    'phase': 'hyperparameter_tuning',
                    'cv_folds': 5,
                    'scoring': 'f1'
                })
                self.mlflow.log_dict(param_grid, 'param_grid.json')

                # Perform grid search
                tuned_model, tuning_info = self.trainer.hyperparameter_tuning_grid(
                    base_model, param_grid, X_train, y_train,
                    model_name=f"{best_model_name}_Tuned", cv=5, scoring='f1'
                )

                # Log best parameters
                self.mlflow.log_params(
                    {f'best_{k}': v for k, v in tuning_info['best_params'].items()})
                self.mlflow.log_metrics(
                    {'best_cv_score': tuning_info['best_score']})

                # Evaluate tuned model
                tuned_results = self.evaluator.evaluate_model(
                    tuned_model, X_train, y_train, X_test, y_test,
                    f"{best_model_name}_Tuned"
                )

                # Log tuned model metrics
                self.mlflow.log_metrics({
                    'tuned_test_accuracy': tuned_results['test_accuracy'],
                    'tuned_test_f1': tuned_results['test_f1'],
                    'tuned_test_roc_auc': tuned_results['test_roc_auc']
                })
        else:
            # Perform tuning without MLflow
            tuned_model, tuning_info = self.trainer.hyperparameter_tuning_grid(
                base_model, param_grid, X_train, y_train,
                model_name=f"{best_model_name}_Tuned", cv=5, scoring='f1'
            )
            tuned_results = self.evaluator.evaluate_model(
                tuned_model, X_train, y_train, X_test, y_test,
                f"{best_model_name}_Tuned"
            )

        # Save tuned model
        self.trainer.save_model(
            tuned_model, f'{self.model_dir}/best_tuned_model.pkl')

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

        # Start MLflow run for SMOTE training
        if self.enable_mlflow:
            with self.mlflow.start_run(run_name="smote_training", nested=True):
                # Log SMOTE parameters
                self.mlflow.log_params({
                    'phase': 'smote',
                    'k_neighbors': 5,
                    'sampling_strategy': 'auto'
                })

                # Train with SMOTE
                smote_trained, smote_info = self.trainer.train_with_smote(
                    smote_model, X_train, y_train,
                    model_name="Best_Model_SMOTE", k_neighbors=5
                )

                # Log SMOTE distribution info
                self.mlflow.log_dict(smote_info, 'smote_info.json')

                # Evaluate
                smote_results = self.evaluator.evaluate_model(
                    smote_trained, X_train, y_train, X_test, y_test,
                    "Best_Model_SMOTE"
                )

                # Log SMOTE model metrics
                self.mlflow.log_metrics({
                    'smote_train_f1': smote_results['train_f1'],
                    'smote_test_f1': smote_results['test_f1'],
                    'smote_test_recall': smote_results['test_recall'],
                    'smote_test_roc_auc': smote_results['test_roc_auc']
                })
        else:
            # Train without MLflow
            smote_trained, smote_info = self.trainer.train_with_smote(
                smote_model, X_train, y_train,
                model_name="Best_Model_SMOTE", k_neighbors=5
            )
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
        if self.enable_mlflow:
            with self.mlflow.start_run(run_name="voting_classifier", nested=True):
                self.mlflow.log_params({
                    'model_type': 'Voting_Classifier',
                    'phase': 'ensemble',
                    'voting_type': 'soft'
                })

                voting_clf = self.builder.build_voting_classifier(
                    use_smote_params=False)
                voting_model, _ = self.trainer.train(
                    voting_clf, X_train, y_train, "Voting_Classifier"
                )
                voting_results = self.evaluator.evaluate_model(
                    voting_model, X_train, y_train, X_test, y_test, "Voting_Classifier"
                )

                self.mlflow.log_metrics({
                    'voting_test_f1': voting_results['test_f1'],
                    'voting_test_roc_auc': voting_results['test_roc_auc']
                })
        else:
            voting_clf = self.builder.build_voting_classifier(
                use_smote_params=False)
            voting_model, _ = self.trainer.train(
                voting_clf, X_train, y_train, "Voting_Classifier"
            )
            voting_results = self.evaluator.evaluate_model(
                voting_model, X_train, y_train, X_test, y_test, "Voting_Classifier"
            )

        # Stacking Classifier
        if self.enable_mlflow:
            with self.mlflow.start_run(run_name="stacking_classifier", nested=True):
                self.mlflow.log_params({
                    'model_type': 'Stacking_Classifier',
                    'phase': 'ensemble',
                    'cv_folds': 5
                })

                stacking_clf = self.builder.build_stacking_classifier(
                    use_smote_params=False)
                stacking_model, _ = self.trainer.train(
                    stacking_clf, X_train, y_train, "Stacking_Classifier"
                )
                stacking_results = self.evaluator.evaluate_model(
                    stacking_model, X_train, y_train, X_test, y_test, "Stacking_Classifier"
                )

                self.mlflow.log_metrics({
                    'stacking_test_f1': stacking_results['test_f1'],
                    'stacking_test_roc_auc': stacking_results['test_roc_auc']
                })
        else:
            stacking_clf = self.builder.build_stacking_classifier(
                use_smote_params=False)
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

        # Log final comparison to MLflow
        if self.enable_mlflow:
            self.mlflow.log_dataframe(final_comparison, 'final_comparison.csv')

        # Get absolute best
        best_model_name = final_comparison.iloc[0]['Model']
        best_f1 = final_comparison.iloc[0]['Test F1']

        logger.info(f"\n🏆 FINAL BEST MODEL: {best_model_name}")
        logger.info(f"   Test F1 Score: {best_f1:.4f}")

        return best_model_name, best_f1, final_comparison

    def run(self):
        """Execute complete training pipeline with MLflow tracking"""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)

        # Start parent MLflow run for entire pipeline
        if self.enable_mlflow:
            with self.mlflow.start_run(run_name="complete_training_pipeline"):
                # Log pipeline configuration
                self.mlflow.log_params({
                    'pipeline': 'complete_training',
                    'random_state': self.random_state,
                    'data_dir': self.data_dir,
                    'model_dir': self.model_dir
                })
                self.mlflow.set_tags({
                    'project': 'telco-churn-prediction',
                    'pipeline_type': 'training',
                    'version': '1.0'
                })

                # Execute pipeline phases
                results = self._execute_pipeline()

                # Log final results
                self.mlflow.log_metrics({
                    'final_best_f1': results['best_f1_score'],
                    'num_models_trained': 8  # 5 baseline + 2 ensemble + 1 SMOTE
                })

                # Log pipeline summary
                summary = {
                    'best_model': results['best_model_name'],
                    'best_f1_score': float(results['best_f1_score']),
                    'phases_completed': ['baseline', 'tuning', 'smote', 'ensemble']
                }
                self.mlflow.log_dict(summary, 'pipeline_summary.json')

                # Register and promote best model to Production
                logger.info("\n" + "="*70)
                logger.info("🎯 REGISTERING BEST MODEL TO PRODUCTION")
                logger.info("="*70)

                try:
                    # Find the best run by test_f1 metric
                    best_run = self.mlflow.get_best_run(
                        metric_name="test_f1", ascending=False)

                    if best_run:
                        run_id = best_run.info.run_id
                        model_uri = f"runs:/{run_id}/model"
                        model_name = "telco-churn-best-model"

                        logger.info(f"📦 Best run ID: {run_id}")
                        logger.info(
                            f"🏆 Best model: {results['best_model_name']}")
                        logger.info(
                            f"📈 Test F1: {results['best_f1_score']:.4f}")

                        # Register the model
                        version = self.mlflow.register_model(
                            model_uri=model_uri,
                            name=model_name,
                            tags={
                                'framework': 'scikit-learn',
                                'problem_type': 'binary_classification',
                                'model_type': results['best_model_name'],
                                'f1_score': str(results['best_f1_score'])
                            },
                            description=f"Best model from training pipeline: {results['best_model_name']} with F1={results['best_f1_score']:.4f}"
                        )

                        if version:
                            # Transition to Production stage
                            self.mlflow.transition_model_stage(
                                name=model_name,
                                version=str(version),
                                stage="Production",
                                archive_existing=True
                            )
                            logger.info(
                                f"✅ Model promoted to Production (version {version})")
                        else:
                            logger.warning("⚠️  Model registration failed")
                    else:
                        logger.warning(
                            "⚠️  No best run found for model registration")
                except Exception as e:
                    logger.error(f"❌ Failed to register/promote model: {e}")

                logger.info("✅ MLflow tracking completed")
                return results
        else:
            # Run without MLflow
            return self._execute_pipeline()

    def _execute_pipeline(self):
        """Internal method to execute pipeline phases"""
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()

        # Quality gates before model training starts.
        quality_report = self._build_training_quality_report(
            X_train, X_test, y_train, y_test
        )
        self._save_json_report('training_quality_report.json', quality_report)
        self._enforce_training_quality_gates(quality_report)
        if self.enable_mlflow:
            self.mlflow.log_dict(
                quality_report, 'training_quality_report.json')

        # Phase 1: Baseline models
        baseline_results = self.train_baseline_models(
            X_train, y_train, X_test, y_test)

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
        ensemble_results = self.train_ensemble_models(
            X_train, y_train, X_test, y_test)

        # Final comparison and best model selection
        best_name, best_f1, final_comparison = self.save_best_model()

        # Build map of all trained candidates and persist final selected model.
        trained_models = dict(baseline_results['models'])
        if tuned_results and 'model' in tuned_results:
            trained_models[f"{baseline_results['best_model_name']}_Tuned"] = tuned_results['model']
        if smote_results and 'model' in smote_results:
            trained_models['Best_Model_SMOTE'] = smote_results['model']
        if ensemble_results:
            trained_models['Voting_Classifier'] = ensemble_results['voting']['model']
            trained_models['Stacking_Classifier'] = ensemble_results['stacking']['model']

        selected_model = trained_models.get(
            best_name, baseline_results['best_model'])
        self.trainer.save_model(
            selected_model, f'{self.model_dir}/best_model.pkl')

        manifest = self._build_artifact_manifest(
            best_model_name=best_name,
            best_model=selected_model,
            quality_report=quality_report
        )
        self._save_json_report('model_manifest.json', manifest)
        if self.enable_mlflow:
            self.mlflow.log_dict(manifest, 'model_manifest.json')

        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"\n📊 Results saved to: {self.model_dir}")
        logger.info(f"🏆 Best model: {best_name}")
        logger.info(f"📈 Best F1 Score: {best_f1:.4f}")

        return {
            'best_model_name': best_name,
            'best_f1_score': best_f1,
            'final_comparison': final_comparison,
            'training_quality_report': quality_report,
            'artifact_manifest': manifest,
            'baseline_results': baseline_results,
            'tuned_results': tuned_results,
            'smote_results': smote_results,
            'ensemble_results': ensemble_results
        }


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    results = pipeline.run()
