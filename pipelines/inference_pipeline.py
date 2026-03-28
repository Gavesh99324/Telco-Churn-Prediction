"""Batch inference pipeline for Telco Customer Churn"""
import sys
import os
import pickle
import json
import hashlib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_inference import ModelInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Batch inference pipeline"""

    def __init__(self,
                 data_dir: str = 'artifacts/data',
                 model_dir: str = 'artifacts/models',
                 output_dir: str = 'artifacts/predictions'):
        """
        Initialize InferencePipeline
        
        Args:
            data_dir: Directory with preprocessed data
            model_dir: Directory with trained models
            output_dir: Directory to save predictions
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Initialize inference engine
        self.inference = ModelInference()
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute SHA256 digest for artifact compatibility checks."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_manifest_compatibility(self):
        """Validate model and preprocessing artifacts against training manifest."""
        manifest_path = Path(self.model_dir) / 'model_manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing artifact manifest: {manifest_path}. "
                "Re-run training pipeline to generate model_manifest.json"
            )

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        required_files = {
            'best_model.pkl': Path(self.model_dir) / 'best_model.pkl',
            'feature_names.pkl': Path(self.data_dir) / 'feature_names.pkl',
            'label_encoders.pkl': Path(self.data_dir) / 'label_encoders.pkl',
            'scaler.pkl': Path(self.data_dir) / 'scaler.pkl',
        }

        failures = []
        expected_hashes = manifest.get('artifact_hashes', {})
        for artifact_name, artifact_path in required_files.items():
            if not artifact_path.exists():
                failures.append(f"missing artifact file: {artifact_path}")
                continue
            expected_hash = expected_hashes.get(artifact_name)
            if expected_hash is None:
                failures.append(f"manifest missing hash for {artifact_name}")
                continue
            actual_hash = self._sha256_file(artifact_path)
            if actual_hash != expected_hash:
                failures.append(
                    f"hash mismatch for {artifact_name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )

        manifest_features = manifest.get('feature_names', [])
        if manifest_features and self.feature_names != manifest_features:
            failures.append("feature_names mismatch between manifest and current artifacts")

        model = self.inference.model
        if model is not None and hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            if manifest_features and model_features != manifest_features:
                failures.append(
                    "model.feature_names_in_ mismatch with manifest feature_names"
                )

        if failures:
            failure_text = "\n - ".join(failures)
            raise ValueError(
                "Artifact compatibility validation failed:\n"
                f" - {failure_text}"
            )

        logger.info("✅ Artifact manifest compatibility validation passed")

    def load_preprocessors(self):
        """Load all preprocessing artifacts"""
        logger.info("Loading preprocessors...")
        
        with open(f'{self.data_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info("   ✅ Loaded scaler")
        
        with open(f'{self.data_dir}/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        logger.info("   ✅ Loaded label encoders")
        
        with open(f'{self.data_dir}/feature_names.pkl', 'rb') as f:
            feature_data = pickle.load(f)
            self.feature_names = feature_data['feature_names']
        logger.info(f"   ✅ Loaded {len(self.feature_names)} feature names")

    def load_model(self, model_name: str = 'best_model.pkl'):
        """Load trained model"""
        model_path = f'{self.model_dir}/{model_name}'
        logger.info(f"Loading model from {model_path}...")
        
        self.inference.load_model(model_path)

    def load_test_data(self):
        """Load test data for inference"""
        logger.info("Loading test data...")
        
        X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        y_test = pd.read_csv(f'{self.data_dir}/y_test.csv')['Churn'].values
        
        logger.info(f"✅ Loaded test data: {X_test.shape}")
        return X_test, y_test

    def make_predictions(self, X_test):
        """Make predictions on test data"""
        logger.info("Making predictions...")
        
        # Get predictions and probabilities
        predictions = self.inference.predict(X_test)
        probabilities = self.inference.predict_proba(X_test)
        
        logger.info(f"✅ Generated {len(predictions)} predictions")
        
        return predictions, probabilities

    def save_predictions(self, X_test, y_test, predictions, probabilities):
        """Save predictions to CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'actual_churn': y_test,
            'predicted_churn': predictions,
            'churn_probability': probabilities[:, 1],
            'no_churn_probability': probabilities[:, 0]
        })
        
        # Add prediction confidence
        results_df['correct_prediction'] = (
            results_df['actual_churn'] == results_df['predicted_churn']
        )
        
        # Save
        output_path = f'{self.output_dir}/predictions_{timestamp}.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"✅ Predictions saved to {output_path}")
        
        # Summary statistics
        accuracy = results_df['correct_prediction'].mean()
        churn_rate = results_df['predicted_churn'].mean()
        high_risk = (results_df['churn_probability'] > 0.7).sum()
        
        logger.info(f"\n📊 PREDICTION SUMMARY:")
        logger.info(f"   Total predictions: {len(results_df)}")
        logger.info(f"   Accuracy: {accuracy:.2%}")
        logger.info(f"   Predicted churn rate: {churn_rate:.2%}")
        logger.info(f"   High risk customers (prob > 0.7): {high_risk}")
        
        return results_df

    def run(self, model_name: str = 'best_model.pkl'):
        """Execute complete inference pipeline"""
        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING INFERENCE PIPELINE")
        logger.info("="*70)
        
        # Load artifacts
        self.load_preprocessors()
        self.load_model(model_name)
        self.validate_manifest_compatibility()
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Make predictions
        predictions, probabilities = self.make_predictions(X_test)
        
        # Save results
        results_df = self.save_predictions(X_test, y_test, predictions, probabilities)
        
        logger.info("\n" + "="*70)
        logger.info("✅ INFERENCE PIPELINE COMPLETE!")
        logger.info("="*70)
        
        return results_df


if __name__ == '__main__':
    pipeline = InferencePipeline()
    results = pipeline.run()
