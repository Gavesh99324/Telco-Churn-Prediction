"""Inference pipeline for Telco Customer Churn"""
import logging
import pickle
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    """Perform inference using trained models"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ModelInference
        
        Args:
            model_path: Path to saved model pickle file (optional)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load trained model from file
        
        Args:
            model_path: Path to model pickle file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_path = model_path
            logger.info(f"✅ Model loaded from {model_path}")
        except FileNotFoundError:
            logger.error(f"❌ Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise

    def set_model(self, model):
        """
        Set model directly (for already trained models)
        
        Args:
            model: Trained sklearn model
        """
        self.model = model
        logger.info("✅ Model set directly")

    def predict(self, X) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction (array-like or DataFrame)
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or set_model() first.")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ Predictions made for {len(X)} samples in {inference_time:.4f}s")
        
        return predictions

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features for prediction (array-like or DataFrame)
            
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or set_model() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions")
        
        start_time = time.time()
        probabilities = self.model.predict_proba(X)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ Probabilities predicted for {len(X)} samples in {inference_time:.4f}s")
        
        return probabilities

    def predict_with_confidence(self, X, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores
        
        Args:
            X: Features for prediction
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probabilities = self.predict_proba(X)
        churn_probabilities = probabilities[:, 1]  # Probability of churn
        
        predictions = (churn_probabilities >= threshold).astype(int)
        confidence_scores = np.where(
            predictions == 1,
            churn_probabilities,
            1 - churn_probabilities
        )
        
        logger.info(f"✅ Predictions with confidence (threshold={threshold})")
        
        return predictions, confidence_scores

    def batch_predict(
        self,
        X,
        batch_size: int = 1000,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Perform batch predictions for large datasets
        
        Args:
            X: Features for prediction
            batch_size: Number of samples per batch
            return_proba: If True, return probabilities instead of classes
            
        Returns:
            Array of predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or set_model() first.")
        
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logger.info(f"Batch prediction: {n_samples} samples in {n_batches} batches")
        
        all_predictions = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            
            if return_proba:
                batch_pred = self.predict_proba(X_batch)
            else:
                batch_pred = self.predict(X_batch)
            
            all_predictions.append(batch_pred)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{n_batches} batches")
        
        result = np.vstack(all_predictions) if return_proba else np.concatenate(all_predictions)
        logger.info(f"✅ Batch prediction complete")
        
        return result

    def predict_with_metadata(
        self,
        X: pd.DataFrame,
        customer_ids: Optional[List] = None,
        include_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions and return DataFrame with metadata
        
        Args:
            X: Features DataFrame
            customer_ids: List of customer IDs (optional)
            include_proba: Include probability scores
            
        Returns:
            DataFrame with predictions and metadata
        """
        predictions = self.predict(X)
        
        result_df = pd.DataFrame({
            'prediction': predictions,
            'churn': predictions.astype(bool)
        })
        
        if customer_ids:
            result_df.insert(0, 'customer_id', customer_ids)
        
        if include_proba:
            probabilities = self.predict_proba(X)
            result_df['churn_probability'] = probabilities[:, 1]
            result_df['no_churn_probability'] = probabilities[:, 0]
        
        logger.info(f"✅ Created prediction DataFrame with {len(result_df)} rows")
        
        return result_df

    def validate_input(self, X, expected_features: Optional[List[str]] = None) -> bool:
        """
        Validate input features
        
        Args:
            X: Input features
            expected_features: List of expected feature names (optional)
            
        Returns:
            True if validation passes
        """
        # Check if model is loaded
        if self.model is None:
            logger.error("❌ Model not loaded")
            return False
        
        # Check shape
        if hasattr(self.model, 'n_features_in_'):
            expected_n_features = self.model.n_features_in_
            actual_n_features = X.shape[1] if len(X.shape) > 1 else 1
            
            if actual_n_features != expected_n_features:
                logger.error(
                    f"❌ Feature mismatch: expected {expected_n_features}, "
                    f"got {actual_n_features}"
                )
                return False
        
        # Check feature names if provided
        if expected_features and isinstance(X, pd.DataFrame):
            missing_features = set(expected_features) - set(X.columns)
            if missing_features:
                logger.error(f"❌ Missing features: {missing_features}")
                return False
        
        logger.info("✅ Input validation passed")
        return True

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance from model
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Feature Importances:")
        logger.info(importance_df.head(top_n).to_string(index=False))
        
        return importance_df.head(top_n)

    def explain_prediction(
        self,
        X,
        index: int = 0,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            X: Features
            index: Index of sample to explain
            feature_names: List of feature names
            
        Returns:
            Dictionary with prediction explanation
        """
        # Get prediction and probability
        sample = X[index:index+1]
        prediction = self.predict(sample)[0]
        
        explanation = {
            'prediction': int(prediction),
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn'
        }
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.predict_proba(sample)[0]
            explanation['no_churn_probability'] = float(proba[0])
            explanation['churn_probability'] = float(proba[1])
        
        # Add feature values if available
        if isinstance(X, pd.DataFrame) and feature_names:
            explanation['feature_values'] = X.iloc[index][feature_names].to_dict()
        
        logger.info(f"\nPrediction Explanation (Sample {index}):")
        for key, value in explanation.items():
            if key != 'feature_values':
                logger.info(f"  {key}: {value}")
        
        return explanation
