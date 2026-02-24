"""Model evaluation metrics for Telco Customer Churn"""
import logging
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate ML model performance"""

    def __init__(self):
        """Initialize ModelEvaluator"""
        self.evaluation_results = []

    def evaluate_model(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_name: Name for logging
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Training predictions
        start_time = time.time()
        y_train_pred = model.predict(X_train)
        train_time = time.time() - start_time
        
        # Test predictions
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        test_time = time.time() - start_time
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        results = {
            'model_name': model_name,
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'test_precision': test_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'test_recall': test_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'test_f1': test_metrics['f1'],
            'test_roc_auc': test_metrics.get('roc_auc', None),
            'confusion_matrix': cm,
            'predictions': y_test_pred,
            'probabilities': y_test_proba,
            'train_predict_time': train_time,
            'test_predict_time': test_time
        }
        
        self.evaluation_results.append(results)
        
        # Log results
        self._log_results(results)
        
        return results

    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics

    def _log_results(self, results: Dict[str, Any]):
        """Log evaluation results"""
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION RESULTS - {results['model_name']}")
        logger.info(f"{'='*70}")
        logger.info(f"Training Metrics:")
        logger.info(f"  Accuracy:  {results['train_accuracy']:.4f}")
        logger.info(f"  Precision: {results['train_precision']:.4f}")
        logger.info(f"  Recall:    {results['train_recall']:.4f}")
        logger.info(f"  F1-Score:  {results['train_f1']:.4f}")
        logger.info(f"\nTest Metrics:")
        logger.info(f"  Accuracy:  {results['test_accuracy']:.4f}")
        logger.info(f"  Precision: {results['test_precision']:.4f}")
        logger.info(f"  Recall:    {results['test_recall']:.4f}")
        logger.info(f"  F1-Score:  {results['test_f1']:.4f}")
        if results['test_roc_auc'] is not None:
            logger.info(f"  ROC-AUC:   {results['test_roc_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"\n{results['confusion_matrix']}")
        logger.info(f"{'='*70}\n")

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for title
            save_path: Path to save plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Confusion matrix saved to {save_path}")
        
        plt.show()

    def plot_roc_curves(
        self,
        models_dict: Dict[str, Tuple[Any, Any, Any]],
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_dict: Dictionary of model_name: (model, X_test, y_test)
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        for name, (model, X_test, y_test) in models_dict.items():
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                logger.warning(f"{name} doesn't support probability predictions")
                continue
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            
            # Plot
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ ROC curves saved to {save_path}")
        
        plt.show()

    def get_classification_report(
        self,
        y_true,
        y_pred,
        model_name: str = "Model"
    ) -> str:
        """
        Generate classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for header
            
        Returns:
            Classification report string
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=['No Churn', 'Churn']
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CLASSIFICATION REPORT - {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"\n{report}")
        logger.info(f"{'='*70}\n")
        
        return report

    def compare_models(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Args:
            save_path: Path to save comparison DataFrame (optional)
            
        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        # Extract key metrics
        comparison_data = []
        for result in self.evaluation_results:
            comparison_data.append({
                'Model': result['model_name'],
                'Train Accuracy': result['train_accuracy'],
                'Test Accuracy': result['test_accuracy'],
                'Test Precision': result['test_precision'],
                'Test Recall': result['test_recall'],
                'Test F1': result['test_f1'],
                'Test ROC-AUC': result.get('test_roc_auc', np.nan)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by Test F1
        df = df.sort_values('Test F1', ascending=False).reset_index(drop=True)
        
        logger.info("\n" + "="*100)
        logger.info("MODEL COMPARISON")
        logger.info("="*100)
        logger.info(f"\n{df.to_string(index=False)}\n")
        logger.info("="*100 + "\n")
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"✅ Comparison saved to {save_path}")
        
        return df

    def plot_metrics_comparison(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing models
        
        Args:
            metrics: List of metrics to plot (default: all test metrics)
            save_path: Path to save plot (optional)
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return
        
        if metrics is None:
            metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        # Prepare data
        model_names = [r['model_name'] for r in self.evaluation_results]
        data = {metric: [r[metric] for r in self.evaluation_results] for metric in metrics}
        
        # Plot
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics)/2 + 0.5)
            ax.bar(x + offset, data[metric], width, label=metric.replace('test_', '').replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Metrics comparison saved to {save_path}")
        
        plt.show()
