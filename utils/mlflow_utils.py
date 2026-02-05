"""MLflow utilities"""
import mlflow


class MLflowManager:
    """Manage MLflow experiments"""

    def __init__(self, tracking_uri='http://localhost:5000'):
        mlflow.set_tracking_uri(tracking_uri)

    def log_params(self, params):
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        """Log metrics"""
        mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path):
        """Log model"""
        mlflow.sklearn.log_model(model, artifact_path)
