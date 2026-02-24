"""
MLflow utilities for experiment tracking, model registry, and artifact management
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Comprehensive MLflow manager for experiment tracking and model registry
    
    Features:
    - Experiment management (create, set, list)
    - Run context managers for automatic tracking
    - Parameter and metric logging
    - Model registry operations
    - Artifact logging (plots, data, files)
    - Model comparison and search
    - Auto-logging for sklearn models
    """

    def __init__(
        self,
        tracking_uri: str = 'http://localhost:5000',
        experiment_name: str = 'telco-churn-prediction',
        enable_autolog: bool = True
    ):
        """
        Initialize MLflowManager
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            enable_autolog: Enable auto-logging for sklearn
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        logger.info(f"Using experiment: {experiment_name} (ID: {self.experiment_id})")
        
        # Enable auto-logging
        if enable_autolog:
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=False
            )
            logger.info("✅ Sklearn auto-logging enabled")

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get existing experiment or create new one"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                logger.info(f"Found existing experiment: {experiment_name}")
                return experiment.experiment_id
        except Exception:
            pass
        
        # Create new experiment
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "telco-churn", "version": "1.0"}
        )
        logger.info(f"Created new experiment: {experiment_name}")
        return experiment_id

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Context manager for MLflow runs
        
        Usage:
            with mlflow_manager.start_run(run_name="baseline_training"):
                mlflow_manager.log_params({...})
                mlflow_manager.log_metrics({...})
        
        Args:
            run_name: Name of the run
            nested: Whether this is a nested run
            tags: Additional tags for the run
            description: Run description
        """
        with mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=tags,
            description=description
        ) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            if run_name:
                logger.info(f"Run name: {run_name}")
            
            try:
                yield run
            except Exception as e:
                logger.error(f"Error in MLflow run: {e}")
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise
            else:
                mlflow.set_tag("status", "completed")
            finally:
                logger.info(f"Ended MLflow run: {run.info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to current run
        
        Args:
            params: Dictionary of parameters
        """
        try:
            # Filter out None values and convert to strings
            filtered_params = {
                k: str(v) for k, v in params.items()
                if v is not None
            }
            mlflow.log_params(filtered_params)
            logger.debug(f"Logged {len(filtered_params)} parameters")
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics to current run
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/epoch number
        """
        try:
            # Filter out None/NaN values
            filtered_metrics = {
                k: float(v) for k, v in metrics.items()
                if v is not None and not pd.isna(v)
            }
            mlflow.log_metrics(filtered_metrics, step=step)
            logger.debug(f"Logged {len(filtered_metrics)} metrics")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log model to current run
        
        Args:
            model: Trained model
            artifact_path: Artifact path in MLflow
            registered_model_name: Name for model registry
            signature: Model signature
            input_example: Example input
        """
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
            logger.info(f"✅ Model logged to '{artifact_path}'")
            if registered_model_name:
                logger.info(f"✅ Model registered as '{registered_model_name}'")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact file to current run
        
        Args:
            local_path: Local file path
            artifact_path: Artifact path in MLflow
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log directory of artifacts to current run
        
        Args:
            local_dir: Local directory path
            artifact_path: Artifact path in MLflow
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"✅ Logged artifacts from: {local_dir}")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def log_figure(
        self,
        figure: plt.Figure,
        artifact_file: str,
        close_figure: bool = True
    ):
        """
        Log matplotlib figure to current run
        
        Args:
            figure: Matplotlib figure
            artifact_file: Filename for the artifact
            close_figure: Whether to close figure after logging
        """
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure: {artifact_file}")
            if close_figure:
                plt.close(figure)
        except Exception as e:
            logger.warning(f"Failed to log figure: {e}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log dictionary as JSON artifact
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact
        """
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dict: {artifact_file}")
        except Exception as e:
            logger.warning(f"Failed to log dict: {e}")

    def log_dataframe(self, df: pd.DataFrame, artifact_file: str):
        """
        Log pandas DataFrame as CSV artifact
        
        Args:
            df: DataFrame to log
            artifact_file: Filename for the artifact
        """
        try:
            # Save locally first
            temp_path = Path("temp_artifacts") / artifact_file
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(temp_path, index=False)
            
            # Log to MLflow
            mlflow.log_artifact(str(temp_path))
            logger.debug(f"Logged dataframe: {artifact_file}")
            
            # Cleanup
            temp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log dataframe: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for current run
        
        Args:
            tags: Dictionary of tags
        """
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register model to model registry
        
        Args:
            model_uri: URI of the model (e.g., runs:/<run_id>/model)
            name: Name for registered model
            tags: Model tags
            description: Model description
            
        Returns:
            Model version
        """
        try:
            result = mlflow.register_model(model_uri, name)
            version = result.version
            
            # Add tags and description
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(name, version, key, value)
            
            if description:
                self.client.update_model_version(
                    name=name,
                    version=version,
                    description=description
                )
            
            logger.info(f"✅ Model registered: {name} (version {version})")
            return version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition model to a new stage
        
        Args:
            name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing versions in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            logger.info(f"✅ Model {name} v{version} → {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")

    def get_best_run(
        self,
        metric_name: str = "test_f1",
        ascending: bool = False
    ) -> Optional[Any]:
        """
        Get best run based on a metric
        
        Args:
            metric_name: Metric to optimize
            ascending: Lower is better if True
            
        Returns:
            Best run object
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="",
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            
            if runs:
                best_run = runs[0]
                logger.info(f"Best run: {best_run.info.run_id}")
                logger.info(f"{metric_name}: {best_run.data.metrics.get(metric_name)}")
                return best_run
            return None
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (None for all)
            
        Returns:
            DataFrame with run comparisons
        """
        try:
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'status': run.info.status,
                    **run.data.metrics
                }
                runs_data.append(run_data)
            
            df = pd.DataFrame(runs_data)
            
            if metrics:
                cols = ['run_id', 'run_name', 'status'] + metrics
                df = df[[col for col in cols if col in df.columns]]
            
            logger.info(f"Compared {len(run_ids)} runs")
            return df
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return pd.DataFrame()

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[Any]:
        """
        Search runs with filter
        
        Args:
            filter_string: MLflow filter string
            order_by: List of order by clauses
            max_results: Maximum results to return
            
        Returns:
            List of runs
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )
            logger.info(f"Found {len(runs)} runs")
            return runs
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []

    def delete_run(self, run_id: str):
        """Delete a run"""
        try:
            self.client.delete_run(run_id)
            logger.info(f"Deleted run: {run_id}")
        except Exception as e:
            logger.error(f"Failed to delete run: {e}")

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get current experiment information"""
        try:
            experiment = self.client.get_experiment(self.experiment_id)
            return {
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'artifact_location': experiment.artifact_location,
                'lifecycle_stage': experiment.lifecycle_stage,
                'tags': experiment.tags
            }
        except Exception as e:
            logger.error(f"Failed to get experiment info: {e}")
            return {}

    def end_run(self):
        """End current run"""
        try:
            mlflow.end_run()
            logger.info("Ended current MLflow run")
        except Exception as e:
            logger.warning(f"Failed to end run: {e}")
        mlflow.sklearn.log_model(model, artifact_path)
