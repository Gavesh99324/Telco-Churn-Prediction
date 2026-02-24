"""Model training DAG with proper task orchestration"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'model_training',
    default_args=default_args,
    description='Complete model training pipeline',
    schedule_interval='@weekly',
    catchup=False,
    tags=['training', 'ml', 'mlflow']
)


def validate_preprocessed_data(**context):
    """Validate preprocessed data exists"""
    import pandas as pd
    from pathlib import Path
    
    required_files = [
        'artifacts/data/X_train.csv',
        'artifacts/data/X_test.csv',
        'artifacts/data/y_train.csv',
        'artifacts/data/y_test.csv',
        'artifacts/data/scaler.pkl',
        'artifacts/data/label_encoders.pkl'
    ]
    
    for file in required_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Required file not found: {file}")
    
    # Validate data shapes
    X_train = pd.read_csv('artifacts/data/X_train.csv')
    X_test = pd.read_csv('artifacts/data/X_test.csv')
    
    print(f"✅ Validation passed:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    context['ti'].xcom_push(key='train_shape', value=X_train.shape)
    context['ti'].xcom_push(key='test_shape', value=X_test.shape)
    return True


def train_baseline_models(**context):
    """Train all baseline models"""
    import pandas as pd
    from src.model_building import ModelBuilder
    from src.model_training import ModelTrainer
    from src.model_evaluation import ModelEvaluator
    
    # Load data
    X_train = pd.read_csv('artifacts/data/X_train.csv')
    X_test = pd.read_csv('artifacts/data/X_test.csv')
    y_train = pd.read_csv('artifacts/data/y_train.csv')['Churn'].values
    y_test = pd.read_csv('artifacts/data/y_test.csv')['Churn'].values
    
    # Build models
    builder = ModelBuilder(random_state=42)
    models = builder.build_all_baseline_models()
    
    # Train and evaluate
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model, _ = trainer.train(model, X_train, y_train, name)
        eval_results = evaluator.evaluate_model(trained_model, X_train, y_train, X_test, y_test, name)
        results[name] = eval_results
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_f1'])
    best_name = best_model[0]
    best_f1 = best_model[1]['test_f1']
    
    print(f"✅ Best baseline model: {best_name} (F1={best_f1:.4f})")
    
    context['ti'].xcom_push(key='best_baseline_name', value=best_name)
    context['ti'].xcom_push(key='best_baseline_f1', value=best_f1)
    
    # Save best model
    import pickle
    with open(f'artifacts/models/{best_name}_baseline.pkl', 'wb') as f:
        pickle.dump(models[best_name], f)


def decide_tuning_branch(**context):
    """Decide whether to perform hyperparameter tuning"""
    ti = context['ti']
    best_f1 = ti.xcom_pull(key='best_baseline_f1', task_ids='train_baseline')
    
    # Only tune if F1 < 0.65
    if best_f1 < 0.65:
        print(f"F1={best_f1:.4f} < 0.65, performing hyperparameter tuning")
        return 'hyperparameter_tuning'
    else:
        print(f"F1={best_f1:.4f} >= 0.65, skipping tuning")
        return 'skip_tuning'


def hyperparameter_tuning(**context):
    """Perform hyperparameter tuning"""
    import pandas as pd
    from src.model_training import ModelTrainer
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    X_train = pd.read_csv('artifacts/data/X_train.csv')
    y_train = pd.read_csv('artifacts/data/y_train.csv')['Churn'].values
    
    # Define param grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    print("Starting hyperparameter tuning...")
    trainer = ModelTrainer()
    model = RandomForestClassifier(random_state=42)
    
    tuned_model, tuning_info = trainer.hyperparameter_tuning_grid(
        model, param_grid, X_train, y_train,
        model_name='RandomForest_Tuned', cv=5, scoring='f1'
    )
    
    print(f"✅ Tuning complete. Best score: {tuning_info['best_score']:.4f}")
    print(f"   Best params: {tuning_info['best_params']}")
    
    context['ti'].xcom_push(key='tuned_score', value=tuning_info['best_score'])


def train_with_smote(**context):
    """Train model with SMOTE"""
    import pandas as pd
    from src.model_training import ModelTrainer
    from src.model_evaluation import ModelEvaluator
    from sklearn.linear_model import LogisticRegression
    
    # Load data
    X_train = pd.read_csv('artifacts/data/X_train.csv')
    X_test = pd.read_csv('artifacts/data/X_test.csv')
    y_train = pd.read_csv('artifacts/data/y_train.csv')['Churn'].values
    y_test = pd.read_csv('artifacts/data/y_test.csv')['Churn'].values
    
    print("Training with SMOTE...")
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    smote_model, smote_info = trainer.train_with_smote(model, X_train, y_train,
                                                        model_name='LogReg_SMOTE', k_neighbors=5)
    
    results = evaluator.evaluate_model(smote_model, X_train, y_train, X_test, y_test, 'LogReg_SMOTE')
    
    print(f"✅ SMOTE training complete. F1={results['test_f1']:.4f}")
    
    context['ti'].xcom_push(key='smote_f1', value=results['test_f1'])


def train_ensemble_models(**context):
    """Train ensemble models"""
    import pandas as pd
    from src.model_building import ModelBuilder
    from src.model_training import ModelTrainer
    from src.model_evaluation import ModelEvaluator
    
    # Load data
    X_train = pd.read_csv('artifacts/data/X_train.csv')
    X_test = pd.read_csv('artifacts/data/X_test.csv')
    y_train = pd.read_csv('artifacts/data/y_train.csv')['Churn'].values
    y_test = pd.read_csv('artifacts/data/y_test.csv')['Churn'].values
    
    builder = ModelBuilder(random_state=42)
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    ensemble_results = {}
    
    # Voting Classifier
    print("Training Voting Classifier...")
    voting_clf = builder.build_voting_classifier(use_smote_params=False)
    voting_model, _ = trainer.train(voting_clf, X_train, y_train, 'Voting_Classifier')
    voting_results = evaluator.evaluate_model(voting_model, X_train, y_train, X_test, y_test, 'Voting')
    ensemble_results['voting'] = voting_results['test_f1']
    
    # Stacking Classifier
    print("Training Stacking Classifier...")
    stacking_clf = builder.build_stacking_classifier(use_smote_params=False)
    stacking_model, _ = trainer.train(stacking_clf, X_train, y_train, 'Stacking_Classifier')
    stacking_results = evaluator.evaluate_model(stacking_model, X_train, y_train, X_test, y_test, 'Stacking')
    ensemble_results['stacking'] = stacking_results['test_f1']
    
    print(f"✅ Ensemble training complete:")
    print(f"   Voting F1: {ensemble_results['voting']:.4f}")
    print(f"   Stacking F1: {ensemble_results['stacking']:.4f}")
    
    context['ti'].xcom_push(key='ensemble_results', value=ensemble_results)


def compare_all_models(**context):
    """Compare all models and select best"""
    from src.model_evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator()
    
    # Generate comparison
    comparison = evaluator.compare_models(save_path='artifacts/models/final_comparison.csv')
    
    best_model = comparison.iloc[0]
    print(f"✅ Best model: {best_model['Model']} (F1={best_model['Test F1']:.4f})")
    
    context['ti'].xcom_push(key='final_best_model', value=best_model['Model'])
    context['ti'].xcom_push(key='final_best_f1', value=best_model['Test F1'])


def register_best_model(**context):
    """Register best model to MLflow"""
    ti = context['ti']
    best_model = ti.xcom_pull(key='final_best_model', task_ids='compare_models')
    best_f1 = ti.xcom_pull(key='final_best_f1', task_ids='compare_models')
    
    print(f"📦 Registering best model: {best_model} (F1={best_f1:.4f})")
    print("   (In production, this would register to MLflow Model Registry)")
    
    # In production with MLflow running:
    # from utils.mlflow_utils import MLflowManager
    # mlflow_manager = MLflowManager()
    # mlflow_manager.register_model(model_uri, "telco-churn-production")
    
    context['ti'].xcom_push(key='registered_model', value=best_model)


def notify_completion(**context):
    """Send completion notification"""
    ti = context['ti']
    
    best_model = ti.xcom_pull(key='final_best_model', task_ids='compare_models')
    best_f1 = ti.xcom_pull(key='final_best_f1', task_ids='compare_models')
    train_shape = ti.xcom_pull(key='train_shape', task_ids='validate_data')
    
    message = f"""
    Model Training Pipeline Completed!
    
    Training samples: {train_shape[0] if train_shape else 'N/A'}
    Best model: {best_model}
    Best F1 score: {best_f1:.4f}
    
    Model registered and ready for deployment.
    """
    
    print(message)
    # In production: send email/slack notification


# Define tasks
validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_preprocessed_data,
    provide_context=True,
    dag=dag
)

baseline_task = PythonOperator(
    task_id='train_baseline',
    python_callable=train_baseline_models,
    provide_context=True,
    dag=dag
)

decide_tuning_task = BranchPythonOperator(
    task_id='decide_tuning',
    python_callable=decide_tuning_branch,
    provide_context=True,
    dag=dag
)

tuning_task = PythonOperator(
    task_id='hyperparameter_tuning',
    python_callable=hyperparameter_tuning,
    provide_context=True,
    dag=dag
)

skip_tuning_task = DummyOperator(
    task_id='skip_tuning',
    dag=dag
)

join_task = DummyOperator(
    task_id='join_branches',
    trigger_rule='none_failed_min_one_success',
    dag=dag
)

smote_task = PythonOperator(
    task_id='train_with_smote',
    python_callable=train_with_smote,
    provide_context=True,
    dag=dag
)

ensemble_task = PythonOperator(
    task_id='train_ensemble',
    python_callable=train_ensemble_models,
    provide_context=True,
    dag=dag
)

compare_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_all_models,
    provide_context=True,
    dag=dag
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_best_model,
    provide_context=True,
    dag=dag
)

notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    provide_context=True,
    dag=dag
)

# Define dependencies
validate_task >> baseline_task >> decide_tuning_task
decide_tuning_task >> [tuning_task, skip_tuning_task]
[tuning_task, skip_tuning_task] >> join_task
join_task >> [smote_task, ensemble_task]
[smote_task, ensemble_task] >> compare_task >> register_task >> notify_task
