"""Data pipeline DAG with proper task orchestration"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
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
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Complete data preprocessing pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['data', 'preprocessing', 'ml']
)


def validate_raw_data(**context):
    """Validate raw data exists and is valid"""
    import pandas as pd
    from pathlib import Path
    
    data_path = 'data/raw/telco_churn.csv'
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if len(df) == 0:
        raise ValueError("Data file is empty")
    
    required_columns = ['customerID', 'gender', 'tenure', 'MonthlyCharges', 'Churn']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Validation passed: {len(df)} records, {len(df.columns)} columns")
    context['ti'].xcom_push(key='record_count', value=len(df))
    return True


def run_data_ingestion(**context):
    """Load and validate raw data"""
    from src.data_ingestion import DataIngestion
    
    ingestion = DataIngestion(data_path='data/raw/telco_churn.csv')
    df = ingestion.ingest()
    
    print(f"✅ Ingestion complete: {len(df)} records")
    context['ti'].xcom_push(key='ingestion_records', value=len(df))


def handle_missing_values(**context):
    """Handle missing values"""
    from src.handle_missing_values import MissingValueHandler
    import pandas as pd
    
    # Load ingested data (in production, use proper artifact storage)
    # For now, re-load from source
    df = pd.read_csv('data/raw/telco_churn.csv')
    
    handler = MissingValueHandler()
    df_clean = handler.handle_missing(df)
    
    # Save intermediate result
    df_clean.to_csv('artifacts/data/01_missing_handled.csv', index=False)
    
    print(f"✅ Missing values handled: {len(df_clean)} records")


def detect_outliers(**context):
    """Detect outliers"""
    from src.outlier_detection import OutlierDetector
    import pandas as pd
    
    df = pd.read_csv('artifacts/data/01_missing_handled.csv')
    
    detector = OutlierDetector(multiplier=1.5)
    df_flagged = detector.handle_outliers(df, remove=False)
    
    # Save
    df_flagged.to_csv('artifacts/data/02_outliers_detected.csv', index=False)
    
    print(f"✅ Outliers detected: {len(df_flagged)} records")


def engineer_features(**context):
    """Engineer new features"""
    from src.feature_engineering import FeatureEngineer
    import pandas as pd
    
    df = pd.read_csv('artifacts/data/02_outliers_detected.csv')
    
    # Separate target
    if 'Churn' in df.columns:
        y = df['Churn']
        X = df.drop('Churn', axis=1)
    else:
        X = df
        y = None
    
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    
    # Re-add target
    if y is not None:
        X_engineered['Churn'] = y.values
    
    # Save
    X_engineered.to_csv('artifacts/data/03_features_engineered.csv', index=False)
    
    print(f"✅ Features engineered: {X_engineered.shape}")
    context['ti'].xcom_push(key='feature_count', value=X_engineered.shape[1])


def encode_and_scale(**context):
    """Encode categorical features and scale"""
    from src.feature_encoding import FeatureEncoder
    from src.feature_scaling import FeatureScaler
    from src.feature_binning import FeatureBinner
    import pandas as pd
    import pickle
    
    df = pd.read_csv('artifacts/data/03_features_engineered.csv')
    
    # Separate target
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # Binning
    binner = FeatureBinner()
    X = binner.create_tenure_groups(X)
    
    # Encoding
    encoder = FeatureEncoder()
    X_encoded = encoder.fit_transform(X, y)
    
    # Scaling
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Save encoded data
    X_scaled['Churn'] = y.values
    X_scaled.to_csv('artifacts/data/04_encoded_scaled.csv', index=False)
    
    # Save transformers
    with open('artifacts/data/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoder.label_encoders, f)
    
    with open('artifacts/data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler.scaler, f)
    
    print(f"✅ Encoding and scaling complete: {X_scaled.shape}")


def split_and_save(**context):
    """Split into train/test and save final artifacts"""
    from src.data_splitter import DataSplitter
    import pandas as pd
    import pickle
    
    df = pd.read_csv('artifacts/data/04_encoded_scaled.csv')
    
    # Separate features and target
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # Split
    splitter = DataSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    # Save splits
    X_train.to_csv('artifacts/data/X_train.csv', index=False)
    X_test.to_csv('artifacts/data/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['Churn']).to_csv('artifacts/data/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['Churn']).to_csv('artifacts/data/y_test.csv', index=False)
    
    # Save feature names
    with open('artifacts/data/feature_names.pkl', 'wb') as f:
        pickle.dump({'feature_names': X_train.columns.tolist()}, f)
    
    print(f"✅ Data split complete:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    context['ti'].xcom_push(key='train_size', value=len(X_train))
    context['ti'].xcom_push(key='test_size', value=len(X_test))


def notify_completion(**context):
    """Send completion notification"""
    ti = context['ti']
    
    record_count = ti.xcom_pull(key='record_count', task_ids='validate_data')
    train_size = ti.xcom_pull(key='train_size', task_ids='split_and_save')
    test_size = ti.xcom_pull(key='test_size', task_ids='split_and_save')
    feature_count = ti.xcom_pull(key='feature_count', task_ids='engineer_features')
    
    message = f"""
    Data Pipeline Completed Successfully!
    
    Records processed: {record_count}
    Features engineered: {feature_count}
    Train set: {train_size}
    Test set: {test_size}
    """
    
    print(message)
    # In production, send email/slack notification


# Define tasks
validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_raw_data,
    provide_context=True,
    dag=dag
)

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=run_data_ingestion,
    provide_context=True,
    dag=dag
)

missing_task = PythonOperator(
    task_id='handle_missing_values',
    python_callable=handle_missing_values,
    provide_context=True,
    dag=dag
)

outliers_task = PythonOperator(
    task_id='detect_outliers',
    python_callable=detect_outliers,
    provide_context=True,
    dag=dag
)

features_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    provide_context=True,
    dag=dag
)

encode_task = PythonOperator(
    task_id='encode_and_scale',
    python_callable=encode_and_scale,
    provide_context=True,
    dag=dag
)

split_task = PythonOperator(
    task_id='split_and_save',
    python_callable=split_and_save,
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
validate_task >> ingest_task >> missing_task >> outliers_task >> features_task >> encode_task >> split_task >> notify_task
