"""Model training DAG"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
)


def run_model_training():
    from pipelines.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    pipeline.run()


task = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag
)
