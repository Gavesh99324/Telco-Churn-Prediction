"""Training pipeline DAG for ECS environment"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'training_pipeline_ecs',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
)


def run_training_pipeline():
    print("Running training pipeline for ECS...")


task = PythonOperator(
    task_id='run_training',
    python_callable=run_training_pipeline,
    dag=dag
)
