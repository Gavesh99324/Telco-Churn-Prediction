"""Data pipeline DAG for ECS environment"""
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
    'data_pipeline_ecs',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)


def run_data_pipeline():
    print("Running data pipeline for ECS...")


task = PythonOperator(
    task_id='run_pipeline',
    python_callable=run_data_pipeline,
    dag=dag
)
