"""Data pipeline DAG"""
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
    'data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)


def run_data_ingestion():
    from pipelines.data_pipeline import DataPipeline
    pipeline = DataPipeline()
    pipeline.run()


task = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    dag=dag
)
