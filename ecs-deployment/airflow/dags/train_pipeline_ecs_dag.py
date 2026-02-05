"""Training pipeline DAG for ECS"""
from airflow import DAG
from datetime import datetime

with DAG('training_pipeline_ecs', start_date=datetime(2024, 1, 1)) as dag:
    pass
