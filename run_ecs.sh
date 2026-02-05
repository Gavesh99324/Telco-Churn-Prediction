#!/bin/bash
# ECS deployment script
echo "Deploying to AWS ECS..."
cd ecs-deployment
./10_bootstrap.sh
./20_networking.sh
./30_iam.sh
./40_cluster_alb.sh
./50_register_tasks.sh
./60_services.sh
./70_airflow_init.sh
./80_airflow_vars.sh
echo "ECS deployment completed!"
