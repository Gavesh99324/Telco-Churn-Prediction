# ECS Deployment Guide

## Overview

This guide explains how to deploy the application to AWS ECS.

## Prerequisites

- AWS Account
- AWS CLI configured
- Docker installed

## Deployment Steps

1. Configure environment: `cp ecs-deployment/00_env.sh.example ecs-deployment/00_env.sh`
2. Edit `00_env.sh` with your AWS details
3. Run deployment scripts in order:
   - `./10_bootstrap.sh` - Create ECR and push images
   - `./20_networking.sh` - Setup VPC and security groups
   - `./30_iam.sh` - Create IAM roles
   - `./40_cluster_alb.sh` - Create ECS cluster and ALB
   - `./50_register_tasks.sh` - Register task definitions
   - `./60_services.sh` - Launch ECS services
   - `./70_airflow_init.sh` - Initialize Airflow
   - `./80_airflow_vars.sh` - Set Airflow variables

## Managing Services

- **Restart**: `./restart_ecs.sh`
- **Stop**: `./stop_ecs.sh`
- **Cleanup**: `./90_cleanup_all.sh`

## Monitoring

Check CloudWatch logs for service output.
