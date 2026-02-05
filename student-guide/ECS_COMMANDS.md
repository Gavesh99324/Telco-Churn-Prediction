# ECS Commands (Student-Friendly)

## Quick Start

```bash
cd ecs-deployment
./00_env.sh  # Configure first
./10_bootstrap.sh
./20_networking.sh
./30_iam.sh
./40_cluster_alb.sh
./50_register_tasks.sh
./60_services.sh
./70_airflow_init.sh
./80_airflow_vars.sh
```

## Management

```bash
# Restart all services
./restart_ecs.sh

# Stop all services
./stop_ecs.sh

# Cleanup everything
./90_cleanup_all.sh
```

## Troubleshooting

- Check CloudWatch logs
- Verify security group rules
- Ensure IAM roles are correct
