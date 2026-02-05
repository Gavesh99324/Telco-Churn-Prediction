# Command Cheat Sheet

## Docker Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start local environment
./run_local.sh
```

## AWS ECS

```bash
# Deploy to ECS
./run_ecs.sh

# Stop ECS services
cd ecs-deployment && ./stop_ecs.sh

# Restart services
cd ecs-deployment && ./restart_ecs.sh
```

## Airflow

```bash
# Access Airflow UI
# http://localhost:8080

# Trigger DAG
# Use Airflow UI or CLI
```
