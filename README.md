# Telco Churn Prediction

End-to-end MLOps platform for predicting customer churn with real-time streaming and automated retraining.

## Features

- 🌊 **Apache Airflow** for workflow orchestration
- 📊 **Kafka** for real-time streaming
- 🤖 **MLflow** for experiment tracking and model registry
- 🐳 **Docker** containerization with health checks
- 📈 **Prometheus + Grafana** for monitoring and observability
- 📝 **Structured JSON logging** for ELK/CloudWatch integration
- ☁️ **AWS ECS** deployment ready
- 🧪 **Comprehensive testing** suite (unit + integration)
- 🔄 **CI/CD** with GitHub Actions

## Quick Start

### 1. Infrastructure Setup

```bash
# Start Kafka streaming infrastructure
docker-compose -f docker-compose.kafka.yml up -d

# Start Airflow orchestration
docker-compose -f docker-compose.airflow.yml up -d

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access Services

| Service          | URL                          | Credentials   |
| ---------------- | ---------------------------- | ------------- |
| Airflow UI       | http://localhost:8080        | admin / admin |
| Kafka UI         | http://localhost:8090        | -             |
| Grafana          | http://localhost:3000        | admin / admin |
| Prometheus       | http://localhost:9090        | -             |
| Producer Health  | http://localhost:8001/health | -             |
| Inference Health | http://localhost:8002/health | -             |
| Analytics Health | http://localhost:8003/health | -             |

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python pipelines/data_pipeline.py

# Run model training
python pipelines/training_pipeline.py

# Run tests
pytest tests/
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Producer   │────▶│    Kafka     │────▶│  Inference   │
│   Service    │     │    Broker    │     │   Service    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                            │                     │
                            │                     ▼
                            │              ┌──────────────┐
                            └─────────────▶│  Analytics   │
                                           │   Service    │
                                           └──────────────┘
                                                  │
                     ┌────────────────────────────┼────────────────┐
                     │                            │                │
                     ▼                            ▼                ▼
              ┌─────────────┐            ┌─────────────┐   ┌──────────┐
              │ Prometheus  │            │  Grafana    │   │ Postgres │
              │  (Metrics)  │            │(Dashboards) │   │Analytics │
              └─────────────┘            └─────────────┘   └──────────┘
```

## Monitoring & Observability

The platform includes comprehensive monitoring with Prometheus and Grafana.

### Grafana Dashboards

1. **ML Model Performance**: Inference metrics, latency, predictions, errors
2. **Kafka Streaming**: Message rates, consumer lag, processing latency
3. **System Health**: Resource usage, service status, error rates

See [monitoring/README.md](monitoring/README.md) for detailed documentation.

### Health Checks

All services expose health endpoints:

- `/health` - Liveness probe
- `/ready` - Readiness probe
- `/metrics` - Prometheus metrics
- `/status` - Detailed service status

### Metrics Examples

```python
from utils.metrics import MetricsCollector

# Record prediction
MetricsCollector.record_prediction(
    model_name='churn_model',
    prediction=1,
    score=0.87
)

# Track Kafka lag
MetricsCollector.set_consumer_lag(
    topic='customer-data',
    consumer_group='churn-inference-group',
    partition=0,
    lag=42
)
```

## Project Structure

```
├── airflow/              # Airflow DAGs and configuration
├── artifacts/            # Model artifacts and preprocessed data
├── data/                 # Raw data files
├── docker/               # Dockerfiles for all services
├── ecs-deployment/       # AWS ECS deployment scripts
├── kafka/                # Kafka streaming services
├── monitoring/           # Prometheus, Grafana, dashboards
├── pipelines/            # ML pipelines (data, training, inference)
├── scripts/              # Utility scripts
├── src/                  # Core ML modules
├── tests/                # Unit and integration tests
└── utils/                # Helper modules (metrics, logging, config)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=pipelines --cov-report=html

# Run specific test file
pytest tests/unit/test_model_building.py
```

### Code Quality

```bash
# Format code
black src/ pipelines/ kafka/

# Lint
flake8 src/ pipelines/ kafka/

# Type checking
mypy src/ pipelines/
```

## AWS ECS Deployment

See `ecs-deployment/` folder for deployment scripts.
