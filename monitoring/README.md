# Monitoring & Observability

Comprehensive monitoring stack for the Telco Churn Prediction platform using Prometheus, Grafana, and structured logging.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Services  │────▶│ Prometheus  │────▶│   Grafana   │
│   Metrics   │     │  (Storage)  │     │ (Dashboards)│
└─────────────┘     └─────────────┘     └─────────────┘
       │
       │            ┌─────────────┐
       └───────────▶│  Exporters  │
                    │  (Node, DB) │
                    └─────────────┘
```

## Components

### 1. **Prometheus** (Port 9090)

- Time-series metrics database
- Scrapes metrics from services every15s
- 30-day data retention
- Alert evaluation engine

### 2. **Grafana** (Port 3000)

- Visualization platform
- Pre-configured dashboards
- Default credentials: `admin/admin`
- Auto-provisioned datasources

### 3. **Metrics Exporters**

- **Node Exporter** (9100): Host system metrics (CPU, memory, disk)
- **cAdvisor** (8081): Docker container metrics
- **PostgreSQL Exporters** (9187, 9188): Database metrics

### 4. **Service Metrics Endpoints**

- **Kafka Producer** (8001): `/metrics`, `/health`, `/ready`
- **Kafka Inference** (8002): `/metrics`, `/health`, `/ready`
- **Kafka Analytics** (8003): `/metrics`, `/health`, `/ready`

## Quick Start

### Deploy Monitoring Stack

```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services
docker-compose -f docker-compose.monitoring.yml ps

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Access Grafana
open http://localhost:3000
```

### Deploy Services with Metrics

```bash
# Rebuild Kafka services with monitoring (after updating requirements)
docker-compose -f docker-compose.kafka.yml build

# Start Kafka stack with health checks
docker-compose -f docker-compose.kafka.yml up -d

# Verify health endpoints
curl http://localhost:8001/health  # Producer
curl http://localhost:8002/health  # Inference
curl http://localhost:8003/health  # Analytics
```

## Dashboards

### 1. ML Model Performance

**Path**: `/dashboards` → "ML Model Performance"

Panels:

- Inference request rate (success/error)
- Inference latency percentiles (p50, p95, p99)
- Total predictions (24h)
- Prediction class distribution
- Error rate gauge
- Prediction score heatmap
- Preprocessing errors by type
- Model load time
- Service health status table

**Key Metrics**:

- `ml_inference_requests_total`: Total inference requests
- `ml_inference_latency_seconds`: Inference latency histogram
- `ml_predictions_total`: Predictions by class
- `ml_prediction_scores`: Score distribution
- `ml_preprocessing_errors_total`: Preprocessing errors

### 2. Kafka Streaming Metrics

**Path**: `/dashboards` → "Kafka Streaming Metrics"

Panels:

- Messages produced per topic
- Messages consumed per consumer group
- Consumer lag by partition (with alerts)
- Message processing latency (p50, p95, p99)
- Total messages 24h (produced/consumed)
- Message error rate gauge
- Batch size distribution
- Error types breakdown
- Consumer group status table

**Key Metrics**:

- `kafka_messages_produced_total`: Messages produced
- `kafka_messages_consumed_total`: Messages consumed
- `kafka_consumer_lag`: Consumer lag by partition
- `kafka_message_processing_seconds`: Processing latency
- `kafka_batch_size`: Batch size histogram

**Alerts**:

- High consumer lag (>1000 messages)

### 3. System Health & Resources

**Path**: `/dashboards` → "System Health & Resources"

Panels:

- Service health overview table
- Memory usage by service
- CPU usage by service
- Error rate by service (with alerts)
- Error distribution by type/severity
- Active connections
- Total services gauge
- Services up/down counters
- Recent error logs

**Key Metrics**:

- `service_up`: Service health status (1=up, 0=down)
- `memory_usage_bytes`: Memory consumption
- `cpu_usage_percent`: CPU utilization
- `errors_total`: Error counter by type/severity
- `active_connections`: Active connections

**Alerts**:

- High error rate (>10 errors/sec)

## Metrics Reference

### ML Metrics

| Metric                          | Type      | Description                   | Labels                       |
| ------------------------------- | --------- | ----------------------------- | ---------------------------- |
| `ml_inference_requests_total`   | Counter   | Total inference requests      | model_name, status           |
| `ml_inference_latency_seconds`  | Histogram | Inference latency             | model_name                   |
| `ml_predictions_total`          | Counter   | Total predictions             | model_name, prediction_class |
| `ml_prediction_scores`          | Histogram | Prediction score distribution | model_name                   |
| `ml_model_load_seconds`         | Gauge     | Model load time               | model_name                   |
| `ml_preprocessing_errors_total` | Counter   | Preprocessing errors          | error_type                   |

### Kafka Metrics

| Metric                             | Type      | Description        | Labels                           |
| ---------------------------------- | --------- | ------------------ | -------------------------------- |
| `kafka_messages_produced_total`    | Counter   | Messages produced  | topic, status                    |
| `kafka_messages_consumed_total`    | Counter   | Messages consumed  | topic, consumer_group, status    |
| `kafka_consumer_lag`               | Gauge     | Consumer lag       | topic, consumer_group, partition |
| `kafka_message_processing_seconds` | Histogram | Processing latency | topic, consumer_group            |
| `kafka_batch_size`                 | Histogram | Batch size         | topic                            |

### System Metrics

| Metric               | Type    | Description                   | Labels                             |
| -------------------- | ------- | ----------------------------- | ---------------------------------- |
| `service_up`         | Gauge   | Service health (1=up, 0=down) | service_name                       |
| `errors_total`       | Counter | Total errors                  | service_name, error_type, severity |
| `memory_usage_bytes` | Gauge   | Memory usage                  | service_name                       |
| `cpu_usage_percent`  | Gauge   | CPU usage                     | service_name                       |
| `active_connections` | Gauge   | Active connections            | service_name, connection_type      |

### Pipeline Metrics

| Metric                      | Type      | Description                | Labels                |
| --------------------------- | --------- | -------------------------- | --------------------- |
| `pipeline_runs_total`       | Counter   | Pipeline runs              | pipeline_name, status |
| `pipeline_duration_seconds` | Histogram | Pipeline duration          | pipeline_name, stage  |
| `records_processed_total`   | Counter   | Records processed          | pipeline_name, stage  |
| `data_quality_score`        | Gauge     | Data quality score (0-100) | pipeline_name, metric |

## Structured Logging

All services use structured JSON logging for integration with ELK/CloudWatch.

### Log Format

```json
{
  "timestamp": "2024-02-24T10:30:45.123456Z",
  "level": "INFO",
  "service": "kafka-producer",
  "environment": "development",
  "message": "Batch milestone reached",
  "logger": "kafka.producer_service",
  "module": "producer_service",
  "function": "send_record",
  "line": 125,
  "records_sent": 1000,
  "topic": "customer-data",
  "partition": 0,
  "offset": 999
}
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational events
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error events with exception details
- **CRITICAL**: Critical errors requiring immediate attention

### Specialized Log Methods

```python
from utils.structured_logger import get_logger

logger = get_logger('my-service', environment='development')

# ML inference log
logger.log_inference(
    model_name='churn_model',
    input_shape=(1, 42),
    prediction=1,
    latency_ms=45.2,
    confidence=0.87
)

# Kafka message log
logger.log_kafka_message(
    topic='customer-data',
    action='produced',
    message_count=100,
    partition=0,
    offset=12345
)

# Pipeline stage log
logger.log_pipeline_stage(
    pipeline_name='data_pipeline',
    stage='feature_engineering',
    status='completed',
    duration_seconds=12.5,
    records_processed=10000
)

# Data quality log
logger.log_data_quality(
    pipeline_name='data_pipeline',
    metric_name='missing_values_percent',
    metric_value=2.5,
    threshold=5.0,
    passed=True
)
```

## Health Checks

All services expose standardized health check endpoints:

### Endpoints

#### `/health` (Liveness Probe)

Indicates if the service is running.

**Response 200 OK**:

```json
{
  "status": "healthy",
  "service": "kafka-producer",
  "timestamp": "2024-02-24T10:30:45.123456Z",
  "uptime_seconds": 3600.25
}
```

#### `/ready` (Readiness Probe)

Indicates if the service is ready to accept traffic.

**Response 200 OK**:

```json
{
  "status": "ready",
  "service": "kafka-producer",
  "timestamp": "2024-02-24T10:30:45.123456Z",
  "checks": {
    "kafka_connection": {
      "status": "pass",
      "passed": true
    }
  }
}
```

**Response 503 Service Unavailable**:

```json
{
  "status": "not_ready",
  "service": "kafka-producer",
  "timestamp": "2024-02-24T10:30:45.123456Z",
  "checks": {
    "kafka_connection": {
      "status": "fail",
      "passed": false
    }
  }
}
```

#### `/metrics` (Prometheus Metrics)

Returns Prometheus-formatted metrics.

#### `/status` (Detailed Status)

Returns detailed service status with system metrics.

**Response 200 OK**:

```json
{
  "service": "kafka-producer",
  "status": "running",
  "ready": true,
  "timestamp": "2024-02-24T10:30:45.123456Z",
  "uptime_seconds": 3600.25,
  "system": {
    "memory": {
      "rss_mb": 512.3,
      "vms_mb": 1024.6,
      "percent": 5.2
    },
    "cpu": {
      "percent": 35.2,
      "num_threads": 8
    },
    "connections": 12
  },
  "python": {
    "version": "3.11.5",
    "pid": 1234
  }
}
```

#### `/ping`

Simple ping endpoint for basic connectivity checks.

**Response 200 OK**:

```json
{ "pong": true }
```

### Docker Health Checks

Services include Docker health checks in `docker-compose.kafka.yml`:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 30s
```

## Usage Examples

### Querying Metrics with PromQL

```promql
# Inference request rate (success)
rate(ml_inference_requests_total{model_name="churn_model", status="success"}[5m])

# 95th percentile inference latency
histogram_quantile(0.95, rate(ml_inference_latency_seconds_bucket[5m]))

# Messages produced per second by topic
sum by (topic) (rate(kafka_messages_produced_total{status="success"}[5m]))

# Consumer lag
kafka_consumer_lag{topic="customer-data", consumer_group="churn-inference-group"}

# Error rate percentage
100 * sum(rate(ml_inference_requests_total{status="error"}[5m]))
    / sum(rate(ml_inference_requests_total[5m]))

# Services down
count(service_up == 0)
```

### Using Metrics in Code

```python
from utils.metrics import MetricsCollector, track_inference_time

# Decorator for automatic timing
@track_inference_time(model_name='churn_model')
def predict(data):
    return model.predict(data)

# Manual metrics
MetricsCollector.record_prediction(
    model_name='churn_model',
    prediction=1,
    score=0.87
)

MetricsCollector.record_kafka_message(
    topic='customer-data',
    status='success'
)

MetricsCollector.set_consumer_lag(
    topic='customer-data',
    consumer_group='churn-inference-group',
    partition=0,
    lag=42
)

MetricsCollector.set_service_health('kafka-producer', True)

MetricsCollector.record_error(
    service_name='kafka-inference',
    error_type='ValueError',
    severity='error'
)
```

## Troubleshooting

### Prometheus Not Scraping Targets

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health, lastError}'

# Check service metrics endpoint
curl http://localhost:8001/metrics

# View Prometheus logs
docker logs telco-prometheus
```

### Grafana Dashboard Not Loading

```bash
# Check Grafana logs
docker logs telco-grafana

# Verify datasource
curl -u admin:admin http://localhost:3000/api/datasources

# Reload provisioning
docker-compose -f docker-compose.monitoring.yml restart grafana
```

### Service Health Check Failing

```bash
# Check health endpoint
curl http://localhost:8001/health

# Check readiness
curl http://localhost:8001/ready

# View service logs
docker logs telco-producer
```

### Missing Metrics

```bash
# Verify metrics are being exposed
curl http://localhost:8001/metrics | grep ml_inference

# Check Prometheus config
docker exec telco-prometheus cat /etc/prometheus/prometheus.yml

# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload
```

## Best Practices

1. **Metrics Naming**: Follow Prometheus naming conventions
   - Use snake_case: `ml_inference_latency_seconds`
   - Include units: `_seconds`, `_bytes`, `_total`
   - Use descriptive labels: `{model_name="churn_model"}`

2. **Cardinality Management**: Avoid high-cardinality labels
   - ❌ Bad: `customer_id`, `timestamp`, `request_id`
   - ✅ Good: `model_name`, `status`, `service_name`

3. **Health Checks**: Implement both liveness and readiness
   - Liveness: Is the process running?
   - Readiness: Can it handle requests?

4. **Logging**: Use structured logging for all services
   - JSON format for easy parsing
   - Include context: service, environment, trace IDs
   - Use appropriate log levels

5. **Alerting**: Set up alerts for critical metrics
   - High error rate (>5%)
   - High consumer lag (>1000 messages)
   - Service down
   - High latency (p95 > 1s)

## Production Considerations

For production deployments:

1. **Enable Loki** for centralized log aggregation
2. **Configure AlertManager** for alert routing
3. **Set up retention policies** based on storage capacity
4. **Enable authentication** for Grafana (change default password)
5. **Use HTTPS** for all external endpoints
6. **Configure backup** for Prometheus data
7. **Set resource limits** for containers
8. **Enable metrics federation** for multi-cluster setups

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)
- [Best Practices for Monitoring](https://prometheus.io/docs/practices/)
