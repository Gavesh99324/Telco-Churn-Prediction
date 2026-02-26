# Monitoring Implementation Quick Start

This guide shows how to add monitoring to the remaining Kafka services (inference and analytics).

## What's Already Done

✅ **Infrastructure Created**:

- `utils/metrics.py` - Prometheus metrics collection
- `utils/structured_logger.py` - Structured JSON logging
- `utils/health_check.py` - Health check server
- `monitoring/prometheus/prometheus.yml` - Prometheus configuration
- `monitoring/grafana/dashboards/` - 3 pre-configured dashboards
- `docker-compose.monitoring.yml` - Monitoring stack

✅ **Producer Service Updated**:

- Metrics tracking for Kafka messages
- Structured logging
- Health check endpoints (port 8001)
- System resource monitoring

✅ **Docker Configuration Updated**:

- Health checks added to all Kafka services
- Metrics ports exposed (8001, 8002, 8003)
- Environment variables for monitoring

## Next Steps

### Step 1: Update Kafka Inference Service

Add the following to [kafka/inference_service.py](kafka/inference_service.py):

```python
# At the top, replace logging imports
from utils.structured_logger import get_logger
from utils.metrics import (
    MetricsCollector,
    track_inference_time,
    track_kafka_processing
)
from utils.health_check import HealthCheckServer, ServiceHealthMonitor

logger = get_logger('kafka-inference', environment=os.getenv('ENVIRONMENT', 'development'))

# In __init__ method of ChurnInferenceService:
self.health_server = None
self.health_monitor = None

# In run() method, add at the start:
# Start health check server
health_port = int(os.getenv('HEALTH_CHECK_PORT', '8002'))
self.health_server = HealthCheckServer('kafka-inference', port=health_port)

def check_model_loaded():
    return self.model is not None

def check_kafka_connection():
    return self.consumer is not None

self.health_server.add_readiness_check('model_loaded', check_model_loaded)
self.health_server.add_readiness_check('kafka_connection', check_kafka_connection)
self.health_server.start_background()

# Start health monitoring
self.health_monitor = ServiceHealthMonitor('kafka-inference', check_interval=30)
self.health_monitor.start()

# Set ready after initialization
self.health_server.set_ready(True)

# In process_message method, wrap with decorator:
@track_inference_time(model_name='churn_model')
@track_kafka_processing(topic='customer-data', consumer_group='churn-inference-group')
def process_message(self, message):
    # existing code...

    # After prediction:
    MetricsCollector.record_prediction(
        model_name='churn_model',
        prediction=int(prediction[0]),
        score=float(proba[0][1])
    )

    # Log inference
    logger.log_inference(
        model_name='churn_model',
        input_shape=features.shape,
        prediction=int(prediction[0]),
        latency_ms=latency * 1000,
        confidence=float(proba[0][1])
    )
```

### Step 2: Update Kafka Analytics Service

Add the following to [kafka/analytics_service.py](kafka/analytics_service.py):

```python
# At the top, replace logging imports
from utils.structured_logger import get_logger
from utils.metrics import MetricsCollector, track_kafka_processing
from utils.health_check import HealthCheckServer, ServiceHealthMonitor

logger = get_logger('kafka-analytics', environment=os.getenv('ENVIRONMENT', 'development'))

# In __init__ method:
self.health_server = None
self.health_monitor = None

# In run() method, add at the start:
# Start health check server
health_port = int(os.getenv('HEALTH_CHECK_PORT', '8003'))
self.health_server = HealthCheckServer('kafka-analytics', port=health_port)

def check_db_connection():
    return self.db_manager is not None

def check_kafka_connection():
    return self.consumer is not None

self.health_server.add_readiness_check('database', check_db_connection)
self.health_server.add_readiness_check('kafka_connection', check_kafka_connection)
self.health_server.start_background()

# Start health monitoring
self.health_monitor = ServiceHealthMonitor('kafka-analytics', check_interval=30)
self.health_monitor.start()

# Set ready after initialization
self.health_server.set_ready(True)

# In process_message method:
@track_kafka_processing(topic='churn-predictions', consumer_group='churn-analytics-group')
def process_message(self, message):
    # existing code...

    # After processing:
    logger.log_kafka_message(
        topic='churn-predictions',
        action='consumed',
        message_count=1,
        partition=message.partition,
        offset=message.offset
    )
```

### Step 3: Rebuild and Deploy

```bash
# Rebuild Kafka services with updated code
docker-compose -f docker-compose.kafka.yml build

# Restart services
docker-compose -f docker-compose.kafka.yml up -d

# Verify health endpoints
curl http://localhost:8001/health  # Producer
curl http://localhost:8002/health  # Inference
curl http://localhost:8003/health  # Analytics

# Check metrics endpoints
curl http://localhost:8001/metrics | grep ml_
curl http://localhost:8002/metrics | grep kafka_
curl http://localhost:8003/metrics | grep service_
```

### Step 4: Deploy Monitoring Stack

```bash
# Start Prometheus and Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# Verify all containers running
docker-compose -f docker-compose.monitoring.yml ps

# Check Prometheus targets (all should be UP)
open http://localhost:9090/targets

# Access Grafana dashboards
open http://localhost:3000
# Login: admin / admin
# Navigate to Dashboards → Browse → MLOps folder
```

### Step 5: Verify Metrics in Grafana

1. Open **ML Model Performance** dashboard
   - Should see inference request rates
   - Latency percentiles (p50, p95, p99)
   - Prediction distribution

2. Open **Kafka Streaming Metrics** dashboard
   - Should see messages produced/consumed
   - Consumer lag by partition
   - Processing latency

3. Open **System Health & Resources** dashboard
   - All services should show "UP" status
   - Memory and CPU usage populated
   - Error rates visible

### Step 6: Test Monitoring

```python
# Run a test script to generate metrics
python -c "
from utils.metrics import MetricsCollector

# Simulate predictions
for i in range(100):
    MetricsCollector.record_prediction(
        model_name='churn_model',
        prediction=i % 2,
        score=0.5 + (i % 50) / 100
    )

# Simulate Kafka messages
for i in range(50):
    MetricsCollector.record_kafka_message('customer-data', 'success')

print('Test metrics generated. Check Grafana dashboards.')
"
```

## Monitoring Checklist

- [ ] Inference service updated with metrics and logging
- [ ] Analytics service updated with metrics and logging
- [ ] All services rebuilt: `docker-compose -f docker-compose.kafka.yml build`
- [ ] Services restarted: `docker-compose -f docker-compose.kafka.yml up -d`
- [ ] Health endpoints responding: `curl localhost:800{1,2,3}/health`
- [ ] Metrics endpoints working: `curl localhost:800{1,2,3}/metrics`
- [ ] Monitoring stack deployed: `docker-compose -f docker-compose.monitoring.yml up -d`
- [ ] Prometheus targets UP: http://localhost:9090/targets
- [ ] Grafana dashboards visible: http://localhost:3000
- [ ] All 3 dashboards showing data
- [ ] Health check tables showing all services UP
- [ ] Logs in JSON format: `docker logs telco-producer | jq`

## Troubleshooting

### Health Endpoint Returns 404

**Problem**: `curl http://localhost:8001/health` returns connection refused or 404.

**Solution**:

1. Verify Flask is installed: `docker exec telco-producer pip list | grep -i flask`
2. Check if health server started: `docker logs telco-producer | grep "Health check"`
3. Verify port is exposed in docker-compose: `docker port telco-producer`

### Metrics Not Appearing in Prometheus

**Problem**: Prometheus targets show "DOWN" or metrics not visible.

**Solution**:

1. Check Prometheus logs: `docker logs telco-prometheus`
2. Verify network connectivity: `docker exec telco-prometheus ping kafka-producer`
3. Check metrics endpoint manually: `curl http://localhost:8001/metrics`
4. Verify target configuration: http://localhost:9090/config

### Grafana Dashboard Shows "No Data"

**Problem**: Dashboards load but panels show "No Data".

**Solution**:

1. Verify Prometheus datasource: Settings → Data Sources → Prometheus → Test
2. Check if metrics exist in Prometheus: http://localhost:9090/graph
3. Verify time range in dashboard (last 15 minutes)
4. Generate test metrics using the test script above

### Services Not Producing Metrics

**Problem**: `/metrics` endpoint exists but shows no custom metrics.

**Solution**:

1. Verify MetricsCollector is being called: Add debug logging
2. Check for import errors: `docker logs telco-producer | grep -i error`
3. Verify registry is being used: Check `utils/metrics.py` REGISTRY usage
4. Test metrics locally: Run the service outside Docker

## Production Deployment

For production, consider:

1. **Security**: Change default Grafana password
2. **Persistence**: Configure volume backups for Prometheus data
3. **Alerting**: Enable AlertManager and configure notification channels
4. **Retention**: Adjust Prometheus retention based on storage
5. **Resource Limits**: Set CPU/memory limits in docker-compose
6. **TLS**: Enable HTTPS for Grafana and Prometheus
7. **Federation**: Configure Prometheus federation for multi-cluster
8. **High Availability**: Deploy Prometheus in HA mode with Thanos

## Additional Resources

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Design](https://grafana.com/docs/grafana/latest/dashboards/)
- [Python Client Documentation](https://github.com/prometheus/client_python)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)

---

**Next**: After completing monitoring, consider implementing:

- API layer (FastAPI) with metrics
- Streamlit dashboard for real-time visualization
- ELK stack integration for centralized logging
- Distributed tracing with Jaeger
