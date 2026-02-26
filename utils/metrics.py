"""Prometheus metrics for monitoring ML services"""
import time
from functools import wraps
from typing import Callable, Any
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# Create custom registry for better control
REGISTRY = CollectorRegistry()

# ============================================================================
# ML MODEL METRICS
# ============================================================================

# Inference metrics
INFERENCE_REQUESTS = Counter(
    'ml_inference_requests_total',
    'Total number of inference requests',
    ['model_name', 'status'],
    registry=REGISTRY
)

INFERENCE_LATENCY = Histogram(
    'ml_inference_latency_seconds',
    'Inference request latency in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05,
             0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
    registry=REGISTRY
)

PREDICTION_SCORES = Histogram(
    'ml_prediction_scores',
    'Distribution of prediction scores',
    ['model_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY
)

PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_name', 'prediction_class'],
    registry=REGISTRY
)

MODEL_LOAD_TIME = Gauge(
    'ml_model_load_seconds',
    'Time taken to load the model',
    ['model_name'],
    registry=REGISTRY
)

PREPROCESSING_ERRORS = Counter(
    'ml_preprocessing_errors_total',
    'Total number of preprocessing errors',
    ['error_type'],
    registry=REGISTRY
)

# ============================================================================
# KAFKA METRICS
# ============================================================================

KAFKA_MESSAGES_PRODUCED = Counter(
    'kafka_messages_produced_total',
    'Total messages produced to Kafka',
    ['topic', 'status'],
    registry=REGISTRY
)

KAFKA_MESSAGES_CONSUMED = Counter(
    'kafka_messages_consumed_total',
    'Total messages consumed from Kafka',
    ['topic', 'consumer_group', 'status'],
    registry=REGISTRY
)

KAFKA_CONSUMER_LAG = Gauge(
    'kafka_consumer_lag',
    'Current consumer lag',
    ['topic', 'consumer_group', 'partition'],
    registry=REGISTRY
)

KAFKA_PROCESSING_LATENCY = Histogram(
    'kafka_message_processing_seconds',
    'Time to process a Kafka message',
    ['topic', 'consumer_group'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05,
             0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

KAFKA_BATCH_SIZE = Histogram(
    'kafka_batch_size',
    'Size of message batches',
    ['topic'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
    registry=REGISTRY
)

# ============================================================================
# DATA PIPELINE METRICS
# ============================================================================

PIPELINE_RUNS = Counter(
    'pipeline_runs_total',
    'Total number of pipeline runs',
    ['pipeline_name', 'status'],
    registry=REGISTRY
)

PIPELINE_DURATION = Histogram(
    'pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline_name', 'stage'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
    registry=REGISTRY
)

RECORDS_PROCESSED = Counter(
    'records_processed_total',
    'Total records processed',
    ['pipeline_name', 'stage'],
    registry=REGISTRY
)

DATA_QUALITY_SCORE = Gauge(
    'data_quality_score',
    'Data quality score (0-100)',
    ['pipeline_name', 'metric'],
    registry=REGISTRY
)

# ============================================================================
# SYSTEM HEALTH METRICS
# ============================================================================

SERVICE_UP = Gauge(
    'service_up',
    'Service health status (1=up, 0=down)',
    ['service_name'],
    registry=REGISTRY
)

ERROR_RATE = Counter(
    'errors_total',
    'Total number of errors',
    ['service_name', 'error_type', 'severity'],
    registry=REGISTRY
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    ['service_name', 'connection_type'],
    registry=REGISTRY
)

MEMORY_USAGE_BYTES = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['service_name'],
    registry=REGISTRY
)

CPU_USAGE_PERCENT = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    ['service_name'],
    registry=REGISTRY
)

# ============================================================================
# TRAINING METRICS
# ============================================================================

MODEL_TRAINING_DURATION = Histogram(
    'model_training_duration_seconds',
    'Model training duration',
    ['model_type', 'dataset_size'],
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200],
    registry=REGISTRY
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy score',
    ['model_name', 'metric_type'],
    registry=REGISTRY
)

TRAINING_SAMPLES = Gauge(
    'training_samples_total',
    'Number of training samples',
    ['model_name', 'dataset'],
    registry=REGISTRY
)

# ============================================================================
# HELPER FUNCTIONS AND DECORATORS
# ============================================================================


def track_inference_time(model_name: str):
    """Decorator to track inference time"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                INFERENCE_LATENCY.labels(
                    model_name=model_name).observe(duration)
                INFERENCE_REQUESTS.labels(
                    model_name=model_name, status='success').inc()
                return result
            except Exception as e:
                INFERENCE_REQUESTS.labels(
                    model_name=model_name, status='error').inc()
                ERROR_RATE.labels(
                    service_name='inference',
                    error_type=type(e).__name__,
                    severity='error'
                ).inc()
                raise
        return wrapper
    return decorator


def track_pipeline_stage(pipeline_name: str, stage: str):
    """Decorator to track pipeline stage execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                PIPELINE_DURATION.labels(
                    pipeline_name=pipeline_name,
                    stage=stage
                ).observe(duration)
                PIPELINE_RUNS.labels(
                    pipeline_name=pipeline_name,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                PIPELINE_RUNS.labels(
                    pipeline_name=pipeline_name,
                    status='error'
                ).inc()
                ERROR_RATE.labels(
                    service_name='pipeline',
                    error_type=type(e).__name__,
                    severity='error'
                ).inc()
                raise
        return wrapper
    return decorator


def track_kafka_processing(topic: str, consumer_group: str):
    """Decorator to track Kafka message processing"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                KAFKA_PROCESSING_LATENCY.labels(
                    topic=topic,
                    consumer_group=consumer_group
                ).observe(duration)
                KAFKA_MESSAGES_CONSUMED.labels(
                    topic=topic,
                    consumer_group=consumer_group,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                KAFKA_MESSAGES_CONSUMED.labels(
                    topic=topic,
                    consumer_group=consumer_group,
                    status='error'
                ).inc()
                ERROR_RATE.labels(
                    service_name='kafka_consumer',
                    error_type=type(e).__name__,
                    severity='error'
                ).inc()
                raise
        return wrapper
    return decorator


class MetricsCollector:
    """Centralized metrics collection"""

    @staticmethod
    def record_prediction(model_name: str, prediction: int, score: float):
        """Record a prediction"""
        PREDICTIONS_TOTAL.labels(
            model_name=model_name,
            prediction_class=str(prediction)
        ).inc()
        PREDICTION_SCORES.labels(model_name=model_name).observe(score)

    @staticmethod
    def record_kafka_message(topic: str, status: str = 'success'):
        """Record Kafka message produced"""
        KAFKA_MESSAGES_PRODUCED.labels(topic=topic, status=status).inc()

    @staticmethod
    def set_consumer_lag(topic: str, consumer_group: str, partition: int, lag: int):
        """Set consumer lag for monitoring"""
        KAFKA_CONSUMER_LAG.labels(
            topic=topic,
            consumer_group=consumer_group,
            partition=str(partition)
        ).set(lag)

    @staticmethod
    def record_data_quality(pipeline_name: str, metric_name: str, score: float):
        """Record data quality metric"""
        DATA_QUALITY_SCORE.labels(
            pipeline_name=pipeline_name,
            metric=metric_name
        ).set(score)

    @staticmethod
    def set_service_health(service_name: str, is_healthy: bool):
        """Set service health status"""
        SERVICE_UP.labels(service_name=service_name).set(
            1 if is_healthy else 0)

    @staticmethod
    def record_error(service_name: str, error_type: str, severity: str = 'error'):
        """Record an error"""
        ERROR_RATE.labels(
            service_name=service_name,
            error_type=error_type,
            severity=severity
        ).inc()

    @staticmethod
    def set_system_metrics(service_name: str, memory_bytes: int, cpu_percent: float):
        """Set system resource metrics"""
        MEMORY_USAGE_BYTES.labels(service_name=service_name).set(memory_bytes)
        CPU_USAGE_PERCENT.labels(service_name=service_name).set(cpu_percent)


def get_metrics_text() -> str:
    """Get metrics in Prometheus text format"""
    return generate_latest(REGISTRY).decode('utf-8')


def get_metrics_content_type() -> str:
    """Get content type for metrics endpoint"""
    return CONTENT_TYPE_LATEST
