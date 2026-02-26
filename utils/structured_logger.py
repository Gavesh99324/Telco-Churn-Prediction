"""Structured logging with JSON format for ELK/CloudWatch integration"""
import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, service_name: str, environment: str = 'development'):
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'service': self.service_name,
            'environment': self.environment,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                           'levelname', 'levelno', 'lineno', 'module', 'msecs',
                           'message', 'pathname', 'process', 'processName',
                           'relativeCreated', 'thread', 'threadName', 'exc_info',
                           'exc_text', 'stack_info', 'extra_fields']:
                if not key.startswith('_'):
                    log_data[key] = value

        return json.dumps(log_data, default=str)


class StructuredLogger:
    """Structured logger with context support"""

    def __init__(self, service_name: str, environment: str = 'development',
                 log_level: str = 'INFO', log_file: Optional[str] = None):
        self.service_name = service_name
        self.environment = environment
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler with JSON format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            StructuredFormatter(service_name, environment))
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                StructuredFormatter(service_name, environment))
            self.logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with context"""
        extra_fields = kwargs.copy()

        # Create a LogRecord with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.upper()),
            '',  # pathname
            0,   # lineno
            message,
            (),  # args
            None,  # exc_info
            kwargs.get('func', ''),  # func
            extra_fields  # extra
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log('DEBUG', message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log('INFO', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log('WARNING', message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if exception:
            kwargs['exception_type'] = type(exception).__name__
            kwargs['exception_message'] = str(exception)
        self._log('ERROR', message, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        if exception:
            kwargs['exception_type'] = type(exception).__name__
            kwargs['exception_message'] = str(exception)
        self._log('CRITICAL', message, **kwargs)

    def log_inference(self, model_name: str, input_shape: tuple,
                      prediction: Any, latency_ms: float, **kwargs):
        """Log ML inference"""
        self.info(
            f"Inference completed",
            event_type='ml_inference',
            model_name=model_name,
            input_shape=str(input_shape),
            prediction=str(prediction),
            latency_ms=round(latency_ms, 2),
            **kwargs
        )

    def log_kafka_message(self, topic: str, action: str, message_count: int = 1,
                          partition: Optional[int] = None, offset: Optional[int] = None,
                          **kwargs):
        """Log Kafka message event"""
        self.info(
            f"Kafka {action}: {message_count} message(s)",
            event_type='kafka_message',
            topic=topic,
            action=action,
            message_count=message_count,
            partition=partition,
            offset=offset,
            **kwargs
        )

    def log_pipeline_stage(self, pipeline_name: str, stage: str, status: str,
                           duration_seconds: Optional[float] = None,
                           records_processed: Optional[int] = None, **kwargs):
        """Log pipeline stage execution"""
        self.info(
            f"Pipeline stage {status}",
            event_type='pipeline_stage',
            pipeline_name=pipeline_name,
            stage=stage,
            status=status,
            duration_seconds=duration_seconds,
            records_processed=records_processed,
            **kwargs
        )

    def log_data_quality(self, pipeline_name: str, metric_name: str,
                         metric_value: float, threshold: Optional[float] = None,
                         passed: Optional[bool] = None, **kwargs):
        """Log data quality metric"""
        self.info(
            f"Data quality check: {metric_name}",
            event_type='data_quality',
            pipeline_name=pipeline_name,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            passed=passed,
            **kwargs
        )

    def log_model_training(self, model_name: str, status: str,
                           duration_seconds: Optional[float] = None,
                           metrics: Optional[Dict[str, float]] = None, **kwargs):
        """Log model training event"""
        log_data = {
            'event_type': 'model_training',
            'model_name': model_name,
            'status': status,
            'duration_seconds': duration_seconds,
        }
        if metrics:
            log_data['metrics'] = metrics
        log_data.update(kwargs)

        self.info(f"Model training {status}", **log_data)

    def log_api_request(self, method: str, endpoint: str, status_code: int,
                        duration_ms: float, client_ip: Optional[str] = None,
                        user_id: Optional[str] = None, **kwargs):
        """Log API request"""
        self.info(
            f"{method} {endpoint} - {status_code}",
            event_type='api_request',
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            client_ip=client_ip,
            user_id=user_id,
            **kwargs
        )

    def log_system_health(self, service_name: str, status: str,
                          memory_mb: Optional[float] = None,
                          cpu_percent: Optional[float] = None, **kwargs):
        """Log system health metrics"""
        self.info(
            f"System health: {status}",
            event_type='system_health',
            service_name=service_name,
            status=status,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            **kwargs
        )


def get_logger(service_name: str, environment: str = 'development',
               log_level: str = 'INFO', log_file: Optional[str] = None) -> StructuredLogger:
    """Get or create a structured logger instance"""
    return StructuredLogger(service_name, environment, log_level, log_file)


# Example usage and testing
if __name__ == '__main__':
    # Create logger
    logger = get_logger(
        'test-service', environment='development', log_level='DEBUG')

    # Test different log types
    logger.debug("Debug message", user_id='test123', request_id='req-456')
    logger.info("Service started successfully", port=8080, version='1.0.0')
    logger.warning("High memory usage detected",
                   memory_mb=1500, threshold=1000)

    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("Error occurred during processing",
                     exception=e, context='test')

    # Test specialized log methods
    logger.log_inference(
        model_name='churn_model_v1',
        input_shape=(1, 42),
        prediction=1,
        latency_ms=45.2,
        confidence=0.87
    )

    logger.log_kafka_message(
        topic='customer-data',
        action='produced',
        message_count=100,
        partition=0,
        offset=12345
    )

    logger.log_pipeline_stage(
        pipeline_name='data_pipeline',
        stage='feature_engineering',
        status='completed',
        duration_seconds=12.5,
        records_processed=10000
    )

    logger.log_data_quality(
        pipeline_name='data_pipeline',
        metric_name='missing_values_percent',
        metric_value=2.5,
        threshold=5.0,
        passed=True
    )

    logger.log_model_training(
        model_name='random_forest',
        status='completed',
        duration_seconds=120.5,
        metrics={'accuracy': 0.875, 'f1_score': 0.832, 'roc_auc': 0.891}
    )

    logger.log_system_health(
        service_name='inference-service',
        status='healthy',
        memory_mb=512.3,
        cpu_percent=35.2,
        active_connections=12
    )

    print("\n✓ Structured logging test completed successfully")
