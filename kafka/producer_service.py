"""Kafka producer service for customer data streaming"""
# fmt: off
# isort: skip_file
import sys
import os
from pathlib import Path

# CRITICAL: Add project root to Python path BEFORE any local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local utils imports (require sys.path to be set first)
from utils.structured_logger import get_logger
from utils.metrics import MetricsCollector, KAFKA_BATCH_SIZE
from utils.health_check import HealthCheckServer, ServiceHealthMonitor

# Third-party imports
from kafka.errors import KafkaError
from kafka import KafkaProducer
from typing import Dict, Any, Optional
import pandas as pd
import time
import json


logger = get_logger(
    'kafka-producer', environment=os.getenv('ENVIRONMENT', 'development'))


class ChurnDataProducer:
    """Produce customer data to Kafka for real-time processing"""

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        topic: str = 'customer-data',
        data_path: str = 'data/raw/telco_churn.csv',
        batch_size: int = 10,
        delay_seconds: float = 1.0
    ):
        """
        Initialize Kafka producer

        Args:
            bootstrap_servers: Kafka broker address
            topic: Topic to produce to
            data_path: Path to customer data CSV
            batch_size: Number of records per batch
            delay_seconds: Delay between batches
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.data_path = data_path
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        self.producer = None
        self.records_sent = 0
        self.health_server = None
        self.health_monitor = None

    def connect(self):
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1,
                compression_type='gzip'
            )
            logger.info("Connected to Kafka",
                        bootstrap_servers=self.bootstrap_servers,
                        topic=self.topic)
            MetricsCollector.set_service_health('kafka-producer', True)
            return True
        except Exception as e:
            logger.error("Failed to connect to Kafka", exception=e,
                         bootstrap_servers=self.bootstrap_servers)
            MetricsCollector.record_error(
                'kafka-producer', 'connection_error', 'critical')
            return False

    def load_customer_data(self) -> Optional[pd.DataFrame]:
        """Load customer data from CSV"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info("Loaded customer data",
                        file_path=self.data_path,
                        record_count=len(df),
                        columns=list(df.columns))
            return df
        except Exception as e:
            logger.error("Failed to load data", exception=e,
                         file_path=self.data_path)
            MetricsCollector.record_error(
                'kafka-producer', 'data_load_error', 'error')
            return None

    def prepare_record(self, row: pd.Series) -> Dict[str, Any]:
        """
        Prepare a customer record for Kafka

        Args:
            row: DataFrame row

        Returns:
            Dictionary with customer data
        """
        record = row.to_dict()

        # Convert numpy types to Python types for JSON serialization
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                record[key] = float(value) if pd.notna(value) else None

        # Add metadata
        record['timestamp'] = time.time()
        record['producer_id'] = 'churn-producer-001'

        return record

    def send_record(self, record: Dict[str, Any]) -> bool:
        """
        Send a record to Kafka

        Args:
            record: Customer record

        Returns:
            True if successful
        """
        start_time = time.time()
        try:
            future = self.producer.send(self.topic, value=record)
            # Wait for confirmation
            result = future.get(timeout=10)
            self.records_sent += 1

            # Record metrics
            MetricsCollector.record_kafka_message(self.topic, 'success')

            if self.records_sent % 100 == 0:
                logger.info("Batch milestone reached",
                            records_sent=self.records_sent,
                            topic=self.topic,
                            partition=result.partition,
                            offset=result.offset)

            return True
        except KafkaError as e:
            logger.error("Kafka error sending record",
                         exception=e, topic=self.topic)
            MetricsCollector.record_kafka_message(self.topic, 'error')
            MetricsCollector.record_error(
                'kafka-producer', 'kafka_send_error', 'error')
            return False
        except Exception as e:
            logger.error("Error sending record", exception=e, topic=self.topic)
            MetricsCollector.record_kafka_message(self.topic, 'error')
            MetricsCollector.record_error(
                'kafka-producer', 'send_error', 'error')
            return False

    def produce_batch(self, df: pd.DataFrame, start_idx: int) -> int:
        """
        Produce a batch of records

        Args:
            df: DataFrame with customer data
            start_idx: Starting index

        Returns:
            Number of records sent
        """
        end_idx = min(start_idx + self.batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        sent_count = 0
        for _, row in batch.iterrows():
            record = self.prepare_record(row)
            if self.send_record(record):
                sent_count += 1

        # Record batch size metrics
        KAFKA_BATCH_SIZE.labels(topic=self.topic).observe(sent_count)

        return sent_count

    def run_continuous(self):
        """Run producer in continuous mode (loop through data)"""
        logger.info("Starting Kafka producer in continuous mode",
                    mode='continuous',
                    batch_size=self.batch_size,
                    delay_seconds=self.delay_seconds,
                    topic=self.topic)

        # Start health check server in background
        health_port = int(os.getenv('HEALTH_CHECK_PORT', '8001'))
        self.health_server = HealthCheckServer(
            'kafka-producer', port=health_port)

        # Add custom readiness check
        def check_kafka_connection():
            return self.producer is not None

        self.health_server.add_readiness_check(
            'kafka_connection', check_kafka_connection)
        self.health_server.start_background()
        logger.info(f"Health check server started", port=health_port)

        # Start health monitoring
        self.health_monitor = ServiceHealthMonitor(
            'kafka-producer', check_interval=30)
        self.health_monitor.start()

        # Connect to Kafka
        if not self.connect():
            logger.error("Failed to connect to Kafka. Exiting.")
            return

        # Set service as ready
        self.health_server.set_ready(True)

        # Load data
        df = self.load_customer_data()
        if df is None:
            logger.error("Failed to load customer data. Exiting.")
            return

        logger.info("Producer ready to send data",
                    total_records=len(df),
                    batch_size=self.batch_size,
                    delay_seconds=self.delay_seconds,
                    topic=self.topic)

        try:
            while True:
                # Loop through all records
                for start_idx in range(0, len(df), self.batch_size):
                    sent = self.produce_batch(df, start_idx)

                    if sent > 0:
                        logger.debug("Batch sent",
                                     batch_number=start_idx//self.batch_size + 1,
                                     records_sent=sent)

                    time.sleep(self.delay_seconds)

                logger.info("Completed data loop, restarting",
                            total_records_sent=self.records_sent)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        except Exception as e:
            logger.error("Unexpected error in producer loop", exception=e)
            MetricsCollector.record_error(
                'kafka-producer', 'runtime_error', 'critical')
        finally:
            self.close()

    def run_once(self):
        """Run producer once (send all data once)"""
        logger.info("Starting Kafka producer in single-run mode",
                    mode='once',
                    batch_size=self.batch_size,
                    topic=self.topic)

        if not self.connect():
            return

        df = self.load_customer_data()
        if df is None:
            return

        logger.info("Sending all records once", total_records=len(df))

        try:
            for start_idx in range(0, len(df), self.batch_size):
                self.produce_batch(df, start_idx)
                time.sleep(self.delay_seconds)

            logger.info("Completed sending all records",
                        total_records_sent=self.records_sent)
        except Exception as e:
            logger.error("Error during single run", exception=e)
            MetricsCollector.record_error(
                'kafka-producer', 'runtime_error', 'error')
        finally:
            self.close()

    def close(self):
        """Close producer connection"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed", records_sent=self.records_sent)
            MetricsCollector.set_service_health('kafka-producer', False)

        if self.health_monitor:
            self.health_monitor.stop()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Kafka customer data producer')
    parser.add_argument('--mode', choices=['continuous', 'once'], default='continuous',
                        help='Run mode: continuous (loop) or once (single pass)')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='customer-data',
                        help='Kafka topic')
    parser.add_argument('--data-path', default='data/raw/telco_churn.csv',
                        help='Path to customer data CSV')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between batches (seconds)')

    args = parser.parse_args()

    producer = ChurnDataProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        data_path=args.data_path,
        batch_size=args.batch_size,
        delay_seconds=args.delay
    )

    if args.mode == 'continuous':
        producer.run_continuous()
    else:
        producer.run_once()
