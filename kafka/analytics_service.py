"""Kafka analytics service for real-time churn metrics"""
import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db_manager import DBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnAnalyticsService:
    """Real-time analytics and aggregation service"""
    
    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        input_topic: str = 'churn-predictions',
        output_topic: str = 'churn-analytics',
        consumer_group: str = 'churn-analytics-group',
        window_size_minutes: int = 60,
        db_path: str = 'churn_analytics.db'
    ):
        """
        Initialize analytics service
        
        Args:
            bootstrap_servers: Kafka broker address
            input_topic: Topic to consume predictions from
            output_topic: Topic to produce analytics to
            consumer_group: Kafka consumer group
            window_size_minutes: Time window for metrics (minutes)
            db_path: SQLite database path
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer_group = consumer_group
        self.window_size = timedelta(minutes=window_size_minutes)
        self.db_path = db_path
        
        self.consumer = None
        self.producer = None
        self.db = None
        
        # Time-windowed buffers
        self.predictions_buffer = deque()
        
        # Aggregated metrics
        self.metrics = {
            'total_predictions': 0,
            'churn_count': 0,
            'no_churn_count': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'avg_churn_probability': 0.0,
            'predictions_per_minute': 0.0
        }
        
        self.last_report_time = time.time()
        self.report_interval = 60  # Report every 60 seconds
    
    def connect_kafka(self):
        """Connect to Kafka"""
        try:
            # Consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info(f"✅ Consumer connected to {self.input_topic}")
            
            # Producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info(f"✅ Producer connected to {self.output_topic}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    def connect_database(self):
        """Initialize database connection"""
        try:
            self.db = DBManager(db_type='sqlite', db_name=self.db_path)
            self.db.connect()
            
            # Create tables if they don't exist
            self._create_tables()
            
            logger.info(f"✅ Database connected: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def _create_tables(self):
        """Create database tables"""
        try:
            # Predictions table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS churn_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT,
                    churn_prediction INTEGER,
                    churn_probability REAL,
                    risk_level TEXT,
                    timestamp REAL,
                    model_version TEXT
                )
            """)
            
            # Analytics table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS churn_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp REAL
                )
            """)
            
            self.db.conn.commit()
            logger.info("✅ Database tables ready")
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
    
    def save_prediction(self, prediction: Dict[str, Any]):
        """Save prediction to database"""
        try:
            query = """
                INSERT INTO churn_predictions 
                (customer_id, churn_prediction, churn_probability, risk_level, timestamp, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            cursor = self.db.conn.cursor()
            cursor.execute(query, (
                prediction.get('customerID', 'unknown'),
                prediction['churn_prediction'],
                prediction['churn_probability'],
                prediction['risk_level'],
                prediction['timestamp'],
                prediction.get('model_version', 'unknown')
            ))
            self.db.conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save prediction: {e}")
    
    def save_metrics(self, metrics: Dict[str, float]):
        """Save metrics to database"""
        try:
            timestamp = time.time()
            cursor = self.db.conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                query = """
                    INSERT INTO churn_analytics (metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?)
                """
                cursor.execute(query, (metric_name, metric_value, timestamp))
            
            self.db.conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save metrics: {e}")
    
    def update_windowed_metrics(self, prediction: Dict[str, Any]):
        """Update metrics based on time window"""
        current_time = datetime.fromtimestamp(prediction['timestamp'])
        
        # Add to buffer
        self.predictions_buffer.append(prediction)
        
        # Remove old predictions outside window
        cutoff_time = current_time - self.window_size
        while self.predictions_buffer and \
              datetime.fromtimestamp(self.predictions_buffer[0]['timestamp']) < cutoff_time:
            self.predictions_buffer.popleft()
        
        # Calculate metrics from window
        if len(self.predictions_buffer) > 0:
            churn_count = sum(1 for p in self.predictions_buffer if p['churn_prediction'] == 1)
            total = len(self.predictions_buffer)
            
            risk_counts = defaultdict(int)
            prob_sum = 0.0
            
            for p in self.predictions_buffer:
                risk_counts[p['risk_level']] += 1
                prob_sum += p['churn_probability']
            
            self.metrics['total_predictions'] = total
            self.metrics['churn_count'] = churn_count
            self.metrics['no_churn_count'] = total - churn_count
            self.metrics['churn_rate'] = churn_count / total if total > 0 else 0.0
            self.metrics['high_risk_count'] = risk_counts['HIGH']
            self.metrics['medium_risk_count'] = risk_counts['MEDIUM']
            self.metrics['low_risk_count'] = risk_counts['LOW']
            self.metrics['avg_churn_probability'] = prob_sum / total if total > 0 else 0.0
            
            # Calculate predictions per minute
            window_minutes = self.window_size.total_seconds() / 60
            self.metrics['predictions_per_minute'] = total / window_minutes if window_minutes > 0 else 0.0
    
    def publish_metrics(self):
        """Publish metrics to Kafka"""
        try:
            metrics_message = {
                'timestamp': time.time(),
                'window_minutes': self.window_size.total_seconds() / 60,
                'metrics': self.metrics.copy()
            }
            
            self.producer.send(self.output_topic, value=metrics_message)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to publish metrics: {e}")
            return False
    
    def log_metrics(self):
        """Log metrics to console"""
        logger.info("="*70)
        logger.info("📊 REAL-TIME ANALYTICS")
        logger.info("="*70)
        logger.info(f"Window: {self.window_size.total_seconds()/60:.0f} minutes")
        logger.info(f"Total Predictions: {self.metrics['total_predictions']}")
        logger.info(f"Churn Rate: {self.metrics['churn_rate']:.2%}")
        logger.info(f"  - Will Churn: {self.metrics['churn_count']}")
        logger.info(f"  - Won't Churn: {self.metrics['no_churn_count']}")
        logger.info(f"Risk Distribution:")
        logger.info(f"  - HIGH: {self.metrics['high_risk_count']}")
        logger.info(f"  - MEDIUM: {self.metrics['medium_risk_count']}")
        logger.info(f"  - LOW: {self.metrics['low_risk_count']}")
        logger.info(f"Avg Churn Probability: {self.metrics['avg_churn_probability']:.2%}")
        logger.info(f"Predictions/Minute: {self.metrics['predictions_per_minute']:.1f}")
        logger.info("="*70)
    
    def process_message(self, message):
        """Process a prediction message"""
        try:
            prediction = message.value
            
            # Save to database
            self.save_prediction(prediction)
            
            # Update windowed metrics
            self.update_windowed_metrics(prediction)
            
            # Periodic reporting
            current_time = time.time()
            if current_time - self.last_report_time >= self.report_interval:
                self.log_metrics()
                self.save_metrics(self.metrics)
                self.publish_metrics()
                self.last_report_time = current_time
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def run(self):
        """Run analytics service"""
        logger.info("="*70)
        logger.info("🚀 STARTING KAFKA ANALYTICS SERVICE")
        logger.info("="*70)
        
        # Connect to Kafka
        if not self.connect_kafka():
            logger.error("Failed to connect to Kafka. Exiting.")
            return
        
        # Connect to database
        if not self.connect_database():
            logger.error("Failed to connect to database. Exiting.")
            return
        
        logger.info(f"📥 Consuming from: {self.input_topic}")
        logger.info(f"📤 Publishing to: {self.output_topic}")
        logger.info(f"💾 Database: {self.db_path}")
        logger.info(f"⏱️  Window: {self.window_size.total_seconds()/60:.0f} minutes")
        logger.info(f"📊 Report interval: {self.report_interval}s")
        logger.info("="*70)
        logger.info("⏳ Waiting for predictions...")
        
        try:
            for message in self.consumer:
                self.process_message(message)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close connections"""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()
        if self.db:
            self.db.close()
        
        logger.info("🔌 Service closed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka analytics service')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='churn-predictions',
                        help='Input topic (predictions)')
    parser.add_argument('--output-topic', default='churn-analytics',
                        help='Output topic (analytics)')
    parser.add_argument('--consumer-group', default='churn-analytics-group',
                        help='Kafka consumer group')
    parser.add_argument('--window', type=int, default=60,
                        help='Time window in minutes')
    parser.add_argument('--db-path', default='churn_analytics.db',
                        help='SQLite database path')
    
    args = parser.parse_args()
    
    service = ChurnAnalyticsService(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        consumer_group=args.consumer_group,
        window_size_minutes=args.window,
        db_path=args.db_path
    )
    
    service.run()
