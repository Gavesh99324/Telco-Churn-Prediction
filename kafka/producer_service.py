"""Kafka producer service for customer data streaming"""
import sys
import os
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            logger.info(f"✅ Connected to Kafka: {self.bootstrap_servers}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    def load_customer_data(self) -> Optional[pd.DataFrame]:
        """Load customer data from CSV"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"✅ Loaded {len(df)} customer records from {self.data_path}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
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
        try:
            future = self.producer.send(self.topic, value=record)
            # Wait for confirmation
            result = future.get(timeout=10)
            self.records_sent += 1
            
            if self.records_sent % 100 == 0:
                logger.info(f"📤 Sent {self.records_sent} records to {self.topic}")
            
            return True
        except KafkaError as e:
            logger.error(f"❌ Kafka error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending record: {e}")
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
        
        return sent_count
    
    def run_continuous(self):
        """Run producer in continuous mode (loop through data)"""
        logger.info("="*70)
        logger.info("🚀 STARTING KAFKA PRODUCER (CONTINUOUS MODE)")
        logger.info("="*70)
        
        # Connect to Kafka
        if not self.connect():
            logger.error("Failed to connect to Kafka. Exiting.")
            return
        
        # Load data
        df = self.load_customer_data()
        if df is None:
            logger.error("Failed to load customer data. Exiting.")
            return
        
        logger.info(f"📊 Total records: {len(df)}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        logger.info(f"⏱️  Delay: {self.delay_seconds}s")
        logger.info(f"📡 Topic: {self.topic}")
        logger.info("="*70)
        
        try:
            while True:
                # Loop through all records
                for start_idx in range(0, len(df), self.batch_size):
                    sent = self.produce_batch(df, start_idx)
                    
                    if sent > 0:
                        logger.debug(f"Batch {start_idx//self.batch_size + 1}: {sent} records sent")
                    
                    time.sleep(self.delay_seconds)
                
                logger.info(f"✅ Completed loop. Total sent: {self.records_sent}. Restarting...")
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.close()
    
    def run_once(self):
        """Run producer once (send all data once)"""
        logger.info("="*70)
        logger.info("🚀 STARTING KAFKA PRODUCER (SINGLE RUN)")
        logger.info("="*70)
        
        if not self.connect():
            return
        
        df = self.load_customer_data()
        if df is None:
            return
        
        logger.info(f"📊 Total records to send: {len(df)}")
        
        try:
            for start_idx in range(0, len(df), self.batch_size):
                self.produce_batch(df, start_idx)
                time.sleep(self.delay_seconds)
            
            logger.info(f"✅ Sent all {self.records_sent} records")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close producer connection"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("🔌 Producer closed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka customer data producer')
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
