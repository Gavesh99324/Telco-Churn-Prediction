"""Kafka inference service for real-time churn predictions"""
import sys
import os
import json
import time
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_inference import ModelInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnInferenceService:
    """Real-time churn prediction service using Kafka"""
    
    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        input_topic: str = 'customer-data',
        output_topic: str = 'churn-predictions',
        model_path: str = 'artifacts/models/best_model.pkl',
        data_dir: str = 'artifacts/data',
        consumer_group: str = 'churn-inference-group'
    ):
        """
        Initialize inference service
        
        Args:
            bootstrap_servers: Kafka broker address
            input_topic: Topic to consume customer data from
            output_topic: Topic to produce predictions to
            model_path: Path to trained model
            data_dir: Directory with preprocessors
            consumer_group: Kafka consumer group
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.model_path = model_path
        self.data_dir = data_dir
        self.consumer_group = consumer_group
        
        self.consumer = None
        self.producer = None
        self.inference = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
        self.predictions_made = 0
        self.errors = 0
    
    def load_model_and_preprocessors(self):
        """Load trained model and preprocessing artifacts"""
        try:
            # Initialize inference engine
            self.inference = ModelInference()
            
            # Load model
            logger.info(f"Loading model from {self.model_path}...")
            self.inference.load_model(self.model_path)
            logger.info("✅ Model loaded")
            
            # Load scaler
            with open(f'{self.data_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("✅ Scaler loaded")
            
            # Load label encoders
            with open(f'{self.data_dir}/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            logger.info("✅ Label encoders loaded")
            
            # Load feature names
            with open(f'{self.data_dir}/feature_names.pkl', 'rb') as f:
                feature_data = pickle.load(f)
                self.feature_names = feature_data['feature_names']
            logger.info(f"✅ Feature names loaded ({len(self.feature_names)} features)")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model/preprocessors: {e}")
            return False
    
    def connect_kafka(self):
        """Connect to Kafka as consumer and producer"""
        try:
            # Create consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',  # Start from latest
                enable_auto_commit=True,
                max_poll_records=10
            )
            logger.info(f"✅ Consumer connected to {self.input_topic}")
            
            # Create producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"✅ Producer connected to {self.output_topic}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    def preprocess_record(self, record: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Preprocess a customer record for inference
        
        Args:
            record: Raw customer data
            
        Returns:
            Preprocessed DataFrame ready for model
        """
        try:
            # Remove metadata fields
            customer_data = {k: v for k, v in record.items() 
                           if k not in ['timestamp', 'producer_id']}
            
            # Create DataFrame
            df = pd.DataFrame([customer_data])
            
            # Note: This assumes data is already preprocessed through data pipeline
            # In production, you'd need to apply all preprocessing steps here
            
            # Ensure correct column order
            if set(df.columns) != set(self.feature_names):
                logger.warning(f"Feature mismatch. Expected {len(self.feature_names)}, got {len(df.columns)}")
                # Try to reorder
                try:
                    df = df[self.feature_names]
                except KeyError as e:
                    logger.error(f"Missing features: {e}")
                    return None
            
            return df
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {e}")
            return None
    
    def make_prediction(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Make churn prediction
        
        Args:
            df: Preprocessed customer data
            
        Returns:
            Prediction result
        """
        try:
            # Get prediction and probability
            prediction = self.inference.predict(df)[0]
            probabilities = self.inference.predict_proba(df)[0]
            
            result = {
                'churn_prediction': int(prediction),
                'churn_probability': float(probabilities[1]),
                'no_churn_probability': float(probabilities[0]),
                'risk_level': self._get_risk_level(probabilities[1]),
                'timestamp': time.time(),
                'model_version': 'best_model_v1'
            }
            
            return result
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None
    
    def _get_risk_level(self, churn_prob: float) -> str:
        """Categorize prediction into risk level"""
        if churn_prob >= 0.7:
            return 'HIGH'
        elif churn_prob >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def send_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Send prediction to output topic
        
        Args:
            prediction: Prediction result
            
        Returns:
            True if successful
        """
        try:
            future = self.producer.send(self.output_topic, value=prediction)
            future.get(timeout=10)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send prediction: {e}")
            return False
    
    def process_message(self, message):
        """
        Process a single Kafka message
        
        Args:
            message: Kafka message
        """
        try:
            # Get customer data
            customer_data = message.value
            
            # Preprocess
            df = self.preprocess_record(customer_data)
            if df is None:
                self.errors += 1
                return
            
            # Make prediction
            prediction = self.make_prediction(df)
            if prediction is None:
                self.errors += 1
                return
            
            # Add customer identifier if available
            if 'customerID' in customer_data:
                prediction['customerID'] = customer_data['customerID']
            
            # Send prediction
            if self.send_prediction(prediction):
                self.predictions_made += 1
                
                if self.predictions_made % 50 == 0:
                    logger.info(f"📊 Predictions: {self.predictions_made} | Errors: {self.errors} | "
                              f"Success rate: {self.predictions_made/(self.predictions_made+self.errors)*100:.1f}%")
            else:
                self.errors += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            self.errors += 1
    
    def run(self):
        """Run inference service"""
        logger.info("="*70)
        logger.info("🚀 STARTING KAFKA INFERENCE SERVICE")
        logger.info("="*70)
        
        # Load model and preprocessors
        if not self.load_model_and_preprocessors():
            logger.error("Failed to load model. Exiting.")
            return
        
        # Connect to Kafka
        if not self.connect_kafka():
            logger.error("Failed to connect to Kafka. Exiting.")
            return
        
        logger.info(f"📥 Consuming from: {self.input_topic}")
        logger.info(f"📤 Producing to: {self.output_topic}")
        logger.info(f"👥 Consumer group: {self.consumer_group}")
        logger.info("="*70)
        logger.info("⏳ Waiting for messages...")
        
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
        """Close Kafka connections"""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        logger.info("="*70)
        logger.info("📊 FINAL STATISTICS")
        logger.info(f"   Total predictions: {self.predictions_made}")
        logger.info(f"   Total errors: {self.errors}")
        if self.predictions_made + self.errors > 0:
            success_rate = self.predictions_made / (self.predictions_made + self.errors) * 100
            logger.info(f"   Success rate: {success_rate:.2f}%")
        logger.info("="*70)
        logger.info("🔌 Service closed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka inference service')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', default='customer-data',
                        help='Input topic (customer data)')
    parser.add_argument('--output-topic', default='churn-predictions',
                        help='Output topic (predictions)')
    parser.add_argument('--model-path', default='artifacts/models/best_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--data-dir', default='artifacts/data',
                        help='Directory with preprocessors')
    parser.add_argument('--consumer-group', default='churn-inference-group',
                        help='Kafka consumer group')
    
    args = parser.parse_args()
    
    service = ChurnInferenceService(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        model_path=args.model_path,
        data_dir=args.data_dir,
        consumer_group=args.consumer_group
    )
    
    service.run()
