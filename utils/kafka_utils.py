"""Kafka utilities"""
from kafka import KafkaProducer, KafkaConsumer
import json


class KafkaUtils:
    """Kafka utility functions"""

    @staticmethod
    def create_producer(bootstrap_servers):
        """Create Kafka producer"""
        return KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    @staticmethod
    def create_consumer(topic, bootstrap_servers):
        """Create Kafka consumer"""
        return KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
