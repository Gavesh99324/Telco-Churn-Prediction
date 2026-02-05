"""Kafka producer service"""
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def produce_data():
    """Produce sample data to Kafka"""
    while True:
        data = {'message': 'Sample churn data'}
        producer.send('churn-predictions', data)
        time.sleep(1)


if __name__ == '__main__':
    produce_data()
