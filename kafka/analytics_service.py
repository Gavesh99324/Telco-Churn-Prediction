"""Kafka analytics service"""
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'churn-analytics',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)


def run_analytics():
    """Run analytics aggregation on Kafka messages"""
    for message in consumer:
        data = message.value
        # Run analytics logic
        print(f"Analyzing: {data}")


if __name__ == '__main__':
    run_analytics()
