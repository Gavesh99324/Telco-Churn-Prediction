"""Kafka inference service"""
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'churn-predictions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)


def run_inference():
    """Run real-time inference on Kafka messages"""
    for message in consumer:
        data = message.value
        # Run inference logic
        print(f"Processing: {data}")


if __name__ == '__main__':
    run_inference()
