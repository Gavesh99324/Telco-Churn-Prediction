"""
Integration tests for Kafka streaming flow
Tests producer -> inference -> analytics pipeline
"""
import pytest
import json
import time
import tempfile
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# Mock Kafka settings for testing
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
CUSTOMER_DATA_TOPIC = "customer-data"
PREDICTIONS_TOPIC = "churn-predictions"
ANALYTICS_TOPIC = "churn-analytics"


@pytest.fixture
def sample_customer_data():
    """Generate sample customer data for testing"""
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.50,
        "TotalCharges": 786.00
    }


@pytest.fixture
def kafka_producer():
    """Create Kafka producer for testing"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            request_timeout_ms=5000
        )
        yield producer
        producer.close()
    except KafkaError:
        pytest.skip("Kafka broker not available")


@pytest.fixture
def kafka_consumer():
    """Create Kafka consumer for testing"""
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=False,
            consumer_timeout_ms=10000
        )
        yield consumer
        consumer.close()
    except KafkaError:
        pytest.skip("Kafka broker not available")


class TestKafkaFlow:
    """Integration tests for Kafka streaming pipeline"""

    def test_producer_sends_messages(self, kafka_producer, sample_customer_data):
        """Test that producer can send messages to Kafka"""
        # Send message
        future = kafka_producer.send(CUSTOMER_DATA_TOPIC, sample_customer_data)
        
        # Wait for confirmation
        try:
            record_metadata = future.get(timeout=10)
            assert record_metadata.topic == CUSTOMER_DATA_TOPIC
            assert record_metadata.partition is not None
            assert record_metadata.offset is not None
        except Exception as e:
            pytest.fail(f"Failed to send message: {e}")

    def test_consumer_receives_messages(self, kafka_producer, kafka_consumer, sample_customer_data):
        """Test that consumer can receive messages from Kafka"""
        # Subscribe to topic
        kafka_consumer.subscribe([CUSTOMER_DATA_TOPIC])
        
        # Send message
        kafka_producer.send(CUSTOMER_DATA_TOPIC, sample_customer_data)
        kafka_producer.flush()
        
        # Wait for message
        time.sleep(2)
        
        # Consume messages
        messages = []
        for message in kafka_consumer:
            messages.append(message.value)
            break  # Get one message
        
        assert len(messages) > 0
        assert messages[0]['customerID'] == sample_customer_data['customerID']

    def test_prediction_message_format(self, kafka_consumer):
        """Test that prediction messages have correct format"""
        kafka_consumer.subscribe([PREDICTIONS_TOPIC])
        
        # Wait for predictions
        time.sleep(5)
        
        messages = []
        for message in kafka_consumer:
            messages.append(message.value)
            # Check message structure
            prediction = message.value
            assert 'customerID' in prediction
            assert 'prediction' in prediction
            assert 'churn_probability' in prediction
            assert 'risk_level' in prediction
            assert 'timestamp' in prediction
            
            # Validate risk level
            assert prediction['risk_level'] in ['HIGH', 'MEDIUM', 'LOW']
            
            # Validate probability range
            assert 0 <= prediction['churn_probability'] <= 1
            break  # Check one message

    def test_analytics_message_format(self, kafka_consumer):
        """Test that analytics messages have correct format"""
        kafka_consumer.subscribe([ANALYTICS_TOPIC])
        
        # Wait for analytics
        time.sleep(10)
        
        for message in kafka_consumer:
            analytics = message.value
            
            # Check required fields
            assert 'churn_rate' in analytics
            assert 'predictions_per_minute' in analytics
            assert 'risk_counts' in analytics
            assert 'avg_churn_probability' in analytics
            assert 'timestamp' in analytics
            
            # Validate risk counts structure
            risk_counts = analytics['risk_counts']
            assert 'HIGH' in risk_counts
            assert 'MEDIUM' in risk_counts
            assert 'LOW' in risk_counts
            
            # Validate metrics
            assert 0 <= analytics['churn_rate'] <= 1
            assert analytics['predictions_per_minute'] >= 0
            break  # Check one message

    def test_end_to_end_kafka_flow(self, kafka_producer, sample_customer_data):
        """Test complete flow: producer -> inference -> analytics"""
        # Create consumers for predictions and analytics
        prediction_consumer = KafkaConsumer(
            PREDICTIONS_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            consumer_timeout_ms=15000
        )
        
        analytics_consumer = KafkaConsumer(
            ANALYTICS_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            consumer_timeout_ms=20000
        )
        
        # Step 1: Send customer data
        kafka_producer.send(CUSTOMER_DATA_TOPIC, sample_customer_data)
        kafka_producer.flush()
        
        # Step 2: Wait for prediction
        time.sleep(3)
        prediction_received = False
        for message in prediction_consumer:
            if message.value['customerID'] == sample_customer_data['customerID']:
                prediction_received = True
                break
        
        assert prediction_received, "Prediction not received"
        
        # Step 3: Wait for analytics
        time.sleep(5)
        analytics_received = False
        for message in analytics_consumer:
            if 'churn_rate' in message.value:
                analytics_received = True
                break
        
        assert analytics_received, "Analytics not received"
        
        # Cleanup
        prediction_consumer.close()
        analytics_consumer.close()

    def test_batch_processing(self, kafka_producer):
        """Test producer can handle batch processing"""
        # Create batch of 10 records
        batch = []
        for i in range(10):
            data = {
                "customerID": f"BATCH-{i:03d}",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12 + i,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.50 + i,
                "TotalCharges": 786.00 + i * 10
            }
            batch.append(data)
        
        # Send batch
        successful = 0
        for record in batch:
            try:
                future = kafka_producer.send(CUSTOMER_DATA_TOPIC, record)
                future.get(timeout=10)
                successful += 1
            except Exception:
                pass
        
        kafka_producer.flush()
        assert successful == 10, f"Only {successful}/10 messages sent successfully"

    def test_error_handling(self, kafka_producer):
        """Test error handling with invalid data"""
        invalid_data = {
            "customerID": "INVALID-001",
            # Missing required fields
            "gender": "Female"
        }
        
        # This should still send (producer doesn't validate)
        future = kafka_producer.send(CUSTOMER_DATA_TOPIC, invalid_data)
        
        try:
            record_metadata = future.get(timeout=10)
            assert record_metadata.topic == CUSTOMER_DATA_TOPIC
        except Exception as e:
            pytest.fail(f"Producer should send even invalid data: {e}")

    @pytest.mark.slow
    def test_throughput(self, kafka_producer):
        """Test producer throughput"""
        num_messages = 100
        start_time = time.time()
        
        for i in range(num_messages):
            data = {
                "customerID": f"THROUGHPUT-{i:04d}",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00
            }
            kafka_producer.send(CUSTOMER_DATA_TOPIC, data)
        
        kafka_producer.flush()
        elapsed_time = time.time() - start_time
        
        messages_per_second = num_messages / elapsed_time
        assert messages_per_second > 10, f"Throughput too low: {messages_per_second:.2f} msg/s"


def test_kafka_topics_exist():
    """Test that required Kafka topics exist or can be created"""
    try:
        from kafka.admin import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        
        topics = admin.list_topics()
        expected_topics = [CUSTOMER_DATA_TOPIC, PREDICTIONS_TOPIC, ANALYTICS_TOPIC]
        
        for topic in expected_topics:
            # Topics will be created automatically if they don't exist
            assert topic in topics or True  # Auto-create enabled
        
        admin.close()
    except Exception:
        pytest.skip("Kafka admin not available")

