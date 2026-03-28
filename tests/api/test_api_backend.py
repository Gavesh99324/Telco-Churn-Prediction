"""API backend tests for happy paths and hardened failure paths."""
import api.main as api_main
import sys
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyModel:
    """Minimal model stub for API prediction tests."""

    feature_names_in_ = ['feature_1']

    def predict(self, data):
        return [1] * len(data)

    def predict_proba(self, data):
        return [[0.1, 0.9] for _ in range(len(data))]


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_main.app)


@pytest.fixture
def valid_customer_payload() -> Dict:
    return {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.7,
        'TotalCharges': 848.4,
    }


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch):
    """Reset mutable global state to deterministic defaults for each test."""
    monkeypatch.setattr(api_main, 'API_AUTH_ENABLED', False)
    monkeypatch.setattr(api_main, 'API_AUTH_TOKEN', '')
    monkeypatch.setattr(api_main, 'MAX_BATCH_SIZE', 200)
    monkeypatch.setattr(api_main, 'MAX_REQUEST_BYTES', 1024 * 1024)

    api_main.model = DummyModel()
    api_main.scaler = None
    api_main.label_encoders = None
    api_main.feature_names = ['feature_1']
    api_main.startup_errors = []
    api_main.dependency_status = {'all_required_available': True}


def test_health_endpoint(client: TestClient):
    response = client.get('/health')

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'healthy'
    assert 'model_loaded' in payload


def test_ready_endpoint_when_ready(client: TestClient):
    response = client.get('/ready')

    assert response.status_code == 200
    assert response.json()['status'] == 'ready'


def test_ready_endpoint_when_not_ready(client: TestClient):
    api_main.model = None
    api_main.startup_errors = ['Failed to load model artifacts']
    api_main.dependency_status = {'all_required_available': False}

    response = client.get('/ready')

    assert response.status_code == 503
    payload = response.json()
    assert payload['error_code'] == 'SERVICE_NOT_READY'
    assert 'details' in payload


def test_model_info_requires_loaded_model(client: TestClient):
    api_main.model = None

    response = client.get('/model/info')

    assert response.status_code == 503
    assert response.json()['error_code'] == 'MODEL_NOT_LOADED'


def test_predict_happy_path(client: TestClient, valid_customer_payload: Dict, monkeypatch):
    monkeypatch.setattr(
        api_main,
        'preprocess_customer_data',
        lambda _: [{'feature_1': 0.42}],
    )

    response = client.post('/predict', json=valid_customer_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload['prediction'] in (0, 1)
    assert 0 <= payload['probability'] <= 1


def test_batch_predict_happy_path(
    client: TestClient,
    valid_customer_payload: Dict,
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        'preprocess_customer_data',
        lambda _: [{'feature_1': 0.42}],
    )

    response = client.post(
        '/batch_predict',
        json={'customers': [valid_customer_payload, valid_customer_payload]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['total_customers'] == 2
    assert len(payload['predictions']) == 2


def test_metrics_endpoint(client: TestClient):
    response = client.get('/metrics')

    assert response.status_code == 200
    assert 'ml_inference_requests_total' in response.text


def test_predict_validation_error(client: TestClient):
    response = client.post('/predict', json={'invalid': 'payload'})

    assert response.status_code == 422
    payload = response.json()
    assert payload['error_code'] == 'VALIDATION_ERROR'


def test_predict_model_not_loaded(client: TestClient, valid_customer_payload: Dict):
    api_main.model = None

    response = client.post('/predict', json=valid_customer_payload)

    assert response.status_code == 503
    assert response.json()['error_code'] == 'MODEL_NOT_LOADED'


def test_predict_schema_mismatch_failure(
    client: TestClient,
    valid_customer_payload: Dict,
    monkeypatch,
):
    def _raise_schema_error(_):
        raise ValueError('schema mismatch')

    monkeypatch.setattr(
        api_main, 'preprocess_customer_data', _raise_schema_error)

    response = client.post('/predict', json=valid_customer_payload)

    assert response.status_code == 500
    assert response.json()['error_code'] == 'PREDICTION_ERROR'


def test_batch_size_safeguard(client: TestClient, valid_customer_payload: Dict, monkeypatch):
    monkeypatch.setattr(api_main, 'MAX_BATCH_SIZE', 1)

    response = client.post(
        '/batch_predict',
        json={'customers': [valid_customer_payload, valid_customer_payload]},
    )

    assert response.status_code == 413
    payload = response.json()
    assert payload['error_code'] == 'BATCH_TOO_LARGE'


def test_request_size_limit_middleware(client: TestClient):
    headers = {'content-length': str(10 * 1024 * 1024)}

    response = client.post('/predict', headers=headers, json={'k': 'v'})

    assert response.status_code == 413
    assert response.json()['error_code'] == 'REQUEST_TOO_LARGE'


def test_auth_hook_blocks_requests_without_key(
    client: TestClient,
    valid_customer_payload: Dict,
):
    api_main.API_AUTH_ENABLED = True
    api_main.API_AUTH_TOKEN = 'test-auth-key'

    response = client.post('/predict', json=valid_customer_payload)

    assert response.status_code == 401
    assert response.json()['error_code'] == 'UNAUTHORIZED'


def test_auth_hook_allows_requests_with_key(
    client: TestClient,
    valid_customer_payload: Dict,
    monkeypatch,
):
    api_main.API_AUTH_ENABLED = True
    api_main.API_AUTH_TOKEN = 'test-auth-key'

    monkeypatch.setattr(
        api_main,
        'preprocess_customer_data',
        lambda _: [{'feature_1': 0.42}],
    )

    response = client.post(
        '/predict',
        json=valid_customer_payload,
        headers={'X-API-Key': 'test-auth-key'},
    )

    assert response.status_code == 200


def test_startup_dependency_validation_detects_missing_required_artifact(monkeypatch):
    base = Path('does-not-exist')

    def _fake_paths():
        return {
            'model': base / 'best_model.pkl',
            'scaler': base / 'scaler.pkl',
            'encoders': base / 'label_encoders.pkl',
            'features': base / 'feature_names.pkl',
        }

    monkeypatch.setattr(api_main, 'get_artifact_paths', _fake_paths)

    status_report = api_main.validate_startup_dependencies()

    assert status_report['all_required_available'] is False
    assert status_report['model']['required'] is True
    assert status_report['model']['exists'] is False
