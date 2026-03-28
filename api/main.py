"""FastAPI REST API for churn prediction model serving."""
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled in runtime checks
    pd = None
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from utils.config import load_config
from utils.metrics import MetricsCollector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=(
        '{"timestamp": "%(asctime)s", '
        '"level": "%(levelname)s", '
        '"message": "%(message)s"}'
    ),
)
logger = logging.getLogger(__name__)


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'on'}


APP_ENV = os.getenv('ENVIRONMENT', 'development').lower()
LOCAL_ENVIRONMENTS = {'development', 'dev', 'local', 'test'}
IS_NON_LOCAL = APP_ENV not in LOCAL_ENVIRONMENTS

CORS_ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv(
        'CORS_ALLOWED_ORIGINS',
        'http://localhost:3000,http://localhost:3001'
    ).split(',') if origin.strip()
]

API_AUTH_ENABLED = _to_bool(
    os.getenv('API_AUTH_ENABLED'),
    default=IS_NON_LOCAL
)
API_AUTH_TOKEN = os.getenv('API_AUTH_TOKEN', '')

MAX_REQUEST_BYTES = int(os.getenv('API_MAX_REQUEST_BYTES', '1048576'))
MAX_BATCH_SIZE = max(1, min(1000, int(os.getenv('API_MAX_BATCH_SIZE', '200'))))

# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="REST API for customer churn prediction using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=["*"],
)


# Pydantic models for request/response
class CustomerData(BaseModel):
    """Customer data for churn prediction"""
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Number of months")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(...,
                                description="Yes, No, or No internet service")
    OnlineBackup: str = Field(...,
                              description="Yes, No, or No internet service")
    DeviceProtection: str = Field(...,
                                  description="Yes, No, or No internet service")
    TechSupport: str = Field(...,
                             description="Yes, No, or No internet service")
    StreamingTV: str = Field(...,
                             description="Yes, No, or No internet service")
    StreamingMovies: str = Field(...,
                                 description="Yes, No, or No internet service")
    Contract: str = Field(...,
                          description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method type")
    MonthlyCharges: float = Field(..., gt=0,
                                  description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0,
                                description="Total charges in dollars")

    @validator('TotalCharges', pre=True)
    def validate_total_charges(cls, v):
        """Handle empty or invalid TotalCharges"""
        if v == '' or v is None:
            return 0.0
        return float(v)

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.70,
                "TotalCharges": 848.40,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="0 (No churn) or 1 (Churn)")
    probability: float = Field(..., ge=0, le=1,
                               description="Churn probability")
    confidence: str = Field(..., description="Low, Medium, or High")
    risk_level: str = Field(..., description="Low, Medium, or High")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerData] = Field(..., min_items=1, max_items=1000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_customers: int
    churn_count: int
    churn_percentage: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str
    model_loaded: bool
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model metadata"""
    model_name: str
    model_path: str
    model_loaded: bool
    feature_count: int
    version: str


class ErrorResponse(BaseModel):
    """Standardized API error response."""
    error_code: str
    message: str
    timestamp: str
    details: Optional[Dict] = None


# Global model and artifacts
model = None
scaler = None
label_encoders = None
feature_names = None
model_path = None
start_time = datetime.now()
dependency_status: Dict[str, Dict] = {}
startup_errors: List[str] = []


def _error_payload(
    error_code: str,
    message: str,
    details: Optional[Dict] = None
) -> Dict:
    return {
        'error_code': error_code,
        'message': message,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'details': details,
    }


def raise_api_error(
    error_code: str,
    message: str,
    http_status: int,
    details: Optional[Dict] = None
):
    raise HTTPException(
        status_code=http_status,
        detail=_error_payload(error_code, message, details)
    )


def _dependency_entry(path: Path, required: bool) -> Dict:
    exists = path.exists()
    return {
        'path': str(path),
        'required': required,
        'exists': exists,
    }


def get_artifact_paths() -> Dict[str, Path]:
    config = load_config()
    models_dir = Path(config.get('models_dir', './artifacts/models'))
    data_dir = Path(config.get('artifacts_dir', './artifacts')) / 'data'
    return {
        'model': models_dir / 'best_model.pkl',
        'scaler': data_dir / 'scaler.pkl',
        'encoders': data_dir / 'label_encoders.pkl',
        'features': data_dir / 'feature_names.pkl',
    }


def validate_startup_dependencies() -> Dict[str, Dict]:
    paths = get_artifact_paths()
    dependencies = {
        'model': _dependency_entry(paths['model'], required=True),
        'scaler': _dependency_entry(paths['scaler'], required=False),
        'encoders': _dependency_entry(paths['encoders'], required=False),
        'features': _dependency_entry(paths['features'], required=False),
    }
    dependencies['all_required_available'] = all(
        dep['exists'] for dep in dependencies.values()
        if isinstance(dep, dict) and dep.get('required')
    )
    return dependencies


def require_auth(x_api_key: Optional[str] = Header(None, alias='X-API-Key')):
    """Gateway-ready auth hook for non-local deployments."""
    if not API_AUTH_ENABLED:
        return

    if not API_AUTH_TOKEN:
        raise_api_error(
            'AUTH_CONFIG_ERROR',
            'API authentication is enabled but API_AUTH_TOKEN is not set',
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if x_api_key != API_AUTH_TOKEN:
        raise_api_error(
            'UNAUTHORIZED',
            'Invalid or missing API key',
            status.HTTP_401_UNAUTHORIZED,
        )


@app.middleware('http')
async def request_size_middleware(request: Request, call_next):
    """Protect API from oversized request payloads."""
    if request.method in {'POST', 'PUT', 'PATCH'}:
        content_length = request.headers.get('content-length')
        if content_length is not None:
            try:
                content_length_int = int(content_length)
            except ValueError:
                content_length_int = 0

            if content_length_int > MAX_REQUEST_BYTES:
                payload = _error_payload(
                    'REQUEST_TOO_LARGE',
                    (
                        'Request exceeds max allowed size '
                        f'({MAX_REQUEST_BYTES} bytes)'
                    ),
                    details={'max_request_bytes': MAX_REQUEST_BYTES}
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=payload
                )
    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and 'error_code' in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload('HTTP_ERROR', str(detail))
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_payload(
            'VALIDATION_ERROR',
            'Request validation failed',
            details={'errors': exc.errors()}
        )
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    logger.exception('Unhandled API error: %s', exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload(
            'INTERNAL_SERVER_ERROR',
            'An unexpected error occurred',
        )
    )


def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, label_encoders, feature_names, model_path

    try:
        artifact_paths = get_artifact_paths()

        # Load model
        model_path = artifact_paths['model']
        if not model_path.exists():
            logger.error('Model not found at %s', model_path)
            return False

        model = joblib.load(model_path)
        logger.info('Model loaded from %s', model_path)

        # Load preprocessing artifacts
        scaler_path = artifact_paths['scaler']
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info('Scaler loaded from %s', scaler_path)

        encoders_path = artifact_paths['encoders']
        if encoders_path.exists():
            label_encoders = joblib.load(encoders_path)
            logger.info('Label encoders loaded from %s', encoders_path)

        features_path = artifact_paths['features']
        if features_path.exists():
            with open(features_path, 'rb') as f:
                loaded_features = pickle.load(f)
                # Handle both dict and list formats
                if isinstance(loaded_features, dict):
                    feature_names = loaded_features.get('feature_names', [])
                elif isinstance(loaded_features, list):
                    feature_names = loaded_features
                else:
                    feature_names = list(loaded_features)
            logger.info(
                'Feature names loaded: %s features',
                len(feature_names)
            )

        if not feature_names and hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            logger.info(
                'Feature names inferred from model: %s',
                len(feature_names)
            )

        logger.info('All model artifacts loaded successfully')
        return True

    except Exception as e:
        logger.error('Error loading model artifacts: %s', str(e))
        return False


def preprocess_customer_data(customer: CustomerData) -> 'pd.DataFrame':
    """Preprocess customer data for prediction"""
    try:
        if pd is None:
            raise_api_error(
                'DEPENDENCY_MISSING',
                'pandas is required for preprocessing but is not installed',
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                details={'dependency': 'pandas'}
            )

        # Convert to DataFrame
        data = pd.DataFrame([customer.dict()])

        # Handle TotalCharges
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(
                data['TotalCharges'], errors='coerce')
            data['TotalCharges'] = data['TotalCharges'].fillna(
                data['MonthlyCharges'])

        # Feature engineering (match training pipeline EXACTLY)
        # 1. Average charge per tenure
        data['avg_charge_per_tenure'] = data['TotalCharges'] / \
            (data['tenure'] + 1)

        # 2. Service count (ALL services)
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        data['service_count'] = 0
        for col in service_cols:
            if col in data.columns:
                data['service_count'] += (data[col] == 'Yes').astype(int)

        # 3. Has phone
        data['has_phone'] = (data['PhoneService'] == 'Yes').astype(int)

        # 4. Has internet
        data['has_internet'] = (data['InternetService'] != 'No').astype(int)

        # 5. Premium services count (security, backup, protection)
        premium_services = ['OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection']
        data['premium_services_count'] = 0
        for col in premium_services:
            if col in data.columns:
                data['premium_services_count'] += (data[col]
                                                   == 'Yes').astype(int)

        # 6. Streaming services count
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        data['streaming_services_count'] = 0
        for col in streaming_cols:
            if col in data.columns:
                data['streaming_services_count'] += (
                    data[col] == 'Yes').astype(int)

        # 7. Senior citizen with dependents
        data['senior_with_dependents'] = (
            (data['SeniorCitizen'] == 1) & (data['Dependents'] == 'Yes')
        ).astype(int)

        # 8. Monthly to total charges ratio
        data['monthly_to_total_ratio'] = data['MonthlyCharges'] / \
            (data['TotalCharges'] + 1)

        # 9. Contract encoding (ordinal)
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        data['contract_encoded'] = data['Contract'].map(contract_map)

        # 10. Tenure group (categorical bins)
        def assign_tenure_group(tenure):
            if tenure <= 12:
                return '0-1 year'
            elif tenure <= 24:
                return '1-2 years'
            elif tenure <= 48:
                return '2-4 years'
            else:
                return '4+ years'

        data['tenure_group'] = data['tenure'].apply(assign_tenure_group)

        # Apply label encoding for binary features
        binary_cols = ['gender', 'Partner', 'Dependents',
                       'PhoneService', 'PaperlessBilling']
        if label_encoders:
            for col in binary_cols:
                if col in data.columns and col in label_encoders:
                    le = label_encoders[col]
                    data[col] = data[col].map(lambda x: le.transform([x])[
                                              0] if x in le.classes_ else 0)

        # One-hot encoding for categorical features (including tenure_group)
        categorical_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaymentMethod', 'tenure_group'
        ]

        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Build expected schema from saved artifact first, then model fallback.
        expected_features = feature_names
        if (
            not expected_features
            and model is not None
            and hasattr(model, 'feature_names_in_')
        ):
            expected_features = list(model.feature_names_in_)

        # Ensure all expected features are present and in exact training order.
        if expected_features:
            data = data.reindex(columns=expected_features, fill_value=0)

        # Apply scaling to the full feature vector if scaler is available.
        if scaler is not None and expected_features:
            data = pd.DataFrame(
                scaler.transform(data),
                columns=data.columns,
                index=data.index
            )

        return data

    except Exception as e:
        logger.error('Error in preprocessing: %s', str(e))
        raise_api_error(
            'PREPROCESSING_ERROR',
            'Failed to preprocess request payload',
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={'reason': str(e)}
        )


def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"


def get_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability"""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global dependency_status, startup_errors

    logger.info('Starting Telco Churn Prediction API...')
    startup_errors = []
    dependency_status = validate_startup_dependencies()

    if not dependency_status.get('all_required_available'):
        startup_errors.append('Required model artifacts are missing')

    success = load_model_artifacts()
    if not success:
        startup_errors.append('Failed to load model artifacts')
        logger.warning('Model artifacts not fully loaded')
    else:
        logger.info('API ready to serve predictions')


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - start_time).total_seconds()

    return HealthResponse(
        status="healthy",
        service="churn-prediction-api",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        model_loaded=model is not None,
        uptime_seconds=round(uptime, 2)
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe"""
    ready = model is not None and not startup_errors
    payload = {
        'status': 'ready' if ready else 'not_ready',
        'model_loaded': model is not None,
        'dependency_status': dependency_status,
        'startup_errors': startup_errors,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    if not ready:
        raise_api_error(
            'SERVICE_NOT_READY',
            'Service is not ready to accept traffic',
            status.HTTP_503_SERVICE_UNAVAILABLE,
            details=payload,
        )

    return payload


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(_: None = Depends(require_auth)):
    """Get model metadata"""
    if model is None:
        raise_api_error(
            'MODEL_NOT_LOADED',
            'Model not loaded',
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return ModelInfo(
        model_name="Best Churn Prediction Model",
        model_path=str(model_path) if model_path else "unknown",
        model_loaded=True,
        feature_count=len(feature_names) if feature_names else 0,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(
    customer: CustomerData,
    _: None = Depends(require_auth)
):
    """Predict churn for a single customer"""
    if model is None:
        raise_api_error(
            'MODEL_NOT_LOADED',
            'Model not loaded',
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        start = datetime.now()

        # Preprocess data
        processed_data = preprocess_customer_data(customer)

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        # Record metrics
        MetricsCollector.record_prediction(
            model_name='churn_api',
            prediction=int(prediction),
            score=float(probability)
        )

        processing_time = (datetime.now() - start).total_seconds() * 1000
        logger.info(
            'Prediction completed in %.2fms - Churn: %s, Prob: %.3f',
            processing_time,
            prediction,
            probability,
        )

        return PredictionResponse(
            prediction=int(prediction),
            probability=round(float(probability), 4),
            confidence=get_confidence_level(float(probability)),
            risk_level=get_risk_level(float(probability)),
            timestamp=datetime.utcnow().isoformat() + 'Z',
            model_version="1.0.0"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Prediction error: %s', str(e))
        raise_api_error(
            'PREDICTION_ERROR',
            'Prediction failed',
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={'reason': str(e)}
        )


@app.post(
    '/batch_predict',
    response_model=BatchPredictionResponse,
    tags=['Prediction']
)
async def batch_predict_churn(
    request: BatchPredictionRequest,
    _: None = Depends(require_auth)
):
    """Predict churn for multiple customers"""
    if model is None:
        raise_api_error(
            'MODEL_NOT_LOADED',
            'Model not loaded',
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if len(request.customers) > MAX_BATCH_SIZE:
        raise_api_error(
            'BATCH_TOO_LARGE',
            f'Batch size exceeds configured limit ({MAX_BATCH_SIZE})',
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details={'max_batch_size': MAX_BATCH_SIZE}
        )

    try:
        start = datetime.now()
        predictions = []

        for customer in request.customers:
            # Preprocess and predict
            processed_data = preprocess_customer_data(customer)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]

            predictions.append(PredictionResponse(
                prediction=int(prediction),
                probability=round(float(probability), 4),
                confidence=get_confidence_level(float(probability)),
                risk_level=get_risk_level(float(probability)),
                timestamp=datetime.utcnow().isoformat() + 'Z',
                model_version="1.0.0"
            ))

        processing_time = (datetime.now() - start).total_seconds() * 1000
        churn_count = sum(1 for p in predictions if p.prediction == 1)
        total = len(predictions)

        logger.info(
            'Batch prediction completed: %s customers in %.2fms',
            total,
            processing_time,
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=total,
            churn_count=churn_count,
            churn_percentage=round((churn_count / total)
                                   * 100, 2) if total > 0 else 0,
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Batch prediction error: %s', str(e))
        raise_api_error(
            'BATCH_PREDICTION_ERROR',
            'Batch prediction failed',
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={'reason': str(e)}
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(_: None = Depends(require_auth)):
    """Prometheus metrics endpoint"""
    from utils.metrics import get_metrics_text, get_metrics_content_type
    from fastapi.responses import Response

    return Response(
        content=get_metrics_text(),
        media_type=get_metrics_content_type()
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('API_PORT', '8000'))
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=port,
        reload=True,
        log_level='info'
    )
