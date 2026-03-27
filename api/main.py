"""FastAPI REST API for Churn Prediction Model Serving"""
from utils.health_check import HealthCheckServer
from utils.metrics import MetricsCollector
from utils.config import load_config
import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

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
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
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


# Global model and artifacts
model = None
scaler = None
label_encoders = None
feature_names = None
model_path = None
start_time = datetime.now()


def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, label_encoders, feature_names, model_path

    try:
        # Load configuration
        config = load_config()
        models_dir = Path(config.get('models_dir', './artifacts/models'))
        data_dir = Path(config.get('artifacts_dir', './artifacts')) / 'data'

        # Load model
        model_path = models_dir / 'best_model.pkl'
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return False

        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load preprocessing artifacts
        scaler_path = data_dir / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")

        encoders_path = data_dir / 'label_encoders.pkl'
        if encoders_path.exists():
            label_encoders = joblib.load(encoders_path)
            logger.info(f"Label encoders loaded from {encoders_path}")

        features_path = data_dir / 'feature_names.pkl'
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
            logger.info(f"Feature names loaded: {len(feature_names)} features")

        if not feature_names and hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            logger.info(
                f"Feature names inferred from model: {len(feature_names)} features")

        logger.info("✅ All model artifacts loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return False


def preprocess_customer_data(customer: CustomerData) -> pd.DataFrame:
    """Preprocess customer data for prediction"""
    try:
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
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'Contract', 'PaymentMethod', 'tenure_group']

        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Build expected schema from saved artifact first, then model fallback.
        expected_features = feature_names
        if not expected_features and model is not None and hasattr(model, 'feature_names_in_'):
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
        logger.error(f"Error in preprocessing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing error: {str(e)}"
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
    logger.info("Starting Telco Churn Prediction API...")
    success = load_model_artifacts()
    if not success:
        logger.warning(
            "⚠️  Model artifacts not fully loaded. Some endpoints may not work.")
    else:
        logger.info("🚀 API ready to serve predictions")


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
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model metadata"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfo(
        model_name="Best Churn Prediction Model",
        model_path=str(model_path) if model_path else "unknown",
        model_loaded=True,
        feature_count=len(feature_names) if feature_names else 0,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerData):
    """Predict churn for a single customer"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
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
            f"Prediction completed in {processing_time:.2f}ms - Churn: {prediction}, Prob: {probability:.3f}")

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
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict_churn(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
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
            f"Batch prediction completed: {total} customers in {processing_time:.2f}ms")

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=total,
            churn_count=churn_count,
            churn_percentage=round((churn_count / total)
                                   * 100, 2) if total > 0 else 0,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    from utils.metrics import get_metrics_text, get_metrics_content_type
    from fastapi.responses import Response

    return Response(
        content=get_metrics_text(),
        media_type=get_metrics_content_type()
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
