"""
FastAPI production API for AML detection.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
import pandas as pd

from src.models.predict import predict, load_model
from src.features.engineering import build_features
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="AML Detection API",
    description="Anti-Money Laundering detection system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def get_model():
    """Load the model on first use; return None if it is not available."""
    global model
    if model is None:
        try:
            logger.info("Loading model")
            model = load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    return model


@app.on_event("startup")
async def startup_event():
    """Try to load model on startup; the API starts without it."""
    if get_model() is None:
        logger.warning("API will start without model (predictions will fail)")


class Transaction(BaseModel):
    """Transaction schema. Optional fields missing from the request are sent to the model as NaN."""
    amount: float = Field(..., gt=0)
    payment_format: int
    hour: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    transaction_count: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "amount": 1500.0,
            "payment_format": 2,
            "hour": 14,
            "day_of_week": 3,
            "transaction_count": 5
        }
    })


class BatchRequest(BaseModel):
    """Batch request schema."""
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0, le=1)
    risk_level: str


class BatchResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: List[PredictionResponse]


def risk_level_for(probability: float) -> str:
    """Map probability to risk level."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.85:
        return "HIGH"
    return "CRITICAL"


def score_transactions(transactions: List[Transaction]) -> List[PredictionResponse]:
    """Score transactions with the loaded model."""
    if not transactions:
        return []

    current_model = get_model()
    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Cannot make predictions."
        )

    df = pd.DataFrame([t.model_dump() for t in transactions]).astype(float)
    df_features = build_features(df)
    probabilities = predict(df_features, current_model, return_proba=True)

    return [
        PredictionResponse(
            prediction=int(p > 0.5),
            probability=float(p),
            risk_level=risk_level_for(float(p))
        )
        for p in probabilities
    ]


@app.get("/")
async def root():
    """Service info."""
    return {"message": "AML Detection API", "status": "online"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": get_model() is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    """Predict single transaction."""
    return score_transactions([transaction])[0]


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch_endpoint(request: BatchRequest):
    """Predict multiple transactions."""
    return BatchResponse(predictions=score_transactions(request.transactions))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
