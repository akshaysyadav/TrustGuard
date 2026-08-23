
# ============================================================
# TRUSTGUARD — PHASE 5B FASTAPI SERVICE
# ============================================================

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from phase5b_risk_engine import predict_product_risk

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="TrustGuard Product Risk API",
    description=(
        "AI-based product metadata scam-risk assessment "
        "using Isolation Forest and risk indicators."
    ),
    version="5.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class ProductMetadata(BaseModel):

    parent_asin: Optional[str] = None

    title: Optional[str] = None

    description: Optional[Any] = None

    features: Optional[List[Any]] = None

    categories: Optional[List[Any]] = None

    images: Optional[Dict[str, Any]] = None

    videos: Optional[Dict[str, Any]] = None

    store: Optional[str] = None

    price: Optional[Any] = None

    average_rating: Optional[float] = None

    rating_number: Optional[float] = None


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class Explanation(BaseModel):

    type: str

    indicator: str

    weight: float

    explanation: str


class IsolationForestResult(BaseModel):

    prediction: int

    decision_score: float

    score_samples: float


class RiskWeights(BaseModel):

    isolation_forest: float

    risk_indicators: float


class RiskResponse(BaseModel):

    risk_score: float = Field(
        ge=0,
        le=100
    )

    risk_level: str

    is_anomaly: bool

    anomaly_score: float = Field(
        ge=0,
        le=100
    )

    indicator_score: float = Field(
        ge=0,
        le=100
    )

    isolation_forest: IsolationForestResult

    risk_weights: RiskWeights

    explanations: List[Explanation]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "trustguard-risk-engine",
        "version": "5.0.0"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=RiskResponse
)
def predict(product: ProductMetadata):

    try:

        # Convert Pydantic object → dictionary
        raw_product = product.model_dump()

        result = predict_product_risk(
            raw_product
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Risk prediction failed: {str(exc)}"
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "TrustGuard Product Risk API",
        "version": "5.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "prediction": "POST /predict"
        }
    }

