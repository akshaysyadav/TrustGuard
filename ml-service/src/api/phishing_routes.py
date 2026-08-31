from fastapi import APIRouter, HTTPException

from src.api.schemas import URLRequest, URLPredictionResponse
from src.phishing.classifier import PhishingClassifier


router = APIRouter(
    prefix="/api/v1/predict",
    tags=["Phishing Detection"]
)


MODEL_PATH = "models/phishing/xgboost_phishing_model.json"

classifier = PhishingClassifier(MODEL_PATH)


@router.post(
    "/url",
    response_model=URLPredictionResponse
)
def predict_url(request: URLRequest):

    try:
        result = classifier.predict(request.url)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )