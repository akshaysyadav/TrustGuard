from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ReviewRequest,
    ReviewPredictionResponse
)

from src.fake_review.classifier import FakeReviewClassifier


router = APIRouter(
    prefix="/api/v1/predict",
    tags=["Fake Review Detection"]
)


MODEL_PATH = "models/fake_review_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


classifier = FakeReviewClassifier(
    MODEL_PATH,
    VECTORIZER_PATH
)


@router.post(
    "/review",
    response_model=ReviewPredictionResponse
)
def predict_review(request: ReviewRequest):

    try:

        result = classifier.predict(
            request.review
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )