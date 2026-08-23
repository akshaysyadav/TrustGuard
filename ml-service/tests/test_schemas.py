import pytest
from pydantic import ValidationError

from src.api.schemas import URLRequest, URLPredictionResponse


def test_valid_url_request():
    request = URLRequest(url="https://google.com")

    assert request.url == "https://google.com"


def test_empty_url_rejected():
    with pytest.raises(ValidationError):
        URLRequest(url="")


def test_prediction_response():
    response = URLPredictionResponse(
        verdict="Phishing",
        confidence=99.34,
        reason="XGBoost Model Prediction"
    )

    assert response.verdict == "Phishing"
    assert response.confidence == 99.34
    assert response.reason == "XGBoost Model Prediction"