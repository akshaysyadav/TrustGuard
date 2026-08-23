from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_whitelisted_url():
    response = client.post(
        "/api/v1/predict/url",
        json={"url": "https://google.com"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["verdict"] == "Legitimate"
    assert data["confidence"] == 100.0
    assert data["reason"] == "Whitelisted Domain"


def test_phishing_prediction_endpoint():
    response = client.post(
        "/api/v1/predict/url",
        json={"url": "http://uqr.to/1il1z"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["verdict"] in ["Phishing", "Legitimate"]
    assert 0 <= data["confidence"] <= 100
    assert data["reason"] == "XGBoost Model Prediction"


def test_empty_url_rejected():
    response = client.post(
        "/api/v1/predict/url",
        json={"url": ""}
    )

    assert response.status_code == 422