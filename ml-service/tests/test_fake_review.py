from fastapi.testclient import TestClient

from src.main import app
from src.fake_review.classifier import FakeReviewClassifier


client = TestClient(app)


def test_fake_review_classifier_loads():

    classifier = FakeReviewClassifier(
        "models/fake_review_model.pkl",
        "models/tfidf_vectorizer.pkl"
    )

    assert classifier.model is not None
    assert classifier.vectorizer is not None


def test_fake_review_prediction_structure():

    response = client.post(
        "/api/v1/predict/review",
        json={
            "review": "This product is absolutely amazing and the quality is excellent."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "verdict" in data
    assert "reason" in data

    assert data["verdict"] in [
        "Fake Review",
        "Genuine Review"
    ]


def test_empty_review_rejected():

    response = client.post(
        "/api/v1/predict/review",
        json={
            "review": ""
        }
    )

    assert response.status_code == 422