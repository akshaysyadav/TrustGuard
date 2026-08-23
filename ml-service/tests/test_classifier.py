from src.phishing.classifier import PhishingClassifier


MODEL_PATH = "models/phishing/xgboost_phishing_model.pkl"


def test_classifier_loads():
    classifier = PhishingClassifier(MODEL_PATH)

    assert classifier.model is not None
    assert classifier.feature_names is not None
    assert len(classifier.feature_names) == 18


def test_whitelisted_domain():
    classifier = PhishingClassifier(MODEL_PATH)

    result = classifier.predict("https://google.com")

    assert result["verdict"] == "Legitimate"
    assert result["confidence"] == 100.0
    assert result["reason"] == "Whitelisted Domain"


def test_model_prediction():
    classifier = PhishingClassifier(MODEL_PATH)

    result = classifier.predict("http://uqr.to/1il1z")

    assert result["reason"] == "XGBoost Model Prediction"
    assert result["verdict"] in ["Phishing", "Legitimate"]
    assert 0 <= result["confidence"] <= 100


def test_prediction_response_structure():
    classifier = PhishingClassifier(MODEL_PATH)

    result = classifier.predict("https://example.com")

    assert "verdict" in result
    assert "confidence" in result
    assert "reason" in result