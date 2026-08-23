import os
import joblib
import pandas as pd

from urllib.parse import urlparse

from src.phishing.url_features import extract_18_url_features

class PhishingClassifier:

    LABEL_MAP = {
        0: "Phishing",
        1: "Legitimate"
    }

    SAFE_DOMAINS = {
        "google.com",
        "github.com",
        "microsoft.com",
        "apple.com",
        "amazon.com"
    }

    def __init__(self, model_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Phishing model not found at: {model_path}"
            )

        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]

    def _get_domain(self, raw_url: str) -> str:

        parsed = urlparse(raw_url)

        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    def predict(self, raw_url: str) -> dict:

        domain = self._get_domain(raw_url)

        # Deterministic whitelist check
        if domain in self.SAFE_DOMAINS:
            return {
                "verdict": "Legitimate",
                "confidence": 100.0,
                "reason": "Whitelisted Domain"
            }

        # Extract the same 18 features used during training
        features_dict = extract_18_url_features(raw_url)

        # Preserve exact training feature order
        input_df = pd.DataFrame(
            [features_dict]
        )[self.feature_names]

        # Model prediction
        prediction = int(self.model.predict(input_df)[0])

        # Model probability
        probabilities = self.model.predict_proba(input_df)[0]

        confidence = float(
            probabilities[prediction] * 100
        )

        return {
            "verdict": self.LABEL_MAP[prediction],
            "confidence": confidence,
            "reason": "XGBoost Model Prediction"
        }