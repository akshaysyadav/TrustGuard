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
        # Amazon's country-specific storefronts (all the same company,
        # all share the same long tracking-parameter URL style)
        "amazon.com",
        "amazon.in",
        "amazon.co.uk",
        "amazon.ca",
        "amazon.de",
        # Major e-commerce platforms whose product links commonly carry
        # long affiliate/UTM tracking query strings — the raw model
        # over-weights URL length and query-parameter count, so these
        # legitimate links otherwise get misread as phishing.
        "meesho.com",
        "flipkart.com",
        "myntra.com",
        "ajio.com",
        "gameloot.in",
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

    def _is_safe_domain(self, domain: str) -> bool:
        """
        True if domain is a trusted domain itself, OR a subdomain of one
        (e.g. docs.google.com, mail.google.com, drive.google.com).
        An exact-match-only check misses that legitimate services
        commonly live on subdomains of their main domain.
        """
        return any(
            domain == safe or domain.endswith("." + safe)
            for safe in self.SAFE_DOMAINS
        )

    def predict(self, raw_url: str) -> dict:

        domain = self._get_domain(raw_url)

        # Deterministic whitelist check (covers subdomains too)
        if self._is_safe_domain(domain):
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