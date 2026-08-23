import os
import joblib

from src.fake_review.preprocessing import clean_text


class FakeReviewClassifier:

    LABEL_MAP = {
        0: "Genuine Review",
        1: "Fake Review"
    }

    def __init__(self, model_path: str, vectorizer_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fake review model not found at: {model_path}"
            )

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at: {vectorizer_path}"
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, review: str) -> dict:

        if not review or not review.strip():
            raise ValueError("Review cannot be empty")

        # Same preprocessing used during training
        cleaned_review = clean_text(review)

        # Convert cleaned text into the same TF-IDF feature space
        review_tfidf = self.vectorizer.transform(
            [cleaned_review]
        )

        # Model prediction
        prediction = int(
            self.model.predict(review_tfidf)[0]
        )

        return {
            "verdict": self.LABEL_MAP[prediction],
            "reason": "Linear SVM Model Prediction"
        }