import os
import re
import joblib

from collections import Counter

from src.fake_review.preprocessing import clean_text


# ============================================================
# SPAM / MARKETING-STYLE PATTERN CHECK
#
# The trained model's dataset labels "fake" as AI-generated text
# written to imitate genuine reviews (e.g. Amazon's CG/OR dataset),
# so it does not represent generic marketing-hype spam (repeated
# words, empty superlatives, brand-promotion instead of product
# feedback). This heuristic catches that specific, different
# pattern as a second, independent signal alongside the ML model.
# ============================================================

SPAM_PHRASES = [
    "best purchase ever", "highly recommended", "highly recommend",
    "the greatest", "waste your time", "don't waste your time",
    "do not waste your time", "purchase this immediately",
    "buy this immediately", "buy immediately", "everyone should",
    "must buy", "must have", "perfect product", "perfect quality",
    "perfect service", "look anywhere else", "look no further",
    "best product", "never seen anything", "best in the world",
]

SUPERLATIVES = [
    "best", "perfect", "greatest", "amazing", "excellent", "highly",
    "never", "entire world", "everyone"
]

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "i", "you", "this", "that", "have", "has",
    "was", "were", "be", "been", "with", "so", "very",
}


def detect_spam_patterns(text: str) -> dict:
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    word_counts = Counter(w for w in words if w not in STOPWORDS)

    reasons = []
    score = 0

    # Any non-trivial word repeated for emphasis (e.g. "perfect perfect
    # perfect", "buy buy buy buy buy"). Stopwords are excluded instead
    # of filtering by word length, so short high-signal words like
    # "buy" still get caught.
    repeated = [w for w, c in word_counts.items() if c >= 3]
    if repeated:
        top = max(repeated, key=lambda w: word_counts[w])
        score += 3
        reasons.append(f"word '{top}' repeated {word_counts[top]}x")

    # Generic marketing / spam phrases
    matched = [p for p in SPAM_PHRASES if p in lowered]
    if matched:
        score += len(matched)
        reasons.append("marketing phrase(s): " + ", ".join(matched[:3]))

    # High density of superlatives/hype words in a short review
    sup_count = sum(lowered.count(s) for s in SUPERLATIVES)
    if 0 < len(words) < 60 and sup_count >= 2:
        score += 2
        reasons.append("high density of superlatives/hype words")

    # Excessive exclamation marks
    exclamations = text.count("!")
    if exclamations >= 4:
        score += 2
        reasons.append(f"{exclamations} exclamation marks")

    # Mostly-uppercase "shouting" text
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 15:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.7:
            score += 2
            reasons.append(f"mostly uppercase ({upper_ratio:.0%} of letters)")

    return {
        "is_suspicious": score >= 3,
        "score": score,
        "reasons": reasons,
    }


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
        ml_verdict = self.LABEL_MAP[prediction]

        # Complementary rule-based check, run on the original
        # (uncleaned) text so multi-word phrases stay intact
        spam_check = detect_spam_patterns(review)

        if ml_verdict == "Genuine Review" and spam_check["is_suspicious"]:
            return {
                "verdict": "Fake Review",
                "reason": "Flagged by pattern check: "
                + "; ".join(spam_check["reasons"]),
            }

        return {
            "verdict": ml_verdict,
            "reason": "Linear SVM Model Prediction",
        }