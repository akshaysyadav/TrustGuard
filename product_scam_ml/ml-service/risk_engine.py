
# ============================================================
# TRUSTGUARD — PHASE 5B PRODUCTION RISK ENGINE
# ============================================================
#
# Input:
#   Raw product metadata JSON
#
# Output:
#   Isolation Forest anomaly score
#   Risk-indicator score
#   Final 0-100 risk score
#   Risk level
#   Top explanations
#
# IMPORTANT:
#   - No model training occurs here.
#   - No calibration occurs here.
#   - Frozen Phase 5A.2 configuration is used.
# ============================================================

import os
import json
import math
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "trustguard_metadata_isolation_forest.joblib"
)

CALIBRATION_FILE = os.path.join(
    BASE_DIR,
    "models",
    "trustguard_risk_calibration.json"
)


# ============================================================
# LOAD ARTIFACTS ONCE
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_FILE}"
    )

if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(
        f"Calibration artifact not found: {CALIBRATION_FILE}"
    )


ARTIFACT = joblib.load(MODEL_FILE)

with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
    CALIBRATION = json.load(f)


MODEL = ARTIFACT["model"]
IMPUTER = ARTIFACT["imputer"]

MODEL_FEATURES = ARTIFACT["model_features"]

LOW_Q01 = float(
    CALIBRATION["bounds"]["low_q01"]
)

HIGH_Q99 = float(
    CALIBRATION["bounds"]["high_q99"]
)

ISOLATION_WEIGHT = float(
    CALIBRATION["risk_weights"]["isolation_forest"]
)

INDICATOR_WEIGHT = float(
    CALIBRATION["risk_weights"]["risk_indicators"]
)

INDICATOR_WEIGHTS = CALIBRATION["indicator_weights"]


# ============================================================
# VALIDATION
# ============================================================

if len(MODEL_FEATURES) != MODEL.n_features_in_:
    raise RuntimeError(
        f"Model expects {MODEL.n_features_in_} features, "
        f"but artifact contains {len(MODEL_FEATURES)}."
    )

if HIGH_Q99 <= LOW_Q01:
    raise RuntimeError(
        "Invalid frozen calibration bounds."
    )


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(
            safe_text(x) for x in value
        )

    return str(value)


def safe_number(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "")
            value = value.strip()

            if not value:
                return default

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


# ============================================================
# AMAZON NESTED FIELD PARSERS
# ============================================================

def parse_array(value):
    """
    Correctly handles:
        numpy.ndarray
        list
        tuple
        scalar
        None
    """

    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def parse_features(value):
    """
    Amazon metadata 'features' field.

    Example:
        [
            "UPC: 123456",
            "Weight: 0.600 lbs"
        ]
    """

    values = parse_array(value)

    return [
        str(x).strip()
        for x in values
        if x is not None and str(x).strip()
    ]


def parse_categories(value):
    """
    Amazon metadata 'categories' field.
    """

    values = parse_array(value)

    return [
        str(x).strip()
        for x in values
        if x is not None and str(x).strip()
    ]


def parse_images(value):
    """
    Amazon metadata 'images' is a dictionary containing
    arrays such as:

        hi_res
        large
        thumb
        variant

    Prefer hi_res, then large, then thumb.
    """

    if not isinstance(value, dict):
        return []

    for key in ["hi_res", "large", "thumb"]:
        values = parse_array(value.get(key))

        cleaned = [
            str(x).strip()
            for x in values
            if x is not None
            and str(x).strip()
            and str(x).lower() != "none"
        ]

        if cleaned:
            return cleaned

    return []


def parse_videos(value):
    """
    Amazon metadata 'videos' is a dictionary:

        {
            "title": [...],
            "url": [...],
            "user_id": [...]
        }
    """

    if not isinstance(value, dict):
        return []

    titles = parse_array(value.get("title"))
    urls = parse_array(value.get("url"))

    max_len = max(
        len(titles),
        len(urls)
    )

    videos = []

    for i in range(max_len):

        title = (
            str(titles[i]).strip()
            if i < len(titles)
            and titles[i] is not None
            else ""
        )

        url = (
            str(urls[i]).strip()
            if i < len(urls)
            and urls[i] is not None
            else ""
        )

        if title or url:
            videos.append({
                "title": title,
                "url": url
            })

    return videos


# ============================================================
# TEXT FEATURE HELPERS
# ============================================================

def word_count(text):
    text = safe_text(text).strip()

    if not text:
        return 0

    return len(text.split())


def uppercase_ratio(text):
    text = safe_text(text)

    letters = [
        c for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    return sum(
        c.isupper() for c in letters
    ) / len(letters)


def special_character_ratio(text):
    text = safe_text(text)

    if not text:
        return 0.0

    return sum(
        not c.isalnum() and not c.isspace()
        for c in text
    ) / len(text)


# ============================================================
# PRICE PARSING
# ============================================================

def parse_price(value):

    number = safe_number(value)

    if not np.isnan(number):
        return number

    text = safe_text(value)

    if not text:
        return np.nan

    import re

    match = re.search(
        r"[-+]?\d*\.?\d+",
        text.replace(",", "")
    )

    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    return np.nan


# ============================================================
# RAW METADATA → FEATURE VECTOR
# ============================================================

def extract_features(product):
    """
    Converts raw product metadata into the feature schema
    expected by the trained Isolation Forest.
    """

    title = safe_text(
        product.get("title")
    )

    description = safe_text(
        product.get("description")
    )

    features = parse_features(
        product.get("features")
    )

    categories = parse_categories(
        product.get("categories")
    )

    images = parse_images(
        product.get("images")
    )

    videos = parse_videos(
        product.get("videos")
    )

    seller = product.get("seller")

    price = parse_price(
        product.get("price")
    )

    average_rating = safe_number(
        product.get("average_rating")
    )

    rating_number = safe_number(
        product.get("rating_number")
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    title_length = len(title)

    title_word_count = word_count(
        title
    )

    upper_ratio = uppercase_ratio(
        title
    )

    special_ratio = special_character_ratio(
        title
    )

    description_length = len(
        description
    )

    description_word_count = word_count(
        description
    )

    # --------------------------------------------------------
    # NESTED METADATA
    # --------------------------------------------------------

    feature_count = len(features)

    feature_text_length = sum(
        len(x) for x in features
    )

    category_count = len(categories)

    image_count = len(images)

    video_count = len(videos)

    has_videos = int(
        video_count > 0
    )

    # --------------------------------------------------------
    # SELLER
    # --------------------------------------------------------

    seller_missing = int(
        seller is None
        or safe_text(seller).strip() == ""
    )

    seller_name_length = len(
        safe_text(seller)
    )

    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    if np.isnan(average_rating):
        average_rating = 0.0

    if np.isnan(rating_number):
        rating_number = 0.0

    log_rating_number = math.log1p(
        max(rating_number, 0)
    )

    rating_extremeness = (
        abs(average_rating - 3.0)
        if average_rating > 0
        else 0.0
    )

    high_rating = int(
        average_rating >= 4.5
    )

    low_review_count = int(
        rating_number <= 5
    )

    high_rating_low_reviews = int(
        average_rating >= 4.5
        and rating_number <= 10
    )

    # --------------------------------------------------------
    # FINAL MODEL FEATURES
    #
    # IMPORTANT:
    # This dictionary is intentionally restricted to the
    # 22 features actually used by the trained model.
    # --------------------------------------------------------

    row = {
        "title_length": title_length,
        "title_word_count": title_word_count,
        "uppercase_ratio": upper_ratio,
        "special_character_ratio": special_ratio,

        "description_length": description_length,
        "description_word_count": description_word_count,

        "feature_count": feature_count,
        "feature_text_length": feature_text_length,
        "category_count": category_count,

        "image_count": image_count,
        "video_count": video_count,
        "has_videos": has_videos,

        "seller_missing": seller_missing,
        "seller_name_length": seller_name_length,

        "price_missing": int(
            np.isnan(price)
        ),

        "average_rating": average_rating,
        "rating_number": rating_number,
        "log_rating_number": log_rating_number,
        "rating_extremeness": rating_extremeness,
        "high_rating": high_rating,
        "low_review_count": low_review_count,
        "high_rating_low_reviews": high_rating_low_reviews,
    }

    return row


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(decision_score):
    """
    Uses frozen Phase 5A.2 Q01/Q99 calibration bounds.

    Lower Isolation Forest decision score
    = greater anomaly risk.
    """

    norm = 1.0 - (
        (decision_score - LOW_Q01)
        / (HIGH_Q99 - LOW_Q01)
    )

    return float(
        np.clip(norm, 0, 1) * 100
    )


# ============================================================
# INDICATORS
# ============================================================

def extract_indicators(row):

    return {
        "extreme_rating": int(
            row["average_rating"] >= 4.8
            or row["average_rating"] <= 1.5
        ),

        "very_low_review_count": int(
            row["rating_number"] <= 5
        ),

        "high_rating_low_reviews": int(
            row["average_rating"] >= 4.5
            and row["rating_number"] <= 10
        ),

        "missing_seller": int(
            row["seller_missing"] == 1
        ),

        "very_short_title": int(
            row["title_length"] < 10
        ),

        "very_short_description": int(
            row["description_length"] < 20
        )
    }


def calculate_indicator_score(row):

    indicators = extract_indicators(row)

    weighted_sum = sum(
        indicators[key] * INDICATOR_WEIGHTS[key]
        for key in INDICATOR_WEIGHTS
    )

    total_weight = sum(
        INDICATOR_WEIGHTS.values()
    )

    return float(
        weighted_sum / total_weight * 100
    )


# ============================================================
# RISK LEVEL
# ============================================================

def assign_risk_level(score):

    if score < 25:
        return "LOW"

    if score < 50:
        return "MEDIUM"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# EXPLANATIONS
# ============================================================

EXPLANATION_TEXT = {

    "extreme_rating":
        "Product has an unusually extreme rating.",

    "very_low_review_count":
        "Product has very few reviews.",

    "high_rating_low_reviews":
        "Product has a high rating but very few reviews.",

    "missing_seller":
        "Seller information is missing.",

    "very_short_title":
        "Product title is unusually short.",

    "very_short_description":
        "Product description is unusually short.",
}


def generate_explanations(row, anomaly_score, indicator_score):

    indicators = extract_indicators(row)

    explanations = []

    # --------------------------------------------------------
    # Indicator explanations
    # --------------------------------------------------------

    for key, triggered in indicators.items():

        if triggered:

            explanations.append({
                "type": "risk_indicator",
                "indicator": key,
                "weight": float(
                    INDICATOR_WEIGHTS[key]
                ),
                "explanation":
                    EXPLANATION_TEXT[key]
            })

    # --------------------------------------------------------
    # Isolation Forest explanation
    # --------------------------------------------------------

    if anomaly_score >= 75:

        explanations.append({
            "type": "model",
            "indicator": "isolation_forest",
            "weight": ISOLATION_WEIGHT,
            "explanation":
                "Product metadata is highly anomalous relative "
                "to the reference metadata distribution."
        })

    elif anomaly_score >= 50:

        explanations.append({
            "type": "model",
            "indicator": "isolation_forest",
            "weight": ISOLATION_WEIGHT,
            "explanation":
                "Product metadata shows elevated anomaly "
                "relative to the reference distribution."
        })

    # --------------------------------------------------------
    # Sort strongest explanations first
    # --------------------------------------------------------

    explanations.sort(
        key=lambda x: x.get("weight", 0),
        reverse=True
    )

    return explanations[:5]


# ============================================================
# MAIN PRODUCTION PIPELINE
# ============================================================

def predict_product_risk(product):

    if not isinstance(product, dict):
        raise TypeError(
            "product must be a dictionary."
        )

    # --------------------------------------------------------
    # Raw → engineered features
    # --------------------------------------------------------

    feature_row = extract_features(
        product
    )

    feature_df = pd.DataFrame(
        [feature_row]
    )

    # Enforce authoritative feature order
    X_raw = feature_df[
        MODEL_FEATURES
    ]

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    X_imputed = IMPUTER.transform(
        X_raw
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    prediction = int(
        MODEL.predict(X_imputed)[0]
    )

    decision_score = float(
        MODEL.decision_function(
            X_imputed
        )[0]
    )

    score_samples = float(
        MODEL.score_samples(
            X_imputed
        )[0]
    )

    is_anomaly = prediction == -1

    # --------------------------------------------------------
    # Risk scores
    # --------------------------------------------------------

    anomaly_score = calculate_anomaly_score(
        decision_score
    )

    indicator_score = calculate_indicator_score(
        feature_row
    )

    risk_score = float(
        np.clip(
            ISOLATION_WEIGHT * anomaly_score
            + INDICATOR_WEIGHT * indicator_score,
            0,
            100
        )
    )

    risk_level = assign_risk_level(
        risk_score
    )

    explanations = generate_explanations(
        feature_row,
        anomaly_score,
        indicator_score
    )

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,

        "is_anomaly": is_anomaly,

        "anomaly_score": round(
            anomaly_score,
            2
        ),

        "indicator_score": round(
            indicator_score,
            2
        ),

        "isolation_forest": {
            "prediction": prediction,
            "decision_score": round(
                decision_score,
                6
            ),
            "score_samples": round(
                score_samples,
                6
            )
        },

        "risk_weights": {
            "isolation_forest":
                ISOLATION_WEIGHT,
            "risk_indicators":
                INDICATOR_WEIGHT
        },

        "explanations": explanations
    }

