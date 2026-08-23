# ============================================================
# TRUSTGUARD — PHASE 5B PRODUCTION RISK ENGINE
# ============================================================
#
# Input:
#   Raw Amazon product metadata JSON
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
#   - Phase 4B.2 model artifact is frozen.
#   - Phase 5A.2 calibration is frozen.
#   - Feature definitions MUST match Phase 4B.2.
# ============================================================

import os
import json
import math
import re

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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
# LOAD FROZEN ARTIFACTS
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

with open(
    CALIBRATION_FILE,
    "r",
    encoding="utf-8"
) as f:
    CALIBRATION = json.load(f)


MODEL = ARTIFACT["model"]

IMPUTER = ARTIFACT["imputer"]

FEATURE_COLUMNS = ARTIFACT["feature_columns"]

MODEL_FEATURES = ARTIFACT["model_features"]


# ============================================================
# FROZEN CALIBRATION
# ============================================================

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

INDICATOR_WEIGHTS = CALIBRATION[
    "indicator_weights"
]


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

if len(MODEL_FEATURES) != MODEL.n_features_in_:
    raise RuntimeError(
        f"Model expects "
        f"{MODEL.n_features_in_} features, "
        f"but artifact contains "
        f"{len(MODEL_FEATURES)}."
    )


if HIGH_Q99 <= LOW_Q01:
    raise RuntimeError(
        "Invalid frozen calibration bounds."
    )


if not math.isclose(
    ISOLATION_WEIGHT + INDICATOR_WEIGHT,
    1.0,
    rel_tol=1e-9
):
    raise RuntimeError(
        "Risk weights must sum to 1.0."
    )


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def is_missing_value(value):

    if value is None:
        return True

    try:

        result = pd.isna(value)

        if isinstance(
            result,
            (bool, np.bool_)
        ):
            return bool(result)

    except Exception:
        pass

    return False


def safe_text(value):

    if is_missing_value(value):
        return ""

    return str(value).strip()


def safe_number(
    value,
    default=np.nan
):

    if is_missing_value(value):
        return default

    try:

        if isinstance(value, str):

            value = value.replace(
                ",",
                ""
            ).strip()

            if not value:
                return default

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# NESTED VALUE HELPERS
# ============================================================

def normalize_sequence(value):

    if value is None:
        return []

    if isinstance(
        value,
        np.ndarray
    ):
        return value.tolist()

    if isinstance(
        value,
        (list, tuple)
    ):
        return list(value)

    return [value]


def valid_items(value):

    items = normalize_sequence(value)

    output = []

    for item in items:

        if is_missing_value(item):
            continue

        if isinstance(item, str):

            if not item.strip():
                continue

        output.append(item)

    return output


# ============================================================
# FEATURES PARSING
# ============================================================

def parse_features(value):

    items = valid_items(value)

    result = []

    for item in items:

        if isinstance(item, dict):

            item = json.dumps(
                item,
                ensure_ascii=False
            )

        text = str(item).strip()

        if text:
            result.append(text)

    return result


def feature_count(value):

    return len(
        parse_features(value)
    )


def feature_text_length(value):

    items = parse_features(value)

    if not items:
        return 0

    return len(
        " ".join(items)
    )


# ============================================================
# CATEGORY PARSING
# ============================================================

def parse_categories(value):

    items = valid_items(value)

    result = []

    for item in items:

        text = str(item).strip()

        if text:
            result.append(text)

    return result


def category_count(value):

    return len(
        parse_categories(value)
    )


def get_leaf_category(value):

    categories = parse_categories(value)

    if not categories:
        return None

    return categories[-1]


# ============================================================
# IMAGE PARSING
#
# IMPORTANT:
# Exact Phase 4B.2 priority:
#
#   large
#   ↓
#   hi_res
#   ↓
#   thumb
#
# Only actual HTTP URLs are counted.
# ============================================================

def parse_images(value):

    if not isinstance(
        value,
        dict
    ):
        return []

    urls = value.get(
        "large",
        []
    )

    if not valid_items(urls):

        urls = value.get(
            "hi_res",
            []
        )

    if not valid_items(urls):

        urls = value.get(
            "thumb",
            []
        )

    result = []

    for item in valid_items(urls):

        text = str(item).strip()

        if text.startswith("http"):

            result.append(text)

    return result


def image_count(value):

    return len(
        parse_images(value)
    )


# ============================================================
# VIDEO PARSING
#
# IMPORTANT:
# Exact Phase 4B.2 behaviour:
# Count actual HTTP video URLs only.
# ============================================================

def parse_videos(value):

    if not isinstance(
        value,
        dict
    ):
        return []

    urls = value.get(
        "url",
        []
    )

    result = []

    for item in valid_items(urls):

        text = str(item).strip()

        if text.startswith("http"):

            result.append(text)

    return result


def video_count(value):

    return len(
        parse_videos(value)
    )


def has_videos(value):

    return int(
        video_count(value) > 0
    )


# ============================================================
# DESCRIPTION PARSING
# ============================================================

def parse_description(value):

    if is_missing_value(value):
        return ""

    if isinstance(
        value,
        dict
    ):

        return json.dumps(
            value,
            ensure_ascii=False
        )

    if isinstance(
        value,
        np.ndarray
    ):

        parts = valid_items(value)

        return " ".join(
            str(x)
            for x in parts
        )

    if isinstance(
        value,
        (list, tuple)
    ):

        parts = valid_items(value)

        return " ".join(
            str(x)
            for x in parts
        )

    return str(value).strip()


def description_length(value):

    return len(
        parse_description(value)
    )


def description_word_count(value):

    text = parse_description(value)

    if not text:
        return 0

    return len(
        re.findall(
            r"\b\w+\b",
            text
        )
    )


# ============================================================
# TITLE FEATURES
# ============================================================

def title_text(value):

    return safe_text(value)


def title_length(value):

    return len(
        title_text(value)
    )


def title_word_count(value):

    text = title_text(value)

    if not text:
        return 0

    return len(
        re.findall(
            r"\b\w+\b",
            text
        )
    )


def uppercase_ratio(value):

    text = title_text(value)

    letters = [
        c
        for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    uppercase = sum(
        c.isupper()
        for c in letters
    )

    return uppercase / len(
        letters
    )


def special_character_ratio(value):

    text = title_text(value)

    if not text:
        return 0.0

    special = sum(
        1
        for c in text
        if not c.isalnum()
        and not c.isspace()
    )

    return special / len(text)


# ============================================================
# SELLER FEATURES
#
# IMPORTANT:
# Phase 4B.2 uses metadata["store"].
# ============================================================

def seller_missing(value):

    text = safe_text(value)

    return int(
        text == ""
    )


def seller_name_length(value):

    text = safe_text(value)

    return len(text)


# ============================================================
# PRICE PARSING
# ============================================================

def parse_price(value):

    if is_missing_value(value):
        return np.nan

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):

        if np.isfinite(value):
            return float(value)

        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    cleaned = re.sub(
        r"[^\d.\-]",
        "",
        text
    )

    if not cleaned:
        return np.nan

    try:

        number = float(cleaned)

        if number < 0:
            return np.nan

        return number

    except Exception:

        return np.nan


# ============================================================
# RATING FEATURES
#
# IMPORTANT:
# Missing values remain NaN.
# The frozen training imputer handles them.
# ============================================================

def build_rating_features(
    average_rating,
    rating_number
):

    average_rating = safe_number(
        average_rating
    )

    rating_number = safe_number(
        rating_number
    )

    if pd.isna(
        rating_number
    ):

        log_rating_number = np.nan

        low_review_count = 0

    else:

        log_rating_number = np.log1p(
            max(
                0,
                rating_number
            )
        )

        # Exact Phase 4B.2 definition.
        low_review_count = int(
            rating_number < 10
        )

    if pd.isna(
        average_rating
    ):

        rating_extremeness = np.nan

        high_rating = 0

    else:

        rating_extremeness = abs(
            average_rating - 3.0
        )

        high_rating = int(
            average_rating >= 4.5
        )

    # Exact Phase 4B.2 definition:
    #
    # high_rating_low_reviews =
    #       high_rating == 1
    #       AND
    #       low_review_count == 1
    #
    high_rating_low_reviews = int(
        high_rating == 1
        and low_review_count == 1
    )

    return {
        "average_rating":
            average_rating,

        "rating_number":
            rating_number,

        "log_rating_number":
            log_rating_number,

        "rating_extremeness":
            rating_extremeness,

        "high_rating":
            high_rating,

        "low_review_count":
            low_review_count,

        "high_rating_low_reviews":
            high_rating_low_reviews
    }


# ============================================================
# CATEGORY PRICE REFERENCE
#
# Uses frozen references from Phase 4B.2 artifact.
# ============================================================

GLOBAL_PRICE_MEDIAN = float(
    ARTIFACT["global_price_median"]
)

CATEGORY_PRICE_MEDIAN = {
    str(k): float(v)
    for k, v
    in ARTIFACT[
        "category_price_median"
    ].items()
}


# ============================================================
# RAW PRODUCT → 26 FEATURE CONTRACT
# ============================================================

def extract_features(product):

    if not isinstance(
        product,
        dict
    ):

        raise TypeError(
            "product must be a dictionary."
        )

    title = product.get(
        "title"
    )

    description = product.get(
        "description"
    )

    features = product.get(
        "features"
    )

    categories = product.get(
        "categories"
    )

    images = product.get(
        "images"
    )

    videos = product.get(
        "videos"
    )

    # IMPORTANT:
    # Amazon metadata training uses "store".
    seller = product.get(
        "store"
    )

    price = product.get(
        "price"
    )

    average_rating = product.get(
        "average_rating"
    )

    rating_number = product.get(
        "rating_number"
    )


    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    title_len = title_length(
        title
    )

    title_words = title_word_count(
        title
    )

    upper_ratio = uppercase_ratio(
        title
    )

    special_ratio = (
        special_character_ratio(
            title
        )
    )

    description_len = (
        description_length(
            description
        )
    )

    description_words = (
        description_word_count(
            description
        )
    )


    # --------------------------------------------------------
    # FEATURES / CATEGORIES
    # --------------------------------------------------------

    feature_cnt = feature_count(
        features
    )

    feature_text_len = (
        feature_text_length(
            features
        )
    )

    category_cnt = category_count(
        categories
    )


    # --------------------------------------------------------
    # MEDIA
    # --------------------------------------------------------

    image_cnt = image_count(
        images
    )

    video_cnt = video_count(
        videos
    )

    videos_present = int(
        video_cnt > 0
    )


    # --------------------------------------------------------
    # SELLER
    # --------------------------------------------------------

    missing_seller = seller_missing(
        seller
    )

    seller_len = seller_name_length(
        seller
    )


    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    parsed_price = parse_price(
        price
    )

    missing_price = int(
        pd.isna(parsed_price)
    )


    # --------------------------------------------------------
    # CATEGORY PRICE FEATURES
    # --------------------------------------------------------

    leaf_category = get_leaf_category(
        categories
    )

    category_reference = (
        CATEGORY_PRICE_MEDIAN.get(
            str(leaf_category),
            GLOBAL_PRICE_MEDIAN
        )
        if leaf_category is not None
        else GLOBAL_PRICE_MEDIAN
    )

    if (
        pd.isna(
            category_reference
        )
        or category_reference <= 0
        or pd.isna(
            parsed_price
        )
    ):

        price_ratio = np.nan
        log_price_ratio = np.nan
        price_anomaly = np.nan

    else:

        price_ratio = (
            parsed_price
            / category_reference
        )

        log_price_ratio = np.log1p(
            price_ratio
        )

        price_anomaly = abs(
            log_price_ratio
        )


    # --------------------------------------------------------
    # RATINGS
    # --------------------------------------------------------

    rating_features = (
        build_rating_features(
            average_rating,
            rating_number
        )
    )


    # --------------------------------------------------------
    # AUTHORITATIVE 26-FEATURE CONTRACT
    # --------------------------------------------------------

    row = {

        "title_length":
            title_len,

        "title_word_count":
            title_words,

        "uppercase_ratio":
            upper_ratio,

        "special_character_ratio":
            special_ratio,

        "description_length":
            description_len,

        "description_word_count":
            description_words,

        "feature_count":
            feature_cnt,

        "feature_text_length":
            feature_text_len,

        "category_count":
            category_cnt,

        "image_count":
            image_cnt,

        "video_count":
            video_cnt,

        "has_videos":
            videos_present,

        "seller_missing":
            missing_seller,

        "seller_name_length":
            seller_len,

        "price_numeric":
            parsed_price,

        "price_missing":
            missing_price,

        "price_ratio_to_category":
            price_ratio,

        "log_price_ratio":
            log_price_ratio,

        "price_anomaly":
            price_anomaly,

        "average_rating":
            rating_features[
                "average_rating"
            ],

        "rating_number":
            rating_features[
                "rating_number"
            ],

        "log_rating_number":
            rating_features[
                "log_rating_number"
            ],

        "rating_extremeness":
            rating_features[
                "rating_extremeness"
            ],

        "high_rating":
            rating_features[
                "high_rating"
            ],

        "low_review_count":
            rating_features[
                "low_review_count"
            ],

        "high_rating_low_reviews":
            rating_features[
                "high_rating_low_reviews"
            ],
    }


    return row


# ============================================================
# ANOMALY SCORE
#
# Frozen Phase 5A.2 Q01/Q99 calibration.
# ============================================================

def calculate_anomaly_score(
    decision_score
):

    norm = 1.0 - (
        (
            decision_score
            - LOW_Q01
        )
        /
        (
            HIGH_Q99
            - LOW_Q01
        )
    )

    return float(
        np.clip(
            norm,
            0,
            1
        )
        * 100
    )


# ============================================================
# RISK INDICATORS
#
# EXACTLY MATCHES PHASE 5A.2
# ============================================================

def extract_indicators(row):

    average_rating = row[
        "average_rating"
    ]

    rating_number = row[
        "rating_number"
    ]

    return {

        "extreme_rating": int(
            pd.notna(
                average_rating
            )
            and (
                average_rating >= 4.8
                or average_rating <= 1.5
            )
        ),

        "very_low_review_count": int(
            pd.notna(
                rating_number
            )
            and rating_number <= 5
        ),

        "high_rating_low_reviews": int(
            pd.notna(
                average_rating
            )
            and pd.notna(
                rating_number
            )
            and average_rating >= 4.5
            and rating_number <= 10
        ),

        "missing_seller": int(
            row[
                "seller_missing"
            ] == 1
        ),

        "very_short_title": int(
            row[
                "title_length"
            ] < 10
        ),

        "very_short_description": int(
            row[
                "description_length"
            ] < 20
        )
    }


def calculate_indicator_score(
    row
):

    indicators = extract_indicators(
        row
    )

    weighted_sum = sum(
        indicators[key]
        * INDICATOR_WEIGHTS[key]
        for key in INDICATOR_WEIGHTS
    )

    total_weight = sum(
        INDICATOR_WEIGHTS.values()
    )

    return float(
        weighted_sum
        / total_weight
        * 100
    )


# ============================================================
# RISK LEVEL
# ============================================================

def assign_risk_level(
    score
):

    if score < 25:
        return "LOW"

    if score < 50:
        return "MEDIUM"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# EXPLANATION TEXT
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
        "Product description is unusually short."
}


# ============================================================
# EXPLANATIONS
# ============================================================

def generate_explanations(
    row,
    anomaly_score
):

    indicators = extract_indicators(
        row
    )

    explanations = []

    # --------------------------------------------------------
    # Indicator explanations
    # --------------------------------------------------------

    for key, triggered in indicators.items():

        if triggered:

            explanations.append({

                "type":
                    "risk_indicator",

                "indicator":
                    key,

                "weight":
                    float(
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

            "type":
                "model",

            "indicator":
                "isolation_forest",

            "weight":
                ISOLATION_WEIGHT,

            "explanation":
                "Product metadata is highly anomalous "
                "relative to the reference metadata "
                "distribution."
        })

    elif anomaly_score >= 50:

        explanations.append({

            "type":
                "model",

            "indicator":
                "isolation_forest",

            "weight":
                ISOLATION_WEIGHT,

            "explanation":
                "Product metadata shows elevated anomaly "
                "relative to the reference distribution."
        })


    # --------------------------------------------------------
    # Strongest explanations first
    # --------------------------------------------------------

    explanations.sort(
        key=lambda x: x.get(
            "weight",
            0
        ),
        reverse=True
    )

    return explanations[:5]


# ============================================================
# MAIN PRODUCTION PIPELINE
# ============================================================

def predict_product_risk(
    product
):

    print("\n========== PYTHON RISK ENGINE INPUT ==========")
    print("PRODUCT:", product)
    print("TITLE:", product.get("title"))
    print("STORE:", product.get("store"))
    print("PRICE:", product.get("price"))
    print("RATING:", product.get("average_rating"))
    print("RATING NUMBER:", product.get("rating_number"))
    print("==============================================\n")

    if not isinstance(
        product,
        dict
    ):

        raise TypeError(
            "product must be a dictionary."
        )


    # --------------------------------------------------------
    # RAW → 26 FEATURES
    # --------------------------------------------------------

    feature_row = extract_features(
        product
    )

    feature_df = pd.DataFrame(
        [feature_row],
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # MODEL FEATURE CONTRACT
    # --------------------------------------------------------

    X_raw = feature_df[
        MODEL_FEATURES
    ]


    # --------------------------------------------------------
    # FROZEN IMPUTER
    # --------------------------------------------------------

    X_imputed = IMPUTER.transform(
        X_raw
    )


    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    prediction = int(
        MODEL.predict(
            X_imputed
        )[0]
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

    is_anomaly = (
        prediction == -1
    )


    # --------------------------------------------------------
    # CALIBRATED ANOMALY SCORE
    # --------------------------------------------------------

    anomaly_score = (
        calculate_anomaly_score(
            decision_score
        )
    )


    # --------------------------------------------------------
    # INDICATOR SCORE
    # --------------------------------------------------------

    indicator_score = (
        calculate_indicator_score(
            feature_row
        )
    )


    # --------------------------------------------------------
    # FINAL RISK SCORE
    # --------------------------------------------------------

    risk_score = float(
        np.clip(
            (
                ISOLATION_WEIGHT
                * anomaly_score
            )
            +
            (
                INDICATOR_WEIGHT
                * indicator_score
            ),
            0,
            100
        )
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    risk_level = assign_risk_level(
        risk_score
    )


    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------

    explanations = (
        generate_explanations(
            feature_row,
            anomaly_score
        )
    )


    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "risk_score":
            round(
                risk_score,
                2
            ),

        "risk_level":
            risk_level,

        # Diagnostic only.
        "is_anomaly":
            is_anomaly,

        "anomaly_score":
            round(
                anomaly_score,
                2
            ),

        "indicator_score":
            round(
                indicator_score,
                2
            ),

        "isolation_forest": {

            "prediction":
                prediction,

            "decision_score":
                round(
                    decision_score,
                    6
                ),

            "score_samples":
                round(
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

        "explanations":
            explanations
    }