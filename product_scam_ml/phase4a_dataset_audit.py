"""
TrustGuard — Phase 4A
Candidate Dataset Audit

Purpose
-------
1. Download the updated He-associated review dataset.
2. Audit schema, identifiers, missing values and Fake_review_product.
3. Determine whether Fake_review_product provides usable product-level
   ground truth.
4. Map TrustGuard's 26 Amazon metadata features to available raw columns.
5. Determine how many production features can actually be computed.
6. Produce a recommendation:
      A) Supervised XGBoost
      B) Isolation Forest + interpretable risk indicators
      C) Hybrid / conditional architecture

IMPORTANT
---------
No pseudo-mappings are allowed.

Examples of mappings deliberately NOT accepted:
    rating_number -> n_of_reviews
    average_rating -> share_5star
    metadata     -> temporal review features
    metadata     -> network features
"""

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Aman\Desktop\TrustGuard\product_scam_ml"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Official updated dataset
DOWNLOAD_URL = (
    "https://bretthollenbeckcom.wordpress.com/"
    "wp-content/uploads/2026/03/"
    "public_reviews_dataset_cleaned.csv_.zip"
)

ZIP_PATH = RAW_DIR / "public_reviews_dataset_cleaned.zip"
EXTRACT_DIR = RAW_DIR / "he_associated_review_data"

METADATA_PATH = (
    RAW_DIR / "amazon_electronics_metadata_sample.parquet"
)

OLD_HE_PATH = (
    RAW_DIR / "product_level_data_without_img_feats.csv.gz"
)


# ============================================================
# TRUSTGUARD'S 26 AMAZON METADATA FEATURES
# ============================================================

TRUSTGUARD_FEATURES = [
    "title_length",
    "title_word_count",
    "uppercase_ratio",
    "special_character_ratio",

    "description_length",
    "description_word_count",

    "feature_count",
    "feature_text_length",

    "category_count",

    "image_count",
    "video_count",
    "has_videos",

    "seller_missing",
    "seller_name_length",

    "price_numeric",
    "price_missing",
    "price_ratio_to_category",

    "log_price_ratio",
    "price_anomaly",

    "average_rating",
    "rating_number",
    "log_rating_number",
    "rating_extremeness",

    "high_rating",
    "low_review_count",
    "high_rating_low_reviews",
]


# ============================================================
# HE-ASSOCIATED FEATURES
# ============================================================

HE_FEATURES = [
    "std_review_len",
    "tfidf_review_body",
    "n_of_reviews",
    "avg_review_rating",
    "avg_days_between_reviews",
    "stdev_days_between_reviews",
    "max_days_between_reviews",
    "min_days_between_reviews",
    "share_helpful_reviews",
    "share_1star",
    "share_5star",
    "share_photo",
    "pagerank",
    "eigenvector_cent",
    "w_degree",
    "clustering_coef",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_column_name(col):
    """
    Normalize column names for matching only.

    Does NOT alter the actual dataset.
    """
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(col).lower()
    ).strip("_")


def flatten_possible_columns(columns):
    return {
        normalize_column_name(c): c
        for c in columns
    }


def safe_read_csv(path):
    """
    Attempts several ways of reading the downloaded dataset.
    """

    try:
        return pd.read_csv(path)
    except Exception:
        pass

    try:
        return pd.read_csv(path, compression="gzip")
    except Exception:
        pass

    raise RuntimeError(
        f"Could not read CSV file: {path}"
    )


# ============================================================
# 1. DOWNLOAD UPDATED HE DATASET
# ============================================================

section("1. DOWNLOADING UPDATED HE-ASSOCIATED DATASET")

if not ZIP_PATH.exists():

    print("Downloading:")
    print(DOWNLOAD_URL)

    urlretrieve(
        DOWNLOAD_URL,
        ZIP_PATH
    )

    print("\nDownload complete.")

else:
    print("Dataset archive already exists:")
    print(ZIP_PATH)


# ============================================================
# 2. EXTRACT DATASET
# ============================================================

section("2. EXTRACTING DATASET")

EXTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    members = z.namelist()

    print("Files contained in archive:")
    for member in members:
        print(" ", member)

    z.extractall(EXTRACT_DIR)


# ============================================================
# 3. FIND CSV
# ============================================================

section("3. LOCATING REVIEW-LEVEL CSV")

csv_candidates = list(
    EXTRACT_DIR.rglob("*.csv")
)

if not csv_candidates:

    raise FileNotFoundError(
        "No CSV file found inside downloaded archive."
    )

for path in csv_candidates:
    print("Found:")
    print(" ", path)


# Prefer the expected cleaned dataset
expected = [
    p for p in csv_candidates
    if "public_reviews_data_cleaned" in p.name.lower()
]

if expected:
    REVIEW_DATA_PATH = expected[0]
else:
    REVIEW_DATA_PATH = csv_candidates[0]


print("\nUsing:")
print(REVIEW_DATA_PATH)


# ============================================================
# 4. LOAD REVIEW DATA
# ============================================================

section("4. LOADING UPDATED REVIEW DATASET")

reviews = pd.read_csv(
    REVIEW_DATA_PATH,
    low_memory=False
)

print("Shape:", reviews.shape)

print("\nColumns:")
for i, col in enumerate(reviews.columns, 1):
    print(f"{i:3}. {col}")

print("\nDtypes:")
print(reviews.dtypes)


# ============================================================
# 5. BASIC DATASET AUDIT
# ============================================================

section("5. BASIC DATASET AUDIT")

print("Rows:", len(reviews))
print("Columns:", len(reviews.columns))

print("\nDuplicate complete rows:")
print(reviews.duplicated().sum())

print("\nMissing values:")
missing = reviews.isna().sum()
missing_pct = (
    missing / len(reviews) * 100
).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_%": missing_pct
})

print(
    missing_report[
        missing_report["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )
)


# ============================================================
# 6. IDENTIFIER AUDIT
# ============================================================

section("6. IDENTIFIER AUDIT")

normalized = flatten_possible_columns(
    reviews.columns
)

identifier_candidates = {
    "product": [
        "product_id",
        "productid",
        "parent_asin",
        "asin",
        "product_asin",
        "product",
    ],

    "reviewer": [
        "reviewer_id",
        "reviewerid",
        "reviewer",
        "user_id",
        "userid",
    ],

    "review": [
        "review_id",
        "reviewid",
        "review_id",
    ],

    "timestamp": [
        "timestamp",
        "review_date",
        "date",
        "review_time",
        "unix_review_time",
    ],
}


identifier_results = {}

for identifier_type, candidates in identifier_candidates.items():

    matches = []

    for candidate in candidates:

        if candidate in normalized:
            matches.append(
                normalized[candidate]
            )

    identifier_results[
        identifier_type
    ] = matches

    print(
        f"{identifier_type:12} -> "
        f"{matches if matches else 'NOT FOUND'}"
    )


# ============================================================
# 7. FAKE_REVIEW_PRODUCT AUDIT
# ============================================================

section("7. Fake_review_product LABEL AUDIT")

fake_product_candidates = [
    c for c in reviews.columns
    if normalize_column_name(c)
    == "fake_review_product"
]

if not fake_product_candidates:

    print(
        "ERROR: Fake_review_product was NOT found."
    )

    fake_product_available = False

else:

    fake_product_available = True

    label_col = fake_product_candidates[0]

    print(
        "Label column:",
        label_col
    )

    print("\nRaw dtype:")
    print(reviews[label_col].dtype)

    print("\nRaw unique values:")
    print(
        reviews[label_col]
        .value_counts(dropna=False)
    )

    print("\nMissing:")
    print(
        reviews[label_col].isna().sum()
    )


# ============================================================
# 8. PRODUCT-LEVEL LABEL CONSISTENCY
# ============================================================

section("8. PRODUCT-LEVEL LABEL CONSISTENCY")

product_columns = identifier_results["product"]

if fake_product_available and product_columns:

    product_col = product_columns[0]

    print(
        "Product identifier:",
        product_col
    )

    label_consistency = (
        reviews
        .groupby(product_col)[label_col]
        .agg(
            n_rows="size",
            n_unique="nunique",
            min_label="min",
            max_label="max"
        )
        .reset_index()
    )

    print(
        "\nUnique products:",
        len(label_consistency)
    )

    inconsistent = label_consistency[
        label_consistency["n_unique"] > 1
    ]

    print(
        "Products with inconsistent labels:",
        len(inconsistent)
    )

    print(
        "Inconsistency percentage:",
        round(
            len(inconsistent)
            / len(label_consistency)
            * 100,
            4
        ),
        "%"
    )

    print("\nProduct-level label distribution:")

    product_labels = (
        label_consistency[
            "min_label"
        ]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(product_labels)

else:

    label_consistency = None

    print(
        "Cannot perform product-level consistency audit."
    )


# ============================================================
# 9. STRICT PRODUCT-LEVEL GROUND TRUTH
# ============================================================

section("9. STRICT PRODUCT-LEVEL GROUND TRUTH TEST")

ground_truth_valid = False

if (
    fake_product_available
    and product_columns
    and label_consistency is not None
):

    n_products = len(label_consistency)

    n_inconsistent = len(
        label_consistency[
            label_consistency["n_unique"] > 1
        ]
    )

    missing_labels = (
        reviews
        .groupby(product_columns[0])[label_col]
        .apply(lambda x: x.isna().any())
        .sum()
    )

    if (
        n_products > 0
        and n_inconsistent == 0
        and missing_labels == 0
    ):
        ground_truth_valid = True

    print(
        "Products:",
        n_products
    )

    print(
        "Inconsistent labels:",
        n_inconsistent
    )

    print(
        "Products containing missing labels:",
        missing_labels
    )

    print(
        "\nGround-truth consistency:",
        "PASS" if ground_truth_valid else "FAIL"
    )

else:

    print("Ground-truth consistency: FAIL")


# ============================================================
# 10. RAW COLUMN CAPABILITY AUDIT
# ============================================================

section("10. RAW COLUMN CAPABILITY AUDIT")

raw_columns = set(
    normalize_column_name(c)
    for c in reviews.columns
)

print(
    "Normalized raw columns available:",
    len(raw_columns)
)


# ============================================================
# 11. 26-FEATURE MAPPING
# ============================================================

section("11. TRUSTGUARD 26-FEATURE COMPATIBILITY")

# Explicit mappings only.
# These are mappings that preserve semantic meaning.

explicit_feature_sources = {

    "title_length": [
        "title",
    ],

    "title_word_count": [
        "title",
    ],

    "uppercase_ratio": [
        "title",
    ],

    "special_character_ratio": [
        "title",
    ],

    "description_length": [
        "description",
    ],

    "description_word_count": [
        "description",
    ],

    "feature_count": [
        "features",
    ],

    "feature_text_length": [
        "features",
    ],

    "category_count": [
        "categories",
    ],

    "image_count": [
        "images",
    ],

    "video_count": [
        "videos",
    ],

    "has_videos": [
        "videos",
    ],

    "seller_missing": [
        "store",
    ],

    "seller_name_length": [
        "store",
    ],

    "price_numeric": [
        "price",
    ],

    "price_missing": [
        "price",
    ],

    "price_ratio_to_category": [
        # Requires category-level aggregation.
        # Raw price alone is insufficient.
        "price",
        "categories",
    ],

    "log_price_ratio": [
        "price",
        "categories",
    ],

    "price_anomaly": [
        "price",
        "categories",
    ],

    "average_rating": [
        "average_rating",
    ],

    "rating_number": [
        "rating_number",
    ],

    "log_rating_number": [
        "rating_number",
    ],

    "rating_extremeness": [
        "average_rating",
    ],

    "high_rating": [
        "average_rating",
    ],

    "low_review_count": [
        "rating_number",
    ],

    "high_rating_low_reviews": [
        "average_rating",
        "rating_number",
    ],
}


feature_audit = []

for feature in TRUSTGUARD_FEATURES:

    required = explicit_feature_sources.get(
        feature,
        []
    )

    normalized_required = [
        normalize_column_name(x)
        for x in required
    ]

    available = [
        x
        for x in normalized_required
        if x in raw_columns
    ]

    missing = [
        x
        for x in normalized_required
        if x not in raw_columns
    ]

    # --------------------------------------------------------
    # Determine whether the feature is computable.
    # --------------------------------------------------------

    if not required:

        status = "NOT_DEFINED"

    elif len(available) == len(required):

        status = "DIRECT_OR_DERIVABLE"

    else:

        status = "MISSING_INPUT"

    feature_audit.append({

        "feature": feature,

        "required_raw_columns":
            ", ".join(required),

        "available_columns":
            ", ".join(available),

        "missing_columns":
            ", ".join(missing),

        "status":
            status
    })


feature_audit_df = pd.DataFrame(
    feature_audit
)

print(
    feature_audit_df.to_string(
        index=False
    )
)


# ============================================================
# 12. COMPATIBILITY COUNTS
# ============================================================

section("12. FEATURE COMPATIBILITY SUMMARY")

direct_count = (
    feature_audit_df["status"]
    .eq("DIRECT_OR_DERIVABLE")
    .sum()
)

missing_count = (
    feature_audit_df["status"]
    .eq("MISSING_INPUT")
    .sum()
)

undefined_count = (
    feature_audit_df["status"]
    .eq("NOT_DEFINED")
    .sum()
)

total_features = len(
    feature_audit_df
)

print(
    f"Total TrustGuard features : {total_features}"
)

print(
    f"Computable from dataset   : {direct_count}"
)

print(
    f"Missing required inputs   : {missing_count}"
)

print(
    f"Undefined mappings        : {undefined_count}"
)

print(
    f"Compatibility              : "
    f"{direct_count / total_features * 100:.2f}%"
)


# ============================================================
# 13. IMPORTANT: DISTINGUISH RAW AVAILABILITY
#     FROM DERIVABILITY
# ============================================================

section("13. RAW AVAILABILITY VS DERIVABILITY")

print(
    """
IMPORTANT:

A feature is considered compatible only when its mathematical
meaning can be reconstructed from the available raw columns.

Examples:

VALID:
    title_length <- title
    title_word_count <- title
    image_count <- images
    price_numeric <- price
    log_rating_number <- rating_number

CONDITIONALLY VALID:
    price_ratio_to_category
    log_price_ratio
    price_anomaly

These require category-level reference statistics.

INVALID:
    n_of_reviews <- rating_number

    average_rating <- share_5star

    metadata <- review temporal behaviour

Those would change the semantic meaning of the feature.
"""
)


# ============================================================
# 14. HE LABEL ↔ PRODUCT ID OVERLAP
# ============================================================

section("14. HE PRODUCT-LEVEL DATA OVERLAP")

if OLD_HE_PATH.exists():

    try:

        old_he = pd.read_csv(
            OLD_HE_PATH,
            compression="gzip"
        )

        print(
            "Existing original He dataset:",
            old_he.shape
        )

        if product_columns:

            print(
                "Updated dataset product IDs:",
                reviews[
                    product_columns[0]
                ].nunique()
            )

        if "product_ID" in old_he.columns:

            print(
                "Original He product IDs:",
                old_he[
                    "product_ID"
                ].nunique()
            )

            print(
                """
NOTE:
The original He product_ID is not automatically assumed
to be the same identifier as the updated review dataset's
product identifier.

An explicit identifier mapping is required before merging.
"""
            )

    except Exception as e:

        print(
            "Could not inspect original He dataset:",
            e
        )

else:

    print(
        "Original product_level_data_without_img_feats.csv.gz"
        " not found."
    )


# ============================================================
# 15. PRODUCTION SUPERVISED MODEL DECISION
# ============================================================

section("15. PRODUCTION MODEL DECISION")

# A supervised production model needs BOTH:
#
#   1. trustworthy product-level labels
#   2. enough production-compatible features
#
# We intentionally use conservative thresholds.

compatibility_ratio = (
    direct_count / total_features
)

if ground_truth_valid and compatibility_ratio >= 0.60:

    decision = (
        "SUPERVISED XGBOOST IS JUSTIFIABLE"
    )

elif ground_truth_valid and compatibility_ratio >= 0.30:

    decision = (
        "HYBRID / LIMITED SUPERVISED MODEL"
    )

else:

    decision = (
        "ISOLATION FOREST + "
        "INTERPRETABLE RISK INDICATORS"
    )


print("\nFINAL DECISION:")
print(decision)


# ============================================================
# 16. MORE DETAILED ARCHITECTURAL RECOMMENDATION
# ============================================================

section("16. ARCHITECTURAL RECOMMENDATION")

if (
    ground_truth_valid
    and compatibility_ratio >= 0.60
):

    recommendation = """
Recommended architecture:

                    AMAZON METADATA
                           |
                           v
                  Feature Engineering
                           |
                           v
                 Production Feature Set
                           |
                           v
                    XGBoost Model
                           |
                           v
                  Scam Risk Probability
                           |
                           v
                 Explainable Risk Factors


The He-associated dataset can provide a defensible
product-level supervised target, provided the label is
consistent and the production feature space is sufficiently
represented.

The original 16-feature He benchmark should remain separate.
"""

elif ground_truth_valid:

    recommendation = """
Recommended architecture:

                 AMAZON METADATA
                       |
                       v
              Metadata Risk Engine
                       |
             +---------+---------+
             |                   |
             v                   v
      Isolation Forest     Rule-based Indicators
             |                   |
             +---------+---------+
                       |
                       v
                Combined Risk Score


A supervised model may be investigated as a secondary
experiment, but should NOT be presented as a fully equivalent
replacement for the original He feature space.

The research benchmark remains:

    He labels + He features + XGBoost

The production engine remains:

    Amazon metadata + anomaly/risk indicators
"""

else:

    recommendation = """
Recommended architecture:

                 AMAZON METADATA
                       |
                       v
              Feature Engineering
                       |
                       v
              Normalized Feature Vector
                       |
              +--------+---------+
              |                  |
              v                  v
       Isolation Forest    Rule Indicators
              |                  |
              +--------+---------+
                       |
                       v
                Product Risk Score
                       |
                       v
               Explainable Factors


Do NOT train a supervised production XGBoost until a
defensible product-level target label is established.

The He dataset should remain exclusively a research benchmark.
"""

print(recommendation)


# ============================================================
# 17. SAVE REPORTS
# ============================================================

section("17. SAVING AUDIT REPORTS")

feature_report_path = (
    REPORT_DIR /
    "phase4a_feature_compatibility.csv"
)

feature_audit_df.to_csv(
    feature_report_path,
    index=False
)

print(
    "Saved:",
    feature_report_path
)


# Save label consistency report
if label_consistency is not None:

    label_report_path = (
        REPORT_DIR /
        "phase4a_product_label_consistency.csv"
    )

    label_consistency.to_csv(
        label_report_path,
        index=False
    )

    print(
        "Saved:",
        label_report_path
    )


# ============================================================
# 18. FINAL JSON SUMMARY
# ============================================================

summary = {

    "dataset": str(
        REVIEW_DATA_PATH
    ),

    "rows": int(
        len(reviews)
    ),

    "columns": int(
        len(reviews.columns)
    ),

    "fake_review_product_available":
        bool(fake_product_available),

    "product_identifier_available":
        bool(product_columns),

    "product_label_consistent":
        bool(ground_truth_valid),

    "trustguard_features_total":
        int(total_features),

    "trustguard_features_computable":
        int(direct_count),

    "trustguard_features_missing_inputs":
        int(missing_count),

    "trustguard_feature_compatibility_pct":
        round(
            compatibility_ratio * 100,
            2
        ),

    "recommended_production_architecture":
        decision
}


summary_path = (
    REPORT_DIR /
    "phase4a_final_audit_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print(
    "Saved:",
    summary_path
)

# ============================================================
# 19. FINAL HUMAN-READABLE SUMMARY
# ============================================================

section("PHASE 4A — FINAL AUDIT SUMMARY")

label_status = "PASS" if ground_truth_valid else "FAIL"
compat_pct = f"{compatibility_ratio * 100:.2f}%"

print(
    f"""
Dataset
-------
Rows                         : {len(reviews):,}
Columns                      : {len(reviews.columns)}

Product-level label
-------------------
Fake_review_product present : {fake_product_available}
Product identifier present  : {bool(product_columns)}
Label consistency            : {label_status}

TrustGuard 26-feature space
---------------------------
Total features               : {total_features}
Computable                   : {direct_count}
Missing inputs               : {missing_count}
Compatibility                : {compat_pct}

Production decision
-------------------
{decision}

Research benchmark
-------------------
He-associated product labels + original He feature space
remain a separate academic benchmark.

Production engine
------------------
Only features genuinely computable from runtime Amazon
inputs should be used.
"""
)

print("\nPhase 4A audit complete.")