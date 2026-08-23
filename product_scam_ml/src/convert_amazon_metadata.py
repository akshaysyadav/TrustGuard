import json
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "amazon_electronics_metadata_sample.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "amazon_metadata_lookup.json"
)


# ============================================================
# CONVERT NUMPY / PANDAS VALUES
# ============================================================

def make_json_serializable(value):

    if isinstance(value, np.ndarray):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if pd.isna(value):
        return None

    return value


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AMAZON METADATA → JSON LOOKUP")
print("=" * 80)

print("\nReading:")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# CREATE ASIN LOOKUP
# ============================================================

lookup = {}

for record in df.to_dict(orient="records"):

    asin = record.get("parent_asin")

    if not asin:
        continue

    cleaned_record = make_json_serializable(record)

    lookup[str(asin)] = cleaned_record


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        lookup,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CONVERSION COMPLETE")
print("=" * 80)

print(f"\nProducts indexed: {len(lookup)}")

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nSample ASINs:")

for asin in list(lookup.keys())[:5]:
    print(" -", asin)

print("\n" + "=" * 80)