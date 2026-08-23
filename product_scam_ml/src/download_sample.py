from huggingface_hub import hf_hub_download
import pandas as pd
import os

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

FILE = "raw_meta_Electronics/full-00000-of-00010.parquet"

output_dir = "/data/raw"
os.makedirs(output_dir, exist_ok=True)

print("Downloading Electronics metadata shard...")

file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE,
    repo_type="dataset"
)

print("Downloaded:")
print(file_path)

print("\nReading sample...")

df = pd.read_parquet(file_path)

sample = df.head(10000)

output_path = "/data/raw/amazon_electronics_metadata_sample.parquet"

sample.to_parquet(
    output_path,
    index=False
)

print(f"\nSaved {len(sample)} records to:")
print(output_path)

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())