import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"

print("=" * 60)
print("FAKE NEWS DATASET — RAW DATA INSPECTION")
print("=" * 60)

# Load datasets
fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print("\n--- FILE INFORMATION ---")
print(f"Fake.csv shape: {fake_df.shape}")
print(f"True.csv shape: {true_df.shape}")

print("\n--- FAKE DATASET COLUMNS ---")
print(fake_df.columns.tolist())

print("\n--- TRUE DATASET COLUMNS ---")
print(true_df.columns.tolist())

print("\n--- FAKE DATASET INFO ---")
print(fake_df.info())

print("\n--- TRUE DATASET INFO ---")
print(true_df.info())

print("\n--- FIRST 3 FAKE ARTICLES ---")
print(fake_df.head(3).to_string())

print("\n--- FIRST 3 REAL ARTICLES ---")
print(true_df.head(3).to_string())

print("\n--- MISSING VALUES: FAKE ---")
print(fake_df.isnull().sum())

print("\n--- MISSING VALUES: TRUE ---")
print(true_df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(f"Fake duplicates: {fake_df.duplicated().sum()}")
print(f"True duplicates: {true_df.duplicated().sum()}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)