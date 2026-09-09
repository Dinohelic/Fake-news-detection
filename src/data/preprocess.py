import pandas as pd
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

FAKE_PATH = RAW_DIR / "Fake.csv"
TRUE_PATH = RAW_DIR / "True.csv"

OUTPUT_PATH = INTERIM_DIR / "cleaned_news.csv"


# ============================================================
# LOAD DATA
# ============================================================

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)


print("=" * 70)
print("PREPROCESSING — FAKE NEWS DATASET")
print("=" * 70)

print(f"\nOriginal dataset size: {len(df):,}")


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def clean_text(text):
    """
    Basic text normalization.

    We intentionally keep the wording and sentence structure
    because the transformer models can benefit from natural
    language context.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["title", "text"],
    keep="first"
)

after_duplicates = len(df)

print("\n--- DUPLICATE REMOVAL ---")
print(
    f"Duplicate title+text rows removed: "
    f"{before_duplicates - after_duplicates:,}"
)
print(f"Dataset size after deduplication: {after_duplicates:,}")


# ============================================================
# REMOVE COMPLETELY EMPTY ARTICLES
# ============================================================

before_empty = len(df)

df = df[
    (df["title"].str.len() > 0) |
    (df["text"].str.len() > 0)
].copy()

after_empty = len(df)

print("\n--- EMPTY CONTENT ---")
print(
    f"Rows with completely empty title and text removed: "
    f"{before_empty - after_empty:,}"
)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

df["content"] = (
    "Title: "
    + df["title"]
    + " Article: "
    + df["text"]
)


# ============================================================
# FINAL COLUMNS
# ============================================================

df = df[
    [
        "title",
        "text",
        "subject",
        "date",
        "label",
        "content"
    ]
]


# ============================================================
# SAVE
# ============================================================

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n--- FINAL DATASET ---")

print(f"Total articles: {len(df):,}")

print("\nClass distribution:")

class_counts = df["label"].value_counts()

print(
    f"FAKE: {class_counts.get(0, 0):,} "
    f"({class_counts.get(0, 0) / len(df) * 100:.2f}%)"
)

print(
    f"REAL: {class_counts.get(1, 0):,} "
    f"({class_counts.get(1, 0) / len(df) * 100:.2f}%)"
)

print("\nColumns:")
print(df.columns.tolist())

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)