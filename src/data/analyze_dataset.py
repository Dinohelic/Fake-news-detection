import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"


# --------------------------------------------------
# Load raw datasets
# --------------------------------------------------

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("=" * 70)
print("DEEP DATA QUALITY ANALYSIS")
print("=" * 70)

print("\n--- DATASET SIZE ---")
print(f"Fake articles : {len(fake_df):,}")
print(f"Real articles : {len(true_df):,}")
print(f"Total articles: {len(df):,}")


# --------------------------------------------------
# 1. Exact duplicate analysis
# --------------------------------------------------

print("\n--- EXACT DUPLICATES ---")

fake_duplicates = fake_df.duplicated().sum()
real_duplicates = true_df.duplicated().sum()

print(f"Fake duplicate rows : {fake_duplicates:,}")
print(f"Real duplicate rows : {real_duplicates:,}")
print(f"Total duplicate rows: {fake_duplicates + real_duplicates:,}")


# --------------------------------------------------
# 2. Duplicate title analysis
# --------------------------------------------------

print("\n--- TITLE DUPLICATES ---")

duplicate_title_rows = df["title"].duplicated(keep=False).sum()

print(
    f"Rows belonging to duplicated titles: "
    f"{duplicate_title_rows:,}"
)

unique_duplicate_titles = (
    df.loc[df["title"].duplicated(keep=False), "title"]
    .nunique()
)

print(
    f"Unique titles appearing multiple times: "
    f"{unique_duplicate_titles:,}"
)


# --------------------------------------------------
# 3. Same title + same text
# --------------------------------------------------

print("\n--- TITLE + TEXT DUPLICATES ---")

title_text_duplicates = df.duplicated(
    subset=["title", "text"],
    keep=False
).sum()

print(
    f"Rows with identical title AND text: "
    f"{title_text_duplicates:,}"
)


# --------------------------------------------------
# 4. Cross-class title + text duplicates
# --------------------------------------------------

print("\n--- CROSS-CLASS TITLE + TEXT CHECK ---")

title_text_labels = (
    df.groupby(["title", "text"])["label"]
    .nunique()
)

cross_class_title_text = (
    title_text_labels > 1
).sum()

print(
    f"Identical title + text appearing in both classes: "
    f"{cross_class_title_text:,}"
)


# --------------------------------------------------
# 5. Empty / very short text
# --------------------------------------------------

print("\n--- ARTICLE TEXT QUALITY ---")

text_words = df["text"].str.split().str.len()

print(f"Empty articles (0 words): {(text_words == 0).sum():,}")
print(f"< 10 words: {(text_words < 10).sum():,}")
print(f"< 20 words: {(text_words < 20).sum():,}")
print(f"< 50 words: {(text_words < 50).sum():,}")


# --------------------------------------------------
# 6. Date investigation
# --------------------------------------------------

print("\n--- DATE QUALITY ---")

# Convert dates using Pandas
parsed_dates = pd.to_datetime(
    df["date"],
    errors="coerce"
)

valid_dates = parsed_dates.notna()

print(f"Valid dates  : {valid_dates.sum():,}")
print(f"Invalid dates: {(~valid_dates).sum():,}")


print("\nEarliest valid date:")
print(parsed_dates.min())

print("\nLatest valid date:")
print(parsed_dates.max())


# Show unusual/invalid date examples
invalid_date_values = df.loc[
    ~valid_dates,
    "date"
].drop_duplicates()

print("\nExamples of invalid date values:")

if len(invalid_date_values) == 0:
    print("None")
else:
    for value in invalid_date_values.head(20):
        print(repr(value))


# --------------------------------------------------
# 7. Date validity by class
# --------------------------------------------------

print("\n--- DATE VALIDITY BY CLASS ---")

fake_dates = pd.to_datetime(
    fake_df["date"],
    errors="coerce"
)

real_dates = pd.to_datetime(
    true_df["date"],
    errors="coerce"
)

print(
    f"Fake invalid dates: "
    f"{fake_dates.isna().sum():,}"
)

print(
    f"Real invalid dates: "
    f"{real_dates.isna().sum():,}"
)


# --------------------------------------------------
# 8. Class distribution
# --------------------------------------------------

print("\n--- CLASS DISTRIBUTION ---")

class_counts = df["label"].value_counts()

print(
    f"FAKE: {class_counts[0]:,} "
    f"({class_counts[0] / len(df) * 100:.2f}%)"
)

print(
    f"REAL: {class_counts[1]:,} "
    f"({class_counts[1] / len(df) * 100:.2f}%)"
)


# --------------------------------------------------
# 9. Article length
# --------------------------------------------------

print("\n--- ARTICLE LENGTH ---")

print(
    text_words.describe().round(2)
)


# --------------------------------------------------
# 10. Length by class
# --------------------------------------------------

print("\n--- ARTICLE LENGTH BY CLASS ---")

print("\nFAKE:")
print(
    fake_df["text"]
    .str.split()
    .str.len()
    .describe()
    .round(2)
)

print("\nREAL:")
print(
    true_df["text"]
    .str.split()
    .str.len()
    .describe()
    .round(2)
)


# --------------------------------------------------
# Final
# --------------------------------------------------

print("\n" + "=" * 70)
print("DEEP DATA QUALITY ANALYSIS COMPLETE")
print("=" * 70)