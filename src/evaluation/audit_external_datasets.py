import re
import pandas as pd

from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "mendeley"
ISOT_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned_news.csv"

DATASETS = {
    "Fake News Dataset": EXTERNAL_DIR / "Fake News Dataset",
    "Fake News Detection Dataset": EXTERNAL_DIR / "Fake News Detection Dataset",
    "Fake or Real News Dataset": EXTERNAL_DIR / "Fake or Real News Dataset",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    """
    Normalize text for overlap detection.
    This is ONLY used for duplicate/overlap checking.
    It does not modify the actual evaluation data.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def load_dataset(dataset_dir):
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    train_df = pd.read_csv(train_path, sep=";")
    test_df = pd.read_csv(test_path, sep=";")

    train_df["split"] = "train"
    test_df["split"] = "test"

    df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    return df


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("EXTERNAL DATASET VALIDATION — AUDIT #6 PREPARATION")
print("=" * 80)

print("\nPurpose:")
print("Validate candidate external datasets before cross-domain evaluation.")
print("NO MODEL TRAINING IS PERFORMED IN THIS SCRIPT.")


# ============================================================
# LOAD ISOT
# ============================================================

print("\n" + "=" * 80)
print("LOADING ORIGINAL ISOT DATASET")
print("=" * 80)

isot_df = pd.read_csv(ISOT_PATH)

print(f"ISOT samples: {len(isot_df):,}")
print(f"ISOT columns: {list(isot_df.columns)}")

isot_text = (
    isot_df["content"]
    .fillna("")
    .apply(normalize_text)
)

isot_text_set = set(
    text for text in isot_text
    if text
)

print(f"Unique ISOT normalized texts: {len(isot_text_set):,}")


# ============================================================
# DATASET SUMMARY
# ============================================================

all_external = {}

print("\n" + "=" * 80)
print("EXTERNAL DATASET SUMMARY")
print("=" * 80)

for name, directory in DATASETS.items():

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    df = load_dataset(directory)

    all_external[name] = df

    print(f"Samples : {len(df):,}")
    print(f"Columns : {list(df.columns)}")

    print("\nLabel values:")

    if "label" in df.columns:
        print(
            df["label"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )
    else:
        print("WARNING: No 'label' column found.")

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .to_string()
    )

    if "text" not in df.columns:
        print("\nWARNING: No 'text' column found.")
        continue

    text = df["text"].fillna("").astype(str)

    empty_count = (
        text.str.strip() == ""
    ).sum()

    unique_count = text.apply(
        normalize_text
    ).nunique()

    duplicate_count = (
        len(text) - unique_count
    )

    word_counts = (
        text
        .apply(lambda x: len(x.split()))
    )

    print("\nText quality:")
    print(f"Empty text       : {empty_count:,}")
    print(f"Unique texts     : {unique_count:,}")
    print(f"Duplicate texts  : {duplicate_count:,}")
    print(f"Mean words       : {word_counts.mean():.2f}")
    print(f"Median words     : {word_counts.median():.2f}")
    print(f"Min words        : {word_counts.min():,}")
    print(f"Max words        : {word_counts.max():,}")


# ============================================================
# DUPLICATE ANALYSIS WITHIN EXTERNAL DATASETS
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE ANALYSIS WITHIN EXTERNAL DATASETS")
print("=" * 80)

for name, df in all_external.items():

    normalized = (
        df["text"]
        .fillna("")
        .apply(normalize_text)
    )

    duplicate_mask = (
        normalized.duplicated(
            keep=False
        )
        & normalized.ne("")
    )

    duplicate_rows = duplicate_mask.sum()

    unique_duplicate_texts = (
        normalized[
            duplicate_mask
        ]
        .nunique()
    )

    print(
        f"\n{name}:"
        f"\n  Rows in duplicate groups : {duplicate_rows:,}"
        f"\n  Unique duplicated texts  : {unique_duplicate_texts:,}"
    )


# ============================================================
# OVERLAP BETWEEN EXTERNAL DATASETS
# ============================================================

print("\n" + "=" * 80)
print("OVERLAP BETWEEN EXTERNAL DATASETS")
print("=" * 80)

dataset_names = list(all_external.keys())

normalized_sets = {}

for name, df in all_external.items():

    normalized_sets[name] = set(
        text
        for text in df["text"]
        .fillna("")
        .apply(normalize_text)
        if text
    )

for i in range(len(dataset_names)):

    for j in range(i + 1, len(dataset_names)):

        name_a = dataset_names[i]
        name_b = dataset_names[j]

        overlap = (
            normalized_sets[name_a]
            & normalized_sets[name_b]
        )

        print(
            f"\n{name_a} ↔ {name_b}"
        )

        print(
            f"Exact normalized text overlap: "
            f"{len(overlap):,}"
        )


# ============================================================
# OVERLAP WITH ORIGINAL ISOT
# ============================================================

print("\n" + "=" * 80)
print("OVERLAP WITH ORIGINAL ISOT DATASET")
print("=" * 80)

external_isot_overlap = {}

for name, text_set in normalized_sets.items():

    overlap = (
        text_set
        & isot_text_set
    )

    external_isot_overlap[name] = overlap

    print(
        f"\n{name}"
    )

    print(
        f"Exact normalized text overlap with ISOT: "
        f"{len(overlap):,}"
    )

    if overlap:

        print("\nExample overlapping texts:")

        for text in list(overlap)[:3]:

            preview = text[:250]

            print(
                f"  {preview}..."
            )


# ============================================================
# LABEL CONSISTENCY / CLASS BALANCE
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

for name, df in all_external.items():

    print(f"\n{name}")

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    for label, count in counts.items():

        percentage = (
            count / total * 100
        )

        print(
            f"Label {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print("""
IMPORTANT:

1. The ISOT dataset is excluded from external evaluation.
2. No model was trained in this audit.
3. External datasets will only be considered for evaluation.
4. Any dataset with substantial exact text overlap with ISOT
   must be investigated before being used as Dataset B.
5. Label semantics must be verified before evaluation.
6. We will not retrain or tune the existing SVM using these datasets.
""")

print("=" * 80)
print("EXTERNAL DATASET VALIDATION COMPLETE")
print("=" * 80)