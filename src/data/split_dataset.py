import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cleaned_news.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("DATASET SPLITTING")
print("=" * 70)

print(f"\nTotal articles: {len(df):,}")


# ============================================================
# FIRST SPLIT
# Train = 80%
# Temporary = 20%
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_SIZE + TEST_SIZE),
    stratify=df["label"],
    random_state=RANDOM_STATE
)


# ============================================================
# SECOND SPLIT
# Validation = 10%
# Test = 10%
#
# temp = 20%
# Half of temp → validation
# Half of temp → test
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE DATASETS
# ============================================================

train_df.to_csv(
    TRAIN_PATH,
    index=False
)

val_df.to_csv(
    VAL_PATH,
    index=False
)

test_df.to_csv(
    TEST_PATH,
    index=False
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_distribution(name, data):
    counts = data["label"].value_counts()

    fake_count = counts.get(0, 0)
    real_count = counts.get(1, 0)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Total: {len(data):,}")
    print(
        f"FAKE: {fake_count:,} "
        f"({fake_count / len(data) * 100:.2f}%)"
    )
    print(
        f"REAL: {real_count:,} "
        f"({real_count / len(data) * 100:.2f}%)"
    )


# ============================================================
# DISTRIBUTION CHECK
# ============================================================

print_distribution("TRAINING SET", train_df)
print_distribution("VALIDATION SET", val_df)
print_distribution("TEST SET", test_df)


# ============================================================
# SIZE CHECK
# ============================================================

print("\n--- SPLIT SIZES ---")

print(
    f"Train:      {len(train_df):,} "
    f"({len(train_df) / len(df) * 100:.2f}%)"
)

print(
    f"Validation: {len(val_df):,} "
    f"({len(val_df) / len(df) * 100:.2f}%)"
)

print(
    f"Test:       {len(test_df):,} "
    f"({len(test_df) / len(df) * 100:.2f}%)"
)

print(
    f"Combined:   "
    f"{len(train_df) + len(val_df) + len(test_df):,}"
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n--- DATA LEAKAGE CHECK ---")

train_content = set(train_df["content"])
val_content = set(val_df["content"])
test_content = set(test_df["content"])

train_val_overlap = train_content.intersection(val_content)
train_test_overlap = train_content.intersection(test_content)
val_test_overlap = val_content.intersection(test_content)

print(
    f"Train ↔ Validation overlap: "
    f"{len(train_val_overlap):,}"
)

print(
    f"Train ↔ Test overlap: "
    f"{len(train_test_overlap):,}"
)

print(
    f"Validation ↔ Test overlap: "
    f"{len(val_test_overlap):,}"
)


# ============================================================
# FINAL STATUS
# ============================================================

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\n✓ No content leakage detected.")
else:
    print("\n⚠ WARNING: Content overlap detected.")


print("\n--- OUTPUT FILES ---")

print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETE")
print("=" * 70)