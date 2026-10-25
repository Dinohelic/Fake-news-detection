from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("PHASE 3.2 — BERT DATA PIPELINE")
print("=" * 70)

print(f"\nModel: {MODEL_NAME}")
print(f"Maximum sequence length: {MAX_LENGTH}")

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✓ Tokenizer loaded")


# ============================================================
# LOAD DATASETS
# ============================================================

def load_split(filename):
    import pandas as pd

    path = DATA_DIR / filename

    df = pd.read_csv(path)

    # Use the already-created content field.
    df = df[["content", "label"]].copy()

    df["content"] = df["content"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(
        df,
        preserve_index=False
    )


print("\nLoading datasets...")

train_dataset = load_split("train.csv")
validation_dataset = load_split("validation.csv")
test_dataset = load_split("test.csv")

print(f"Train      : {len(train_dataset):,}")
print(f"Validation : {len(validation_dataset):,}")
print(f"Test       : {len(test_dataset):,}")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_batch(batch):
    return tokenizer(
        batch["content"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("\nTokenizing datasets...")

tokenized_train = train_dataset.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing train"
)

tokenized_validation = validation_dataset.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing validation"
)

tokenized_test = test_dataset.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing test"
)

print("✓ Tokenization complete")


# ============================================================
# SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECK")
print("=" * 70)

sample = tokenized_train[0]

print("\nAvailable fields:")
print(tokenized_train.column_names)

print("\nFirst sample:")
print(f"Label: {sample['label']}")

print(f"Token count: {len(sample['input_ids'])}")

print("\nFirst 20 token IDs:")
print(sample["input_ids"][:20])

print("\nDecoded beginning:")
print(
    tokenizer.decode(
        sample["input_ids"][:50],
        skip_special_tokens=False
    )
)

print("\nExpected label mapping:")
print("0 = FAKE")
print("1 = REAL")

print("\n" + "=" * 70)
print("PHASE 3.2 PIPELINE CHECK COMPLETE")
print("=" * 70)

print("\nNo model training was performed.")
