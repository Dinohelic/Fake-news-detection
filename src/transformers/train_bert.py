from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "validation.csv"

OUTPUT_DIR = PROJECT_ROOT / "models" / "transformers" / "bert"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ============================================================
# HARDWARE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 70)
print("PHASE 3.3 — BERT TRAINING")
print("=" * 70)

print(f"\nModel          : {MODEL_NAME}")
print(f"Max length     : {MAX_LENGTH}")
print(f"Device         : {DEVICE}")
print(f"Seed           : {SEED}")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df = df[["content", "label"]].copy()

    df["content"] = df["content"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(
        df,
        preserve_index=False
    )


print("\nLoading datasets...")

train_dataset = load_dataset(TRAIN_FILE)
validation_dataset = load_dataset(VAL_FILE)

print(f"Train      : {len(train_dataset):,}")
print(f"Validation : {len(validation_dataset):,}")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✓ Tokenizer loaded")


def tokenize(batch):
    return tokenizer(
        batch["content"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("\nTokenizing...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["content"],
    desc="Tokenizing train"
)

validation_dataset = validation_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["content"],
    desc="Tokenizing validation"
)

print("✓ Tokenization complete")


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if DEVICE == "mps" else None,
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "FAKE",
        1: "REAL",
    },
    label2id={
        "FAKE": 0,
        "REAL": 1,
    },
)

print("✓ BERT classification model loaded")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    # Training
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,

    # Physical batch size
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    # Effective batch size = 4 × 4 = 16
    gradient_accumulation_steps=4,

    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",

    # Select best checkpoint using validation F1
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,

    # Storage
    save_total_limit=1,

    # Logging
    logging_strategy="steps",
    logging_steps=100,

    # Reproducibility
    seed=SEED,
    data_seed=SEED,

    # Performance
    dataloader_num_workers=0,

    # Disable external experiment tracking
    report_to="none",

    # MPS does not currently benefit from CUDA-specific settings
    fp16=False,
    bf16=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING BERT TRAINING")
print("=" * 70)

print("\nTraining configuration:")
print("Epochs                  : 2")
print("Learning rate           : 2e-5")
print("Physical batch size     : 4")
print("Gradient accumulation   : 4")
print("Effective batch size    : 16")
print("Weight decay            : 0.01")
print("Warmup                  : Not explicitly configured")
print("Best model metric       : Validation F1")
print("Test set                : NOT USED")
print("=" * 70)

trainer.train()


# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING BEST BERT MODEL")
print("=" * 70)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\nModel saved to:")
print(OUTPUT_DIR)

print("\nBest checkpoint:")
print(trainer.state.best_model_checkpoint)

print("\nBest validation metric:")
print(trainer.state.best_metric)

print("\n" + "=" * 70)
print("PHASE 3.3 BERT TRAINING COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("The test set has not been used.")
print("External datasets have not been used.")
print("No external-data tuning was performed.")
