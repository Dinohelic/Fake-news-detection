from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "mendeley"

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_text_only.joblib"
MODEL_PATH = MODEL_DIR / "linear_svm_text_only.joblib"


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "Fake News Dataset": EXTERNAL_DIR / "Fake News Dataset" / "test.csv",
    "Fake News Detection Dataset": EXTERNAL_DIR / "Fake News Detection Dataset" / "test.csv",
    "Fake or Real News Dataset": EXTERNAL_DIR / "Fake or Real News Dataset" / "test.csv",
}


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

print("=" * 80)
print("AUDIT #6 — CROSS-DOMAIN EVALUATION")
print("=" * 80)

print("\nLoading frozen model...")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

print("✓ TF-IDF vectorizer loaded")
print("✓ Linear SVM loaded")

print("\nIMPORTANT:")
print("No fitting or tuning will be performed on external datasets.")


# ============================================================
# EVALUATION
# ============================================================

all_results = []

for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 80)
    print(dataset_name)
    print("=" * 80)

    # Mendeley CSV files are semicolon-separated
    df = pd.read_csv(file_path, sep=";")

    print(f"\nOriginal test samples: {len(df):,}")

    # Keep only required columns
    df = df[["text", "label"]].copy()

    # Remove missing text
    df["text"] = df["text"].fillna("").astype(str)

    before_dedup = len(df)

    # Normalize only for duplicate detection
    normalized = (
        df["text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df.loc[~normalized.duplicated()].copy()

    duplicates_removed = before_dedup - len(df)

    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Final evaluation samples: {len(df):,}")

    # Labels
    y_true = df["label"].astype(int)

    print("\nLabel distribution:")
    print(y_true.value_counts().sort_index())

    # --------------------------------------------------------
    # TRANSFORM ONLY
    # --------------------------------------------------------

    X_external = vectorizer.transform(df["text"])

    print(f"\nTF-IDF shape: {X_external.shape}")

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(X_external)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nResults:")
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               FAKE    REAL")
    print(f"Actual FAKE   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"Actual REAL   {cm[1,0]:5d}   {cm[1,1]:5d}")

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    all_results.append({
        "dataset": dataset_name,
        "original_samples": before_dedup,
        "duplicates_removed": duplicates_removed,
        "evaluation_samples": len(df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fake_correct": cm[0, 0],
        "fake_incorrect": cm[0, 1],
        "real_incorrect": cm[1, 0],
        "real_correct": cm[1, 1],
    })


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(all_results)

output_csv = RESULTS_DIR / "cross_domain_results.csv"

results_df.to_csv(output_csv, index=False)

print("\n" + "=" * 80)
print("CROSS-DOMAIN SUMMARY")
print("=" * 80)

print(
    results_df[
        [
            "dataset",
            "evaluation_samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)

print(f"\nSaved results:")
print(output_csv)


# ============================================================
# COMPARISON PLOT
# ============================================================

metrics = ["accuracy", "precision", "recall", "f1"]

x = range(len(results_df))

plt.figure(figsize=(12, 7))

for metric in metrics:
    plt.plot(
        x,
        results_df[metric],
        marker="o",
        label=metric.upper(),
    )

plt.xticks(
    list(x),
    results_df["dataset"],
    rotation=15,
    ha="right",
)

plt.ylim(0, 1.05)

plt.ylabel("Score")
plt.title("Cross-Domain Evaluation — Frozen Text-Only Linear SVM")

plt.legend()
plt.tight_layout()

plot_path = FIGURES_DIR / "14_cross_domain_evaluation.png"

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nSaved figure:")
print(plot_path)

print("\n" + "=" * 80)
print("AUDIT #6 COMPLETE")
print("=" * 80)
