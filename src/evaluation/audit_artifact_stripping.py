import re
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

RESULTS_PATH = RESULTS_DIR / "artifact_stripping_results.csv"
FIGURE_PATH = FIGURES_DIR / "13_artifact_stripping_comparison.png"

RANDOM_STATE = 42

print("=" * 70)
print("BIAS AUDIT #5 — ARTIFACT-STRIPPED EVALUATION")
print("=" * 70)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTraining samples   : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")
print(f"Test samples       : {len(test_df):,}")


# ---------------------------------------------------------
# Artifact removal
# ---------------------------------------------------------

ARTIFACT_PATTERNS = [
    r"\breuters\b",
    r"\bvia\b",
    r"\bfeatured image\b",
    r"\bimage via\b",
    r"\bgetty images\b",
    r"\bgetty\b",
    r"\bread more\b",
    r"\bvideo article\b",
    r"\bvideo\b"
]


def remove_artifacts(text):
    """
    Remove selected source/formatting markers identified
    during previous bias audits.
    """

    text = str(text)

    for pattern in ARTIFACT_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_content(df):
    """
    Recreate the same title + text representation used
    in the original SVM experiment.
    """

    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)

    return (
        "Title: "
        + title
        + " Article: "
        + text
    )


train_original = prepare_content(train_df)
val_original = prepare_content(val_df)
test_original = prepare_content(test_df)

train_stripped = train_original.apply(remove_artifacts)
val_stripped = val_original.apply(remove_artifacts)
test_stripped = test_original.apply(remove_artifacts)


# ---------------------------------------------------------
# Compare artifact frequency
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("ARTIFACT OCCURRENCES")
print("-" * 70)

for pattern in ARTIFACT_PATTERNS:

    name = pattern.replace(r"\b", "").replace("\\", "")

    train_count = train_original.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).sum()

    stripped_count = train_stripped.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).sum()

    print(
        f"{name:<20} "
        f"Original: {train_count:>6,}   "
        f"After stripping: {stripped_count:>6,}"
    )


# ---------------------------------------------------------
# Experiments
# ---------------------------------------------------------

experiments = {
    "Original": {
        "train": train_original,
        "val": val_original,
        "test": test_original
    },

    "Artifact-Stripped": {
        "train": train_stripped,
        "val": val_stripped,
        "test": test_stripped
    }
}


y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

results = []


# ---------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------

for name, data in experiments.items():

    print("\n" + "-" * 70)
    print(f"EXPERIMENT: {name}")
    print("-" * 70)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True
    )

    print("Fitting TF-IDF on training data only...")

    X_train = vectorizer.fit_transform(data["train"])
    X_val = vectorizer.transform(data["val"])
    X_test = vectorizer.transform(data["test"])

    print(f"Training matrix   : {X_train.shape}")
    print(f"Validation matrix : {X_val.shape}")
    print(f"Test matrix       : {X_test.shape}")

    print("Training Linear SVM...")

    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    # Validation
    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    # Test
    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    print("\nValidation:")
    print(f"Accuracy : {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall   : {val_recall:.4f}")
    print(f"F1 Score : {val_f1:.4f}")

    print("\nTest:")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

    print("\nTest Confusion Matrix:")
    print(cm)

    results.append({
        "Input": name,

        "Validation Accuracy": val_accuracy,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall,
        "Validation F1": val_f1,

        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1,

        "FAKE Correct": cm[0, 0],
        "FAKE → REAL": cm[0, 1],
        "REAL → FAKE": cm[1, 0],
        "REAL Correct": cm[1, 1]
    })


# ---------------------------------------------------------
# Results
# ----------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("ARTIFACT STRIPPING COMPARISON")
print("=" * 70)

display_columns = [
    "Input",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

print(
    results_df[display_columns].to_string(
        index=False,
        formatters={
            "Test Accuracy": "{:.4f}".format,
            "Test Precision": "{:.4f}".format,
            "Test Recall": "{:.4f}".format,
            "Test F1": "{:.4f}".format
        }
    )
)


# ---------------------------------------------------------
# Calculate performance change
# ---------------------------------------------------------

original = results_df.iloc[0]
stripped = results_df.iloc[1]

accuracy_change = (
    stripped["Test Accuracy"]
    - original["Test Accuracy"]
)

f1_change = (
    stripped["Test F1"]
    - original["Test F1"]
)

print("\n" + "-" * 70)
print("PERFORMANCE CHANGE")
print("-" * 70)

print(
    f"Accuracy change : {accuracy_change:+.4f} "
    f"({accuracy_change * 100:+.2f} percentage points)"
)

print(
    f"F1 change       : {f1_change:+.4f} "
    f"({f1_change * 100:+.2f} percentage points)"
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

metrics = [
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

for i, row in results_df.iterrows():

    positions = [
        value + (i - 0.5) * width
        for value in x
    ]

    values = [
        row[metric]
        for metric in metrics
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=row["Input"]
    )

plt.xticks(
    list(x),
    ["Accuracy", "Precision", "Recall", "F1"]
)

plt.ylim(
    max(0.0, results_df["Test Accuracy"].min() - 0.10),
    1.00
)

plt.ylabel("Score")
plt.title("Artifact Stripping — Linear SVM")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Done
# ---------------------------------------------------------

print("\n--- OUTPUT FILES ---")
print(f"Results:\n{RESULTS_PATH}")
print(f"\nFigure:\n{FIGURE_PATH}")

print("\n" + "=" * 70)
print("ARTIFACT STRIPPING AUDIT COMPLETE")
print("=" * 70)