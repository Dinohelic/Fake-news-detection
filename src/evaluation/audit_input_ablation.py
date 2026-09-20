import pandas as pd
import joblib
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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

RESULTS_PATH = (
    RESULTS_DIR /
    "input_ablation_results.csv"
)

FIGURE_PATH = (
    FIGURES_DIR /
    "12_input_ablation_comparison.png"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_FEATURES = 100_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BIAS AUDIT #4 — TITLE vs TEXT vs TITLE + TEXT")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTraining samples   : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")
print(f"Test samples       : {len(test_df):,}")


# ============================================================
# INPUT VARIANTS
# ============================================================

input_variants = {
    "Title Only": {
        "train": train_df["title"].fillna(""),
        "val": val_df["title"].fillna(""),
        "test": test_df["title"].fillna("")
    },

    "Text Only": {
        "train": train_df["text"].fillna(""),
        "val": val_df["text"].fillna(""),
        "test": test_df["text"].fillna("")
    },

    "Title + Text": {
        "train": train_df["content"].fillna(""),
        "val": val_df["content"].fillna(""),
        "test": test_df["content"].fillna("")
    }
}


# ============================================================
# TARGET
# ============================================================

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# TRAIN / EVALUATE EACH INPUT VARIANT
# ============================================================

for name, data in input_variants.items():

    print("\n" + "-" * 70)
    print(f"EXPERIMENT: {name}")
    print("-" * 70)

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True
    )

    print("Fitting TF-IDF on training data only...")

    X_train = vectorizer.fit_transform(
        data["train"]
    )

    X_val = vectorizer.transform(
        data["val"]
    )

    X_test = vectorizer.transform(
        data["test"]
    )

    print(
        f"Training matrix   : {X_train.shape}"
    )

    print(
        f"Validation matrix : {X_val.shape}"
    )

    print(
        f"Test matrix       : {X_test.shape}"
    )


    # --------------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------------

    print("Training Linear SVM...")

    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = model.predict(
        X_val
    )

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


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

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


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nValidation:")
    print(
        f"Accuracy : {val_accuracy:.4f}"
    )
    print(
        f"Precision: {val_precision:.4f}"
    )
    print(
        f"Recall   : {val_recall:.4f}"
    )
    print(
        f"F1 Score : {val_f1:.4f}"
    )

    print("\nTest:")
    print(
        f"Accuracy : {test_accuracy:.4f}"
    )
    print(
        f"Precision: {test_precision:.4f}"
    )
    print(
        f"Recall   : {test_recall:.4f}"
    )
    print(
        f"F1 Score : {test_f1:.4f}"
    )

    print("\nTest Confusion Matrix:")
    print(cm)


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

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


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("INPUT ABLATION COMPARISON")
print("=" * 70)

display_columns = [
    "Input",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "Test Accuracy": "{:.4f}".format,
            "Test Precision": "{:.4f}".format,
            "Test Recall": "{:.4f}".format,
            "Test F1": "{:.4f}".format
        }
    )
)


# ============================================================
# SAVE CSV
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# PLOT
# ============================================================

metrics = [
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

x = range(len(metrics))
width = 0.25

plt.figure(
    figsize=(11, 6)
)

for i, row in results_df.iterrows():

    values = [
        row[metric]
        for metric in metrics
    ]

    positions = [
        value + (i - 1) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=row["Input"]
    )


plt.xticks(
    list(x),
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

plt.ylim(
    0.90,
    1.00
)

plt.ylabel(
    "Score"
)

plt.title(
    "Input Ablation — Linear SVM"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n--- OUTPUT FILES ---")

print(
    f"Results:\n{RESULTS_PATH}"
)

print(
    f"\nFigure:\n{FIGURE_PATH}"
)

print("\n" + "=" * 70)
print("INPUT ABLATION AUDIT COMPLETE")
print("=" * 70)