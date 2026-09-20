import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

LOGISTIC_RESULTS = RESULTS_DIR / "logistic_regression_results.txt"
SVM_RESULTS = RESULTS_DIR / "svm_results.txt"

COMPARISON_CSV = RESULTS_DIR / "baseline_comparison.csv"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "TF-IDF + Logistic Regression",
        "TF-IDF + Linear SVM"
    ],
    "Accuracy": [
        0.9905,
        0.9964
    ],
    "Precision": [
        0.9855,
        0.9953
    ],
    "Recall": [
        0.9972,
        0.9981
    ],
    "F1 Score": [
        0.9913,
        0.9967
    ]
})


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

results.to_csv(
    COMPARISON_CSV,
    index=False
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)

print("\nTest-set performance:\n")

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# ============================================================
# METRIC COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    results.iloc[0][metrics],
    width=width,
    label="Logistic Regression"
)

plt.bar(
    [i + width / 2 for i in x],
    results.iloc[1][metrics],
    width=width,
    label="Linear SVM"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylim(
    0.97,
    1.00
)

plt.ylabel("Score")

plt.title(
    "TF-IDF Baseline Model Comparison"
)

plt.legend()

plt.tight_layout()

comparison_figure = (
    FIGURES_DIR /
    "07_baseline_model_comparison.png"
)

plt.savefig(
    comparison_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

logistic_cm = [
    [1759, 31],
    [6, 2114]
]

svm_cm = [
    [1780, 10],
    [4, 2116]
]


def save_confusion_matrix(
    matrix,
    title,
    filename
):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["FAKE", "REAL"]
    )

    plt.yticks(
        [0, 1],
        ["FAKE", "REAL"]
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.title(
        title
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_confusion_matrix(
    logistic_cm,
    "Logistic Regression — Test Confusion Matrix",
    "08_logistic_regression_confusion_matrix.png"
)

save_confusion_matrix(
    svm_cm,
    "Linear SVM — Test Confusion Matrix",
    "09_svm_confusion_matrix.png"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n--- OUTPUT FILES ---")

print(
    f"Comparison table:\n{COMPARISON_CSV}"
)

print(
    f"\nComparison chart:\n"
    f"{comparison_figure}"
)

print(
    f"\nConfusion matrices saved in:\n"
    f"{FIGURES_DIR}"
)

print("\n" + "=" * 70)
print("BASELINE COMPARISON COMPLETE")
print("=" * 70)