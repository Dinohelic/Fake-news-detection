import pandas as pd
import joblib

from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "linear_svm.joblib"

RESULTS_PATH = RESULTS_DIR / "svm_results.txt"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BASELINE MODEL — TF-IDF + LINEAR SVM")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training samples   : {len(train_df):,}")
print(f"Validation samples : {len(val_df):,}")
print(f"Test samples       : {len(test_df):,}")


# ============================================================
# INPUT / TARGET
# ============================================================

X_train = train_df["content"].fillna("")
y_train = train_df["label"]

X_val = val_df["content"].fillna("")
y_val = val_df["label"]

X_test = test_df["content"].fillna("")
y_test = test_df["label"]


# ============================================================
# LOAD EXISTING TF-IDF VECTORIZER
# ============================================================

print("\n" + "-" * 70)
print("LOADING TF-IDF VECTORIZER")
print("-" * 70)

vectorizer = joblib.load(VECTORIZER_PATH)

print("TF-IDF vectorizer loaded.")


# ============================================================
# TRANSFORM DATA
# ============================================================

print("\nTransforming datasets...")

X_train_tfidf = vectorizer.transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training matrix shape   : {X_train_tfidf.shape}")
print(f"Validation matrix shape : {X_val_tfidf.shape}")
print(f"Test matrix shape       : {X_test_tfidf.shape}")


# ============================================================
# TRAIN LINEAR SVM
# ============================================================

print("\n" + "-" * 70)
print("TRAINING LINEAR SVM")
print("-" * 70)

model = LinearSVC(
    C=1.0,
    random_state=RANDOM_STATE
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Training complete.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(f"\n{name}")
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# ============================================================
# VALIDATION
# ============================================================

val_results = evaluate_model(
    "VALIDATION SET",
    X_val_tfidf,
    y_val
)


# ============================================================
# TEST
# ============================================================

test_results = evaluate_model(
    "TEST SET",
    X_test_tfidf,
    y_test
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "-" * 70)
print("TEST CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        model.predict(X_test_tfidf),
        target_names=["FAKE", "REAL"],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(RESULTS_PATH, "w") as f:

    f.write("TF-IDF + LINEAR SVM\n")
    f.write("=" * 60 + "\n\n")

    f.write("Model Configuration\n")
    f.write("-" * 60 + "\n")
    f.write("Algorithm: LinearSVC\n")
    f.write("C: 1.0\n")
    f.write("\n")

    f.write("Validation Results\n")
    f.write("-" * 60 + "\n")
    f.write(f"Accuracy : {val_results['accuracy']:.4f}\n")
    f.write(f"Precision: {val_results['precision']:.4f}\n")
    f.write(f"Recall   : {val_results['recall']:.4f}\n")
    f.write(f"F1 Score : {val_results['f1']:.4f}\n\n")

    f.write("Test Results\n")
    f.write("-" * 60 + "\n")
    f.write(f"Accuracy : {test_results['accuracy']:.4f}\n")
    f.write(f"Precision: {test_results['precision']:.4f}\n")
    f.write(f"Recall   : {test_results['recall']:.4f}\n")
    f.write(f"F1 Score : {test_results['f1']:.4f}\n\n")

    f.write("Test Confusion Matrix\n")
    f.write("-" * 60 + "\n")
    f.write(str(test_results["confusion_matrix"]))


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("LINEAR SVM BASELINE COMPLETE")
print("=" * 70)

print("\nSaved model:")
print(MODEL_PATH)

print("\nUsing existing TF-IDF vectorizer:")
print(VECTORIZER_PATH)

print("\nSaved results:")
print(RESULTS_PATH)