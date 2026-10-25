from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_text_only.joblib"
MODEL_PATH = MODEL_DIR / "linear_svm_text_only.joblib"


print("=" * 70)
print("PREPARING FROZEN TEXT-ONLY SVM")
print("=" * 70)


# Load original ISOT training data
df = pd.read_csv(TRAIN_FILE)

print(f"\nTraining samples: {len(df):,}")

X_train_text = df["text"].fillna("")
y_train = df["label"]


# TF-IDF configuration from Audit #4
print("\nTraining TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)

print(f"TF-IDF shape: {X_train.shape}")


# Linear SVM configuration from Audit #4
print("\nTraining Linear SVM...")

model = LinearSVC(
    C=1.0,
    random_state=42
)

model.fit(X_train, y_train)


# Save frozen artifacts
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(model, MODEL_PATH)


print("\n" + "=" * 70)
print("FROZEN MODEL READY")
print("=" * 70)

print(f"\nVectorizer:")
print(VECTORIZER_PATH)

print(f"\nModel:")
print(MODEL_PATH)

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nLabel mapping:")
print("0 = FAKE")
print("1 = REAL")

print("\nIMPORTANT:")
print("The vectorizer and SVM must NOT be refitted on external datasets.")

print("\nDone.")
