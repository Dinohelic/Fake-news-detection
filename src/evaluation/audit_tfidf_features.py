import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "linear_svm.joblib"

CSV_PATH = RESULTS_DIR / "tfidf_feature_importance.csv"
FIGURE_PATH = FIGURES_DIR / "11_tfidf_feature_importance.png"


# ============================================================
# SETTINGS
# ============================================================

TOP_N = 25


# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================

print("=" * 70)
print("BIAS AUDIT #2 — TF-IDF FEATURE IMPORTANCE")
print("=" * 70)

print("\nLoading TF-IDF vectorizer...")
vectorizer = joblib.load(VECTORIZER_PATH)

print("Loading Linear SVM...")
model = joblib.load(MODEL_PATH)

print(f"Number of TF-IDF features: {len(vectorizer.get_feature_names_out()):,}")


# ============================================================
# GET FEATURES + COEFFICIENTS
# ============================================================

features = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]


# ============================================================
# CREATE FEATURE TABLE
# ============================================================

feature_df = pd.DataFrame({
    "feature": features,
    "coefficient": coefficients
})


# Negative = FAKE
fake_features = (
    feature_df
    .sort_values(
        "coefficient",
        ascending=True
    )
    .head(TOP_N)
    .copy()
)

# Positive = REAL
real_features = (
    feature_df
    .sort_values(
        "coefficient",
        ascending=False
    )
    .head(TOP_N)
    .copy()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "-" * 70)
print(f"TOP {TOP_N} FEATURES ASSOCIATED WITH FAKE")
print("-" * 70)

print(
    fake_features[
        ["feature", "coefficient"]
    ].to_string(
        index=False
    )
)


print("\n" + "-" * 70)
print(f"TOP {TOP_N} FEATURES ASSOCIATED WITH REAL")
print("-" * 70)

print(
    real_features[
        ["feature", "coefficient"]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPLETE FEATURE TABLE
# ============================================================

feature_df_sorted = feature_df.sort_values(
    "coefficient"
)

feature_df_sorted.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plot_fake = fake_features.sort_values(
    "coefficient"
)

plot_real = real_features.sort_values(
    "coefficient"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 10)
)


# ------------------------------------------------------------
# FAKE FEATURES
# ------------------------------------------------------------

axes[0].barh(
    plot_fake["feature"],
    plot_fake["coefficient"]
)

axes[0].set_title(
    "Strongest FAKE-associated TF-IDF Features"
)

axes[0].set_xlabel(
    "SVM coefficient"
)


# ------------------------------------------------------------
# REAL FEATURES
# ------------------------------------------------------------

axes[1].barh(
    plot_real["feature"],
    plot_real["coefficient"]
)

axes[1].set_title(
    "Strongest REAL-associated TF-IDF Features"
)

axes[1].set_xlabel(
    "SVM coefficient"
)


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

print(f"Complete feature table:")
print(CSV_PATH)

print(f"\nVisualization:")
print(FIGURE_PATH)

print("\n" + "=" * 70)
print("TF-IDF FEATURE AUDIT COMPLETE")
print("=" * 70)