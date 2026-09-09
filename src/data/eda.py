import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
true_df = pd.read_csv(DATA_DIR / "True.csv")

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)


# ============================================================
# FEATURE ENGINEERING FOR EDA
# ============================================================

df["title_words"] = df["title"].str.split().str.len()
df["text_words"] = df["text"].str.split().str.len()


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

class_counts = df["label"].value_counts()

plt.figure(figsize=(8, 6))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.title("Fake vs Real News Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Articles")

for i, value in enumerate(class_counts.values):
    plt.text(
        i,
        value + 300,
        f"{value:,}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. ARTICLE LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["text_words"],
    bins=100
)

plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

# Focus on the majority of articles
plt.xlim(0, 2000)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_article_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. TITLE LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["title_words"],
    bins=40
)

plt.title("Title Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_title_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. ARTICLE LENGTH BY CLASS
# ============================================================

fake_lengths = fake_df["text"].str.split().str.len()
real_lengths = true_df["text"].str.split().str.len()

plt.figure(figsize=(8, 6))

plt.boxplot(
    [fake_lengths, real_lengths],
    tick_labels=["FAKE", "REAL"],
    showfliers=False
)

plt.title("Article Length by Class")
plt.xlabel("Class")
plt.ylabel("Number of Words")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_article_length_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. SUBJECT DISTRIBUTION
# ============================================================

subject_counts = df["subject"].value_counts()

plt.figure(figsize=(10, 7))

plt.barh(
    subject_counts.index[::-1],
    subject_counts.values[::-1]
)

plt.title("News Articles by Subject")
plt.xlabel("Number of Articles")
plt.ylabel("Subject")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_subject_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. DUPLICATE CONTENT ANALYSIS
# ============================================================

exact_duplicates = df.duplicated(
    subset=["title", "text"]
).sum()

unique_articles = (
    len(df)
    - exact_duplicates
)

duplicate_values = [
    unique_articles,
    exact_duplicates
]

labels = [
    "Unique title + text",
    "Duplicate title + text"
]

plt.figure(figsize=(8, 6))

plt.bar(
    labels,
    duplicate_values
)

plt.title("Duplicate Article Content Analysis")
plt.xlabel("Content Type")
plt.ylabel("Number of Rows")

for i, value in enumerate(duplicate_values):
    plt.text(
        i,
        value + 200,
        f"{value:,}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_duplicate_content.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"\nTotal articles: {len(df):,}")

print("\nGenerated figures:")

for figure in sorted(FIGURE_DIR.glob("*.png")):
    print(f"  ✓ {figure.name}")

print("\nFigures saved to:")
print(FIGURE_DIR)

print("\n" + "=" * 70)