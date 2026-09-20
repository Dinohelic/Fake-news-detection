import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "source_marker_analysis.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MARKERS TO INVESTIGATE
# ============================================================

MARKERS = [
    "reuters",
    "getty",
    "getty images",
    "read more",
    "via",
    "featured image",
    "image via",
    "video",
    "video article",
    "washington reuters",
    "article reuters",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["content"] = (
    df["content"]
    .fillna("")
    .str.lower()
)

df["title"] = (
    df["title"]
    .fillna("")
    .str.lower()
)

df["text"] = (
    df["text"]
    .fillna("")
    .str.lower()
)


# ============================================================
# ANALYSIS
# ============================================================

rows = []

for marker in MARKERS:

    content_contains = df["content"].str.contains(
        marker,
        regex=False
    )

    title_contains = df["title"].str.contains(
        marker,
        regex=False
    )

    text_contains = df["text"].str.contains(
        marker,
        regex=False
    )

    fake_mask = df["label"] == 0
    real_mask = df["label"] == 1

    fake_total = fake_mask.sum()
    real_total = real_mask.sum()

    fake_count = (
        content_contains & fake_mask
    ).sum()

    real_count = (
        content_contains & real_mask
    ).sum()

    fake_pct = (
        fake_count / fake_total * 100
    )

    real_pct = (
        real_count / real_total * 100
    )

    rows.append({
        "marker": marker,
        "FAKE_count": fake_count,
        "REAL_count": real_count,
        "FAKE_%": fake_pct,
        "REAL_%": real_pct,
        "title_occurrences": title_contains.sum(),
        "text_occurrences": text_contains.sum(),
        "total_occurrences": content_contains.sum(),
    })


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(rows)

results = results.sort_values(
    "total_occurrences",
    ascending=False
)


# ============================================================
# PRINT
# ============================================================

print("=" * 70)
print("BIAS AUDIT #3 — SOURCE / FORMATTING MARKER ANALYSIS")
print("=" * 70)

print(
    "\nMarker frequency by class:\n"
)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved results:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("SOURCE MARKER AUDIT COMPLETE")
print("=" * 70)