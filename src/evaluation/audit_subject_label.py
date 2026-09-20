import pandas as pd
import matplotlib.pyplot as plt

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

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("BIAS AUDIT #1 — SUBJECT ↔ LABEL ANALYSIS")
print("=" * 70)

print(f"\nTraining samples: {len(df):,}")


# ============================================================
# SUBJECT × LABEL COUNTS
# ============================================================

counts = pd.crosstab(
    df["subject"],
    df["label"]
)

counts = counts.rename(
    columns={
        0: "FAKE",
        1: "REAL"
    }
)

# Make sure both columns exist
for column in ["FAKE", "REAL"]:
    if column not in counts.columns:
        counts[column] = 0

counts = counts[
    ["FAKE", "REAL"]
]


# ============================================================
# PERCENTAGE WITHIN EACH SUBJECT
# ============================================================

percentages = counts.div(
    counts.sum(axis=1),
    axis=0
) * 100

percentages = percentages.rename(
    columns={
        "FAKE": "FAKE_%",
        "REAL": "REAL_%"
    }
)


# ============================================================
# COMBINE RESULTS
# ============================================================

summary = counts.join(
    percentages
)

summary["TOTAL"] = (
    summary["FAKE"] +
    summary["REAL"]
)

summary["FAKE_RATIO"] = (
    summary["FAKE"] /
    summary["TOTAL"]
)

summary = summary.sort_values(
    "FAKE_RATIO",
    ascending=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n--- SUBJECT DISTRIBUTION ---")

print(
    summary[
        [
            "FAKE",
            "REAL",
            "TOTAL",
            "FAKE_%",
            "REAL_%"
        ]
    ].to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# EXTREME SUBJECTS
# ============================================================

print("\n--- SUBJECT LABEL CONCENTRATION ---")

for subject, row in summary.iterrows():

    fake_pct = row["FAKE_%"]
    real_pct = row["REAL_%"]

    if fake_pct >= 90 or real_pct >= 90:

        dominant = (
            "FAKE"
            if fake_pct >= 90
            else "REAL"
        )

        print(
            f"⚠ {subject}: "
            f"{fake_pct:.2f}% FAKE / "
            f"{real_pct:.2f}% REAL "
            f"→ strongly associated with {dominant}"
        )


# ============================================================
# SAVE CSV
# ============================================================

output_csv = (
    RESULTS_DIR /
    "subject_label_analysis.csv"
)

summary.to_csv(
    output_csv
)


# ============================================================
# PLOT
# ============================================================

plot_data = summary[
    ["FAKE_%", "REAL_%"]
]

ax = plot_data.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7)
)

ax.set_title(
    "Subject-wise FAKE / REAL Distribution"
)

ax.set_xlabel(
    "Subject"
)

ax.set_ylabel(
    "Percentage (%)"
)

ax.set_ylim(
    0,
    100
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Class"
)

plt.tight_layout()


output_figure = (
    FIGURES_DIR /
    "10_subject_label_distribution.png"
)

plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n--- OUTPUT FILES ---")

print(
    f"CSV:\n{output_csv}"
)

print(
    f"\nFigure:\n{output_figure}"
)

print("\n" + "=" * 70)
print("SUBJECT ↔ LABEL AUDIT COMPLETE")
print("=" * 70)