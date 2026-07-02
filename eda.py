"""
Student Performance – Exploratory Data Analysis
================================================
Run:  python eda.py
All plots saved to  visuals/eda/
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PALETTE    = ["#6C63FF", "#FF6584", "#43D8B0", "#FFA552", "#3A86FF", "#FF006E"]
BG_DARK    = "#0F0F1A"
BG_CARD    = "#1A1A2E"
TEXT_COLOR = "#E0E0F0"
GRID_COLOR = "#2A2A4A"
VIS_DIR    = "visuals/eda"
os.makedirs(VIS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_CARD,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.4,
    "legend.facecolor":  BG_CARD,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/StudentsPerformance.csv")
print(f"Shape: {df.shape}")
print(df.head(3).to_string())

# ── Feature engineering (needed for EDA too) ──────────────────────────────────
df["avg_score"]     = df[["math_score", "reading_score", "writing_score"]].mean(axis=1).round(2)
df["score_std"]     = df[["math_score", "reading_score", "writing_score"]].std(axis=1).round(2)
df["log_absences"]  = np.log1p(df["absences"])

# Ordinal encode parental education
edu_order = ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"]
df["edu_level"] = df["parental_level_of_education"].map(
    {v: i for i, v in enumerate(edu_order)}
)

# Pass/Fail flag (avg >= 60)
df["passed"] = (df["avg_score"] >= 60).astype(int)

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass balance (passed): {df['passed'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Score Distributions (3 subjects + avg)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Score Distributions by Subject", fontsize=18, fontweight="bold",
             color=TEXT_COLOR, y=1.01)

subjects = ["math_score", "reading_score", "writing_score", "avg_score"]
labels   = ["Math Score", "Reading Score", "Writing Score", "Average Score"]
colors   = PALETTE[:4]

for ax, col, label, color in zip(axes.flat, subjects, labels, colors):
    data = df[col]
    ax.hist(data, bins=28, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.5,
               label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color="#FFD700", linestyle=":", linewidth=1.5,
               label=f"Median={data.median():.1f}")
    ax.set_title(label, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Skewness annotation
    sk = stats.skew(data)
    ax.text(0.97, 0.95, f"Skew={sk:.2f}", transform=ax.transAxes,
            ha="right", va="top", color="#FFD700", fontsize=9)

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/01_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_score_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
num_cols = ["weekly_study_hours", "absences", "edu_level",
            "math_score", "reading_score", "writing_score", "avg_score"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG_DARK)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5,
            linecolor=BG_DARK, annot_kws={"size": 9, "color": TEXT_COLOR},
            ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold",
             color=TEXT_COLOR, pad=15)
ax.tick_params(colors=TEXT_COLOR)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig(f"{VIS_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Average Score by Categorical Features (2x2 grid)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Average Score by Demographic & Academic Factors",
             fontsize=17, fontweight="bold", color=TEXT_COLOR, y=1.01)

cat_configs = [
    ("gender",                    "Gender",              PALETTE[:2]),
    ("lunch",                     "Lunch Type",          PALETTE[1:3]),
    ("test_preparation_course",   "Test Prep Course",    PALETTE[2:4]),
    ("race_ethnicity",            "Race/Ethnicity",      PALETTE),
]

for ax, (col, title, colors) in zip(axes.flat, cat_configs):
    grp = df.groupby(col)["avg_score"].mean().sort_values(ascending=False)
    bars = ax.bar(grp.index, grp.values, color=colors[:len(grp)],
                  edgecolor="white", linewidth=0.5, width=0.55)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Mean Avg Score", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                color=TEXT_COLOR, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/03_avg_score_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_avg_score_by_category.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Study Hours vs Avg Score (scatter + regression)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Study Hours & Absences vs. Academic Performance",
             fontsize=16, fontweight="bold", color=TEXT_COLOR)

# Scatter: study hours vs avg score
ax = axes[0]
sc = ax.scatter(df["weekly_study_hours"], df["avg_score"],
                c=df["avg_score"], cmap="viridis", alpha=0.55, s=25, linewidths=0)
m, b, r, p, _ = stats.linregress(df["weekly_study_hours"], df["avg_score"])
x_line = np.linspace(df["weekly_study_hours"].min(), df["weekly_study_hours"].max(), 200)
ax.plot(x_line, m*x_line + b, color="#FF6584", linewidth=2.5, label=f"r={r:.3f}")
ax.set_xlabel("Weekly Study Hours", fontsize=11)
ax.set_ylabel("Average Score", fontsize=11)
ax.set_title("Study Hours vs Average Score", fontsize=13, color=TEXT_COLOR)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.colorbar(sc, ax=ax, label="Avg Score")

# Scatter: absences vs avg score
ax = axes[1]
sc2 = ax.scatter(df["absences"], df["avg_score"],
                 c=df["avg_score"], cmap="plasma", alpha=0.55, s=25, linewidths=0)
m2, b2, r2, _, _ = stats.linregress(df["absences"], df["avg_score"])
x2 = np.linspace(0, df["absences"].max(), 200)
ax.plot(x2, m2*x2 + b2, color="#43D8B0", linewidth=2.5, label=f"r={r2:.3f}")
ax.set_xlabel("Absences", fontsize=11)
ax.set_ylabel("Average Score", fontsize=11)
ax.set_title("Absences vs Average Score", fontsize=13, color=TEXT_COLOR)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.colorbar(sc2, ax=ax, label="Avg Score")

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/04_study_absences_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_study_absences_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Parental Education vs Avg Score (ordered bar)
# ═══════════════════════════════════════════════════════════════════════════════
edu_means = df.groupby("parental_level_of_education")["avg_score"].mean().reindex(edu_order)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(edu_means.index, edu_means.values,
               color=sns.color_palette("viridis", len(edu_means)),
               edgecolor="white", linewidth=0.4, height=0.55)
ax.set_xlabel("Mean Average Score", fontsize=11)
ax.set_title("Average Score by Parental Level of Education",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)
ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, edu_means.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=11, fontweight="bold", color=TEXT_COLOR)
plt.tight_layout()
fig.savefig(f"{VIS_DIR}/05_parental_education_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_parental_education_score.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Box plots: Score by test prep × gender
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
fig.suptitle("Score Distribution by Test Prep Course & Gender",
             fontsize=16, fontweight="bold", color=TEXT_COLOR)

subjects2 = ["math_score", "reading_score", "writing_score"]
labels2   = ["Math", "Reading", "Writing"]

for ax, col, lbl in zip(axes, subjects2, labels2):
    sub = df[[col, "test_preparation_course", "gender"]]
    groups  = ["none", "completed"]
    genders = ["female", "male"]
    positions = [1, 2, 4, 5]
    box_data  = [
        sub[(sub["test_preparation_course"] == tp) & (sub["gender"] == g)][col]
        for tp in groups for g in genders
    ]
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    medianprops=dict(color="#FFD700", linewidth=2))
    colors_bp = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
    for patch, c in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["No Prep", "Completed Prep"], fontsize=10)
    ax.set_title(f"{lbl} Score", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("Score" if lbl == "Math" else "", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

# Legend
legend_patches = [
    mpatches.Patch(color=PALETTE[0], label="Female – No Prep"),
    mpatches.Patch(color=PALETTE[1], label="Male – No Prep"),
    mpatches.Patch(color=PALETTE[2], label="Female – Completed"),
    mpatches.Patch(color=PALETTE[3], label="Male – Completed"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=10, framealpha=0.2, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
fig.savefig(f"{VIS_DIR}/06_boxplot_prep_gender.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_boxplot_prep_gender.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Absences Distribution (original vs log-transformed)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Absences: Before and After Log Transformation",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)

ax = axes[0]
ax.hist(df["absences"], bins=20, color=PALETTE[0], alpha=0.85,
        edgecolor="white", linewidth=0.4)
sk0 = stats.skew(df["absences"])
ax.set_title(f"Raw Absences  (skew={sk0:.2f})", fontsize=12, color=TEXT_COLOR)
ax.set_xlabel("Absences"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.hist(df["log_absences"], bins=20, color=PALETTE[2], alpha=0.85,
        edgecolor="white", linewidth=0.4)
sk1 = stats.skew(df["log_absences"])
ax.set_title(f"log1p(Absences)  (skew={sk1:.2f})", fontsize=12, color=TEXT_COLOR)
ax.set_xlabel("log1p(Absences)"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/07_absences_log_transform.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_absences_log_transform.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EDA SUMMARY STATISTICS")
print("="*60)
print(f"Total records       : {len(df)}")
print(f"Missing values      : {df.isnull().sum().sum()}")
print(f"Pass rate (avg>=60) : {df['passed'].mean()*100:.1f}%")
print(f"Mean avg score      : {df['avg_score'].mean():.2f}")
print(f"Std avg score       : {df['avg_score'].std():.2f}")
print(f"Study-score r       : {stats.pearsonr(df['weekly_study_hours'], df['avg_score'])[0]:.3f}")
print(f"Absence-score r     : {stats.pearsonr(df['absences'], df['avg_score'])[0]:.3f}")
print(f"Edu level-score r   : {stats.pearsonr(df['edu_level'], df['avg_score'])[0]:.3f}")
print("="*60)
print("\nAll EDA visuals saved to:", VIS_DIR)
