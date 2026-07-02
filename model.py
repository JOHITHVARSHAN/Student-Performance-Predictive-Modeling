"""
Student Performance – Predictive Modeling Pipeline
===================================================
Models : Linear Regression, Random Forest, Gradient Boosting
Target : avg_score (mean of math / reading / writing)
Run    : python model.py
Outputs: visuals/model/  +  metrics.txt
"""

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.preprocessing    import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import Ridge
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics          import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection       import permutation_importance

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PALETTE    = ["#6C63FF", "#FF6584", "#43D8B0", "#FFA552", "#3A86FF", "#FF006E"]
BG_DARK    = "#0F0F1A"
BG_CARD    = "#1A1A2E"
TEXT_COLOR = "#E0E0F0"
GRID_COLOR = "#2A2A4A"
VIS_DIR    = "visuals/model"
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

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("data/StudentsPerformance.csv")

# Composite features
df["avg_score"]    = df[["math_score", "reading_score", "writing_score"]].mean(axis=1).round(2)
df["score_range"]  = (df[["math_score", "reading_score", "writing_score"]].max(axis=1) -
                      df[["math_score", "reading_score", "writing_score"]].min(axis=1))
df["log_absences"] = np.log1p(df["absences"])

# ── Features & target ─────────────────────────────────────────────────────────
TARGET = "avg_score"
CAT_ORD = ["parental_level_of_education"]   # ordinal
CAT_NOM = ["gender", "race_ethnicity", "lunch", "test_preparation_course"]  # nominal
NUM     = ["weekly_study_hours", "log_absences"]

FEATURE_COLS = CAT_ORD + CAT_NOM + NUM
X = df[FEATURE_COLS]
y = df[TARGET]

# ── Train / test split (80 / 20, stratified by score decile) ──────────────────
y_decile = pd.qcut(y, q=10, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_decile
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════
edu_order  = ["some high school", "high school", "some college",
              "associate's degree", "bachelor's degree", "master's degree"]

preprocessor = ColumnTransformer(transformers=[
    ("ord",  OrdinalEncoder(categories=[edu_order]), CAT_ORD),
    ("nom",  OneHotEncoder(drop="first", sparse_output=False), CAT_NOM),
    ("num",  StandardScaler(), NUM),
], remainder="drop")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BASELINE — mean prediction
# ═══════════════════════════════════════════════════════════════════════════════
baseline_pred  = np.full(len(y_test), y_train.mean())
baseline_rmse  = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_mae   = mean_absolute_error(y_test, baseline_pred)
baseline_r2    = r2_score(y_test, baseline_pred)
print(f"\nBaseline RMSE={baseline_rmse:.3f}  MAE={baseline_mae:.3f}  R2={baseline_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODELS
# ═══════════════════════════════════════════════════════════════════════════════
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest":    RandomForestRegressor(n_estimators=300, max_depth=10,
                                              min_samples_leaf=4, random_state=42,
                                              n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                   learning_rate=0.05,
                                                   subsample=0.8, random_state=42),
}

results    = {}
pipelines  = {}
kf         = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    # 5-fold CV on train set
    cv_r2   = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
    cv_rmse = cross_val_score(pipe, X_train, y_train, cv=kf,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
    # Fit on full train, evaluate on test
    pipe.fit(X_train, y_train)
    y_pred   = pipe.predict(X_test)
    residuals = y_test.values - y_pred

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    err_reduction = (baseline_rmse - rmse) / baseline_rmse * 100

    results[name] = {
        "r2":              r2,
        "rmse":            rmse,
        "mae":             mae,
        "cv_r2_mean":      cv_r2.mean(),
        "cv_r2_std":       cv_r2.std(),
        "cv_rmse_mean":    -cv_rmse.mean(),
        "err_reduction_%": err_reduction,
        "y_pred":          y_pred,
        "residuals":       residuals,
    }
    pipelines[name] = pipe
    print(f"\n{name}")
    print(f"  CV  R2={cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"  Test R2={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    print(f"  Error reduction vs baseline: {err_reduction:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE (best model = Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]["r2"])
best_pipe = pipelines[best_name]
best_model = best_pipe.named_steps["model"]

# Get feature names after preprocessing
pre = best_pipe.named_steps["pre"]
ord_names  = CAT_ORD
nom_names  = list(pre.named_transformers_["nom"].get_feature_names_out(CAT_NOM))
num_names  = NUM
all_feat   = ord_names + nom_names + num_names

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    # Permutation importance fallback
    X_test_pre = pre.transform(X_test)
    perm = permutation_importance(best_model, X_test_pre, y_test, n_repeats=20, random_state=42)
    importances = perm.importances_mean

imp_df = (pd.DataFrame({"Feature": all_feat, "Importance": importances})
          .sort_values("Importance", ascending=False)
          .head(12))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALS
# ═══════════════════════════════════════════════════════════════════════════════

# ── FIGURE A — Model Comparison Bar Chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Model Comparison: R², RMSE, Error Reduction",
             fontsize=16, fontweight="bold", color=TEXT_COLOR)

model_names = list(results.keys())
r2_vals  = [results[n]["r2"]              for n in model_names]
rmse_vals = [results[n]["rmse"]           for n in model_names]
err_vals  = [results[n]["err_reduction_%"] for n in model_names]

colors_bar = PALETTE[:3]

for ax, vals, title, ylabel, fmt in zip(
    axes,
    [r2_vals, rmse_vals, err_vals],
    ["R² Score", "RMSE", "Error Reduction vs Baseline (%)"],
    ["R²", "RMSE (points)", "% Reduction"],
    [".4f", ".3f", ".1f"]
):
    bars = ax.bar(model_names, vals, color=colors_bar,
                  edgecolor="white", linewidth=0.5, width=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=11,
                color=TEXT_COLOR, fontweight="bold")

# Baseline reference on RMSE plot
axes[1].axhline(baseline_rmse, color="#FFD700", linestyle="--", linewidth=1.5,
                label=f"Baseline={baseline_rmse:.2f}")
axes[1].legend(fontsize=9)
# Add 0% baseline on error reduction plot
axes[2].axhline(0, color="#FFD700", linestyle="--", linewidth=1.5)

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/A_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: A_model_comparison.png")

# ── FIGURE B — Feature Importance ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors_imp = sns.color_palette("viridis", len(imp_df))
bars = ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
               color=list(reversed(colors_imp)), edgecolor="white", linewidth=0.4,
               height=0.65)
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title(f"Feature Importance — {best_name}",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, imp_df["Importance"][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9, color=TEXT_COLOR)
plt.tight_layout()
fig.savefig(f"{VIS_DIR}/B_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: B_feature_importance.png")

# ── FIGURE C — Predicted vs Actual (best model) ───────────────────────────────
best_pred = results[best_name]["y_pred"]
fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(y_test, best_pred, c=results[best_name]["residuals"],
                cmap="coolwarm", alpha=0.6, s=30, linewidths=0)
mn = min(y_test.min(), best_pred.min()) - 2
mx = max(y_test.max(), best_pred.max()) + 2
ax.plot([mn, mx], [mn, mx], color="#FFD700", linewidth=2, linestyle="--",
        label="Perfect prediction")
ax.set_xlabel("Actual Score", fontsize=12)
ax.set_ylabel("Predicted Score", fontsize=12)
ax.set_title(f"Predicted vs Actual — {best_name}\n"
             f"R²={results[best_name]['r2']:.4f}  RMSE={results[best_name]['rmse']:.3f}",
             fontsize=14, fontweight="bold", color=TEXT_COLOR)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.colorbar(sc, ax=ax, label="Residual")
plt.tight_layout()
fig.savefig(f"{VIS_DIR}/C_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: C_predicted_vs_actual.png")

# ── FIGURE D — Residuals Histogram + QQ plot ──────────────────────────────────
best_resid = results[best_name]["residuals"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Residual Analysis — {best_name}",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)

ax = axes[0]
ax.hist(best_resid, bins=28, color=PALETTE[0], alpha=0.85,
        edgecolor="white", linewidth=0.4, density=True)
x_norm = np.linspace(best_resid.min(), best_resid.max(), 200)
ax.plot(x_norm, stats.norm.pdf(x_norm, best_resid.mean(), best_resid.std()),
        color="#FFD700", linewidth=2, label="Normal PDF")
ax.axvline(0, color="#FF6584", linewidth=1.5, linestyle="--", label="Zero")
ax.set_xlabel("Residual (Actual - Predicted)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Residuals Distribution", fontsize=13, color=TEXT_COLOR)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
# Annotate stats
ax.text(0.02, 0.95,
        f"Mean={best_resid.mean():.3f}\nStd={best_resid.std():.3f}\n"
        f"Skew={stats.skew(best_resid):.3f}\nKurtosis={stats.kurtosis(best_resid):.3f}",
        transform=ax.transAxes, va="top", fontsize=9, color=TEXT_COLOR,
        bbox=dict(facecolor=BG_DARK, alpha=0.7, edgecolor=GRID_COLOR))

ax = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(best_resid, dist="norm")
ax.scatter(osm, osr, color=PALETTE[2], alpha=0.6, s=20)
ref_x = np.array([osm.min(), osm.max()])
ax.plot(ref_x, slope*ref_x + intercept, color="#FFD700", linewidth=2,
        label=f"Fit line (r={r:.4f})")
ax.set_xlabel("Theoretical Quantiles", fontsize=11)
ax.set_ylabel("Sample Quantiles", fontsize=11)
ax.set_title("Q-Q Plot", fontsize=13, color=TEXT_COLOR)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/D_residuals_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: D_residuals_analysis.png")

# ── FIGURE E — Cross-Validation R² comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title("5-Fold Cross-Validation R² by Model",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)

cv_means = [results[n]["cv_r2_mean"] for n in model_names]
cv_stds  = [results[n]["cv_r2_std"]  for n in model_names]
x_pos    = np.arange(len(model_names))

bars = ax.bar(x_pos, cv_means, color=colors_bar, edgecolor="white",
              linewidth=0.5, width=0.45, zorder=3)
ax.errorbar(x_pos, cv_means, yerr=[2*s for s in cv_stds],
            fmt="none", color=TEXT_COLOR, capsize=7, linewidth=2, zorder=4)
ax.set_xticks(x_pos); ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Cross-Validated R²", fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3, zorder=0)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cv_stds)*2 + 0.02,
            f"{mean:.4f}\n±{std:.4f}", ha="center", fontsize=10,
            color=TEXT_COLOR, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/E_cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: E_cross_validation.png")

# ── FIGURE F — All-model Predicted vs Actual comparison (3-panel) ─────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
fig.suptitle("Predicted vs Actual: All Models", fontsize=16,
             fontweight="bold", color=TEXT_COLOR)

for ax, name, color in zip(axes, model_names, colors_bar):
    pred = results[name]["y_pred"]
    ax.scatter(y_test, pred, color=color, alpha=0.5, s=20, linewidths=0)
    lim = [min(y_test.min(), pred.min())-2, max(y_test.max(), pred.max())+2]
    ax.plot(lim, lim, color="#FFD700", linewidth=2, linestyle="--")
    ax.set_xlabel("Actual Score", fontsize=10)
    ax.set_ylabel("Predicted Score" if name == model_names[0] else "", fontsize=10)
    ax.set_title(f"{name}\nR²={results[name]['r2']:.4f}  RMSE={results[name]['rmse']:.3f}",
                 fontsize=12, color=TEXT_COLOR)
    ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(f"{VIS_DIR}/F_all_models_pva.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: F_all_models_pva.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. METRICS REPORT
# ═══════════════════════════════════════════════════════════════════════════════
report_lines = [
    "=" * 62,
    "  STUDENT PERFORMANCE PREDICTIVE MODELING – METRICS REPORT",
    "=" * 62,
    f"\nDataset   : 1000 students,  10 features",
    f"Target    : avg_score (mean of math / reading / writing)",
    f"Split     : 80% train ({len(X_train)})  |  20% test ({len(X_test)})",
    f"\n--- BASELINE (mean predictor) ---",
    f"  RMSE = {baseline_rmse:.4f}",
    f"  MAE  = {baseline_mae:.4f}",
    f"  R2   = {baseline_r2:.4f}",
]

for name in model_names:
    r = results[name]
    report_lines += [
        f"\n--- {name.upper()} ---",
        f"  Test  R2   = {r['r2']:.4f}",
        f"  Test  RMSE = {r['rmse']:.4f}",
        f"  Test  MAE  = {r['mae']:.4f}",
        f"  CV    R2   = {r['cv_r2_mean']:.4f} +/- {r['cv_r2_std']:.4f}",
        f"  Error reduction vs baseline = {r['err_reduction_%']:.1f}%",
    ]

best_r = results[best_name]
report_lines += [
    "\n" + "=" * 62,
    f"  BEST MODEL: {best_name}",
    f"  R2={best_r['r2']:.4f}  RMSE={best_r['rmse']:.4f}  "
    f"Err.Reduction={best_r['err_reduction_%']:.1f}%",
    "=" * 62,
    "\nRESUME BULLET:",
    f'  "Developed and validated {best_name} predicting student',
    f"   academic outcomes with R2={best_r['r2']:.2f}, RMSE={best_r['rmse']:.2f};",
    f"   reduced prediction error by {best_r['err_reduction_%']:.0f}% vs baseline",
    '   mean predictor; implemented full end-to-end ML pipeline."',
    "\n" + "=" * 62,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("\nMetrics saved to: metrics.txt")
print(f"Model visuals saved to: {VIS_DIR}/")
