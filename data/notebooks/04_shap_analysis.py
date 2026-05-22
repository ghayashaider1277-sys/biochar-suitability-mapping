# ============================================================
# Biochar Suitability Mapping — Ghayas Haider Sajid
# Quaid-i-Azam University, Islamabad, Pakistan
# MPhil Thesis | Under review: Soil and Tillage Research
# ============================================================
# Script 04: SHAP Interpretability Analysis
# Description: Applies SHapley Additive exPlanations (SHAP)
#              to the best-performing ML model to quantify
#              each feature's contribution to predictions.
#              Produces beeswarm summary plots, mean absolute
#              SHAP bar charts, and dependence plots for the
#              top three drivers of biochar suitability.
# Reference:   Lundberg & Lee (2017), NeurIPS 30.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = "outputs/"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE  = 42
SAMPLE_SIZE   = 2000   # SHAP is computed on a subsample for speed


# ── 1. Load data and best model ───────────────────────────────
print("Loading data and best model...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "feature_stack.csv"))

TARGET_COL   = "suitability_index"
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

# Human-readable feature labels for plots
FEATURE_LABELS = {
    "soc":          "Soil Organic Carbon",
    "ph":           "Soil pH",
    "bulk_density": "Bulk Density",
    "clay":         "Clay Content",
    "cec":          "Cation Exchange Capacity",
    "slope":        "Slope",
    "twi":          "Topographic Wetness Index",
    "ndvi":         "NDVI",
    "rainfall":     "Annual Rainfall",
    "temperature":  "Mean Temperature",
    "lulc":         "Land Use / Land Cover"
}

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Subsample for SHAP computation (representative random sample)
np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
X_sample   = X.iloc[sample_idx].copy()
X_sample.columns = [FEATURE_LABELS.get(c, c) for c in X_sample.columns]

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
print(f"  Model loaded. Sample size for SHAP: {len(X_sample):,}\n")


# ── 2. Compute SHAP values ────────────────────────────────────
print("Computing SHAP values (this may take a few minutes)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
print(f"  SHAP values computed. Shape: {np.array(shap_values).shape}\n")

# Save SHAP values for reproducibility
shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_values.csv"), index=False)
print("  SHAP values saved: outputs/shap_values.csv")


# ── 3. Mean absolute SHAP — feature importance ranking ───────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature":         X_sample.columns,
    "Mean |SHAP|":     mean_abs_shap
}).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

print("\nSHAP Feature Importance Ranking")
print("=" * 45)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Mean |SHAP|"] / mean_abs_shap.max() * 30)
    print(f"  {row['Feature']:<32} {bar}")
print("=" * 45)


# ── 4. Beeswarm summary plot ──────────────────────────────────
print("\nGenerating beeswarm summary plot...")
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(
    shap_values, X_sample,
    plot_type="dot",
    max_display=10,
    show=False,
    color_bar_label="Feature value (normalised)"
)
plt.title(
    "SHAP Summary — Biochar Suitability Drivers\n"
    "Colour = normalised feature value  |  x-axis = impact on prediction",
    fontsize=10, pad=10
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_beeswarm.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_beeswarm.png")


# ── 5. Mean absolute SHAP bar chart ──────────────────────────
print("Generating mean |SHAP| bar chart...")
top_n   = 10
top_df  = importance_df.head(top_n)
colors  = plt.cm.YlOrBr(np.linspace(0.3, 0.85, top_n))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(
    top_df["Feature"][::-1],
    top_df["Mean |SHAP|"][::-1],
    color=colors[::-1], edgecolor="white", linewidth=0.5
)
for bar, val in zip(bars, top_df["Mean |SHAP|"][::-1]):
    ax.text(bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8)

ax.set_xlabel("Mean |SHAP value|", fontsize=10)
ax.set_title(
    "Feature Importance by Mean Absolute SHAP Value\n"
    "Biochar Suitability Mapping — Ghayas Haider Sajid",
    fontsize=10, pad=10
)
ax.set_xlim(0, top_df["Mean |SHAP|"].max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_bar_importance.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_bar_importance.png")


# ── 6. Dependence plots — top 3 drivers ──────────────────────
print("Generating dependence plots for top 3 features...")
top3_features = importance_df["Feature"].head(3).tolist()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feature in zip(axes, top3_features):
    feat_idx = list(X_sample.columns).index(feature)
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_sample,
        interaction_index="auto",
        ax=ax,
        show=False,
        dot_size=10,
        alpha=0.5
    )
    ax.set_title(feature, fontsize=10, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

plt.suptitle(
    "SHAP Dependence Plots — Top 3 Suitability Drivers\n"
    "Colour shows interaction feature (auto-selected by SHAP)",
    fontsize=10, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_dependence_top3.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_dependence_top3.png")


# ── 7. Waterfall plot — single prediction explanation ─────────
print("Generating waterfall plot (single prediction)...")
explanation = shap.Explanation(
    values        = shap_values[0],
    base_values   = explainer.expected_value,
    data          = X_sample.iloc[0].values,
    feature_names = X_sample.columns.tolist()
)
fig, ax = plt.subplots(figsize=(8, 5))
shap.waterfall_plot(explanation, max_display=10, show=False)
plt.title("SHAP Waterfall — Single Pixel Prediction Explanation",
          fontsize=10, pad=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_waterfall_sample.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_waterfall_sample.png")

print("\nSHAP analysis complete.")
print(f"Top driver: {importance_df.iloc[0]['Feature']}  "
      f"(Mean |SHAP| = {importance_df.iloc[0]['Mean |SHAP|']:.4f})")
