# ============================================================
# Biochar Suitability Mapping — Ghayas Haider Sajid
# Quaid-i-Azam University, Islamabad, Pakistan
# MPhil Thesis | Under review: Soil and Tillage Research
# ============================================================
# Script 03: Machine Learning Model Training and Evaluation
# Description: Trains four ML algorithms (Random Forest,
#              XGBoost, Decision Tree, Bagging Regressor)
#              on the preprocessed feature stack. Evaluates
#              performance using R², RMSE, MAE, and NSE via
#              5-fold cross-validation and a held-out test set.
#              Saves the best model for downstream SHAP analysis
#              and suitability mapping.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, KFold, cross_val_score
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor, BaggingRegressor
from sklearn.metrics           import r2_score, mean_squared_error, mean_absolute_error
from xgboost                   import XGBRegressor

OUTPUT_DIR  = "outputs/"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_FOLDS      = 5

# ── 1. Load feature stack ─────────────────────────────────────
print("Loading feature stack...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "feature_stack.csv"))

# Target variable: biochar suitability index (continuous 0–1)
# This should be a field-validated or literature-derived score
# in your actual dataset. Adjust the column name as needed.
TARGET_COL   = "suitability_index"
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]:,}")
print(f"  Target range: {y.min():.3f} – {y.max():.3f}\n")


# ── 2. Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples\n")


# ── 3. Define models ──────────────────────────────────────────
# Hyperparameters are tuned based on cross-validation.
# For full hyperparameter search, see the comments below
# each model block.
MODELS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE
        # For GridSearchCV: param_grid = {
        #   "n_estimators": [100, 200, 300],
        #   "max_depth": [None, 10, 20],
        #   "max_features": ["sqrt", "log2"]
        # }
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0
    ),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ),
    "Bagging Regressor": BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=12),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}


# ── 4. Evaluation metrics ─────────────────────────────────────
def nash_sutcliffe_efficiency(y_true, y_pred):
    """
    Nash-Sutcliffe Efficiency (NSE): standard metric in
    hydrological and environmental modelling.
    NSE = 1 → perfect model | NSE < 0 → worse than mean.
    """
    numerator   = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    """
    Trains a model and returns a dict of performance metrics
    on both the cross-validated training set and the test set.
    """
    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2  = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2")

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return {
        "Model":         name,
        "CV R² (mean)":  cv_r2.mean().round(4),
        "CV R² (std)":   cv_r2.std().round(4),
        "Test R²":       round(r2_score(y_te, y_pred), 4),
        "Test RMSE":     round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
        "Test MAE":      round(mean_absolute_error(y_te, y_pred), 4),
        "Test NSE":      round(nash_sutcliffe_efficiency(y_te, y_pred), 4),
        "_y_pred":       y_pred   # stored for plotting, dropped from table
    }


# ── 5. Train and evaluate all models ─────────────────────────
print("Training and evaluating models...")
print("=" * 66)

results      = []
trained_models = {}
predictions    = {}

for model_name, model in MODELS.items():
    print(f"  {model_name}...")
    res = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
    predictions[model_name] = res.pop("_y_pred")
    results.append(res)
    trained_models[model_name] = model
    joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name.replace(' ', '_')}.pkl"))

results_df = pd.DataFrame(results).set_index("Model")
print("\nModel Performance Summary")
print("=" * 66)
print(results_df.to_string())
print("=" * 66)

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance.csv"))
print("\nResults saved: outputs/model_performance.csv")


# ── 6. Identify best model ────────────────────────────────────
best_model_name = results_df["Test R²"].idxmax()
best_model      = trained_models[best_model_name]
best_r2         = results_df.loc[best_model_name, "Test R²"]

print(f"\nBest model: {best_model_name}  (Test R² = {best_r2})")
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
print("Best model saved: outputs/models/best_model.pkl")


# ── 7. Scatter plots: observed vs predicted ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
colors_map = {
    "Random Forest":    "#2A6EBB",
    "XGBoost":          "#E05A2B",
    "Decision Tree":    "#4CAF50",
    "Bagging Regressor":"#8B4EC8"
}

for ax, (model_name, y_pred) in zip(axes, predictions.items()):
    r2   = results_df.loc[model_name, "Test R²"]
    rmse = results_df.loc[model_name, "Test RMSE"]

    ax.scatter(y_test, y_pred, alpha=0.4, s=14,
               color=colors_map[model_name], edgecolors="none")

    lim = [min(y_test.min(), y_pred.min()) - 0.02,
           max(y_test.max(), y_pred.max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=0.9, label="1:1 line")
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_title(f"{model_name}\nR² = {r2}  |  RMSE = {rmse}",
                 fontsize=10, pad=6)
    ax.set_xlabel("Observed suitability index", fontsize=9)
    ax.set_ylabel("Predicted suitability index", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

plt.suptitle(
    "Biochar Suitability Mapping — Observed vs Predicted\n"
    "Quaid-i-Azam University | Ghayas Haider Sajid",
    fontsize=11, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_scatter.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("Scatter plots saved: outputs/figures/model_comparison_scatter.png")


# ── 8. Metric comparison bar chart ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics    = ["Test R²", "Test RMSE", "Test NSE"]
bar_colors = list(colors_map.values())

for ax, metric in zip(axes, metrics):
    values = results_df[metric].values
    bars   = ax.bar(results_df.index, values, color=bar_colors,
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(metric, fontsize=10)
    ax.set_xticks(range(len(results_df.index)))
    ax.set_xticklabels(results_df.index, rotation=15, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle("Model Performance Comparison", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_performance_bars.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("Performance chart saved: outputs/figures/model_performance_bars.png")
print("\nModel training and evaluation complete.")
