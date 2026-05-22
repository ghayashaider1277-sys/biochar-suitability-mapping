# ============================================================
# Biochar Suitability Mapping — Ghayas Haider Sajid
# Quaid-i-Azam University, Islamabad, Pakistan
# MPhil Thesis | Under review: Soil and Tillage Research
# ============================================================
# Script 02: AHP Criteria Weighting
# Description: Implements the Analytical Hierarchy Process
#              (AHP) for objective weighting of soil and
#              environmental criteria. Computes the pairwise
#              comparison matrix, derives priority weights via
#              eigenvector method, and validates consistency
#              using the Consistency Ratio (CR < 0.10).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Define criteria ───────────────────────────────────────
# Ten soil and environmental criteria for biochar suitability.
# Order must match rows and columns of the pairwise matrix.
CRITERIA = [
    "Soil Organic Carbon",
    "Soil pH",
    "Bulk Density",
    "Clay Content",
    "Cation Exchange Capacity",
    "Slope",
    "TWI",
    "NDVI",
    "Annual Rainfall",
    "Mean Temperature"
]
N = len(CRITERIA)

# ── 2. Pairwise comparison matrix ───────────────────────────
# Values represent relative importance on Saaty's 1–9 scale:
#   1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme
# Upper triangle entries are expert-informed; lower triangle
# is filled as reciprocals automatically below.
# References: Saaty (1980); Bandaru et al. (2021, Geoderma)

UPPER_TRIANGLE = np.array([
    # SOC   pH    BD    Clay  CEC   Slope TWI   NDVI  Rain  Temp
    [  1,    2,    3,    2,    2,    5,    5,    4,    3,    4  ],  # SOC
    [  0,    1,    2,    1,    2,    4,    4,    3,    3,    3  ],  # pH
    [  0,    0,    1,   0.5,   1,    3,    3,    2,    2,    2  ],  # BD
    [  0,    0,    0,    1,    1,    3,    3,    2,    2,    2  ],  # Clay
    [  0,    0,    0,    0,    1,    3,    4,    2,    2,    3  ],  # CEC
    [  0,    0,    0,    0,    0,    1,    1,   0.5,  0.5,   1  ],  # Slope
    [  0,    0,    0,    0,    0,    0,    1,   0.5,  0.5,   1  ],  # TWI
    [  0,    0,    0,    0,    0,    0,    0,    1,    1,    2  ],  # NDVI
    [  0,    0,    0,    0,    0,    0,    0,    0,    1,    1  ],  # Rainfall
    [  0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ],  # Temp
], dtype=float)


def build_pairwise_matrix(upper):
    """
    Constructs a full symmetric pairwise matrix from the upper
    triangle. Lower triangle entries are reciprocals of the
    corresponding upper triangle values.
    """
    matrix = upper.copy()
    for i in range(N):
        for j in range(N):
            if i > j:
                matrix[i, j] = 1.0 / matrix[j, i]
    return matrix


# ── 3. Compute AHP weights via eigenvector method ────────────
def compute_ahp_weights(matrix):
    """
    Derives priority weights using the principal eigenvector
    method (Saaty, 1980). Steps:
      1. Normalise each column by its sum.
      2. Average each row of the normalised matrix.
    Returns the weight vector (sums to 1.0).
    """
    col_sums      = matrix.sum(axis=0)
    norm_matrix   = matrix / col_sums
    weights       = norm_matrix.mean(axis=1)
    return weights


# ── 4. Consistency check ─────────────────────────────────────
# Random Index (RI) values from Saaty (1980) for n = 1..10
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def compute_consistency_ratio(matrix, weights):
    """
    Computes the Consistency Ratio (CR).
    CR = CI / RI, where CI = (λ_max - n) / (n - 1).
    A CR < 0.10 indicates acceptable consistency.
    """
    weighted_sum = matrix @ weights
    lambda_vec   = weighted_sum / weights
    lambda_max   = lambda_vec.mean()
    ci           = (lambda_max - N) / (N - 1)
    ri           = RI_TABLE.get(N, 1.49)
    cr           = ci / ri
    return lambda_max, ci, cr


# ── 5. Run AHP ───────────────────────────────────────────────
print("Running AHP weight calculation...")
print(f"Number of criteria: {N}\n")

pairwise_matrix = build_pairwise_matrix(UPPER_TRIANGLE)
weights         = compute_ahp_weights(pairwise_matrix)
lambda_max, ci, cr = compute_consistency_ratio(pairwise_matrix, weights)

# ── 6. Report results ────────────────────────────────────────
print("=" * 52)
print(f"{'Criterion':<30} {'Weight':>8} {'Weight %':>10}")
print("-" * 52)
results = sorted(zip(CRITERIA, weights), key=lambda x: -x[1])
for criterion, weight in results:
    print(f"{criterion:<30} {weight:>8.4f} {weight*100:>9.2f}%")
print("-" * 52)
print(f"{'Total':<30} {weights.sum():>8.4f} {weights.sum()*100:>9.2f}%")
print("=" * 52)
print(f"\nConsistency check:")
print(f"  λ_max = {lambda_max:.4f}")
print(f"  CI    = {ci:.4f}")
print(f"  RI    = {RI_TABLE[N]:.2f}  (n = {N})")
print(f"  CR    = {cr:.4f}  {'✓ Acceptable (< 0.10)' if cr < 0.10 else '✗ Revise pairwise judgements'}")

if cr >= 0.10:
    raise ValueError(
        f"CR = {cr:.4f} exceeds the 0.10 threshold. "
        "Revise pairwise comparison judgements before proceeding."
    )

# ── 7. Save weights to CSV ───────────────────────────────────
weights_df = pd.DataFrame({
    "criterion": CRITERIA,
    "weight":    weights,
    "weight_pct": weights * 100
}).sort_values("weight", ascending=False).reset_index(drop=True)

weights_df.to_csv(os.path.join(OUTPUT_DIR, "ahp_weights.csv"), index=False)
print(f"\nWeights saved: outputs/ahp_weights.csv")

# Save full pairwise matrix
matrix_df = pd.DataFrame(pairwise_matrix, index=CRITERIA, columns=CRITERIA)
matrix_df.to_csv(os.path.join(OUTPUT_DIR, "ahp_pairwise_matrix.csv"))
print("Pairwise matrix saved: outputs/ahp_pairwise_matrix.csv")

# ── 8. Plot weight bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sorted_criteria = weights_df["criterion"].tolist()
sorted_weights  = weights_df["weight_pct"].tolist()
colors = plt.cm.YlOrBr(np.linspace(0.3, 0.85, len(sorted_criteria)))

bars = ax.barh(sorted_criteria[::-1], sorted_weights[::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.6)

for bar, val in zip(bars, sorted_weights[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9)

ax.set_xlabel("AHP Weight (%)", fontsize=11)
ax.set_title(
    "AHP Criteria Weights for Biochar Suitability Assessment\n"
    f"CR = {cr:.4f} (Acceptable < 0.10)",
    fontsize=11, pad=12
)
ax.set_xlim(0, max(sorted_weights) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figures", "ahp_weights.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("Weight chart saved: outputs/figures/ahp_weights.png")
print("\nAHP weighting complete.")
