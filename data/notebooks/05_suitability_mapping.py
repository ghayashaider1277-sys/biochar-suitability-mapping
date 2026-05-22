# ============================================================
# Biochar Suitability Mapping — Ghayas Haider Sajid
# Quaid-i-Azam University, Islamabad, Pakistan
# MPhil Thesis | Under review: Soil and Tillage Research
# ============================================================
# Script 05: Suitability Map Generation
# Description: Applies the best-trained ML model to the full
#              preprocessed feature stack to generate a
#              continuous biochar suitability index across
#              Pakistan. Classifies the continuous index into
#              five suitability classes, exports the result
#              as a GeoTIFF, and produces a publication-ready
#              map figure with legend and area statistics.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import rasterio
from rasterio.plot import show
import geopandas as gpd
import joblib
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = "outputs/"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MAPS_DIR    = os.path.join(OUTPUT_DIR, "maps")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MAPS_DIR,    exist_ok=True)

# ── 1. Suitability classification scheme ─────────────────────
# Five classes following FAO land suitability framework,
# adapted for biochar application in agricultural soils.
SUITABILITY_CLASSES = {
    1: {"label": "Highly suitable",    "range": (0.80, 1.00), "color": "#2D6A4F"},
    2: {"label": "Moderately suitable","range": (0.60, 0.80), "color": "#74C69D"},
    3: {"label": "Marginally suitable","range": (0.40, 0.60), "color": "#FFD166"},
    4: {"label": "Currently unsuitable","range":(0.20, 0.40), "color": "#F4845F"},
    5: {"label": "Permanently unsuitable","range":(0.00,0.20),"color": "#C1121F"},
}


# ── 2. Load model and reference raster metadata ──────────────
print("Loading best model and reference raster...")
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

# Use preprocessed SOC raster as reference for spatial metadata
ref_raster_path = os.path.join(OUTPUT_DIR, "reproj_soc.tif")
with rasterio.open(ref_raster_path) as ref:
    ref_meta    = ref.meta.copy()
    ref_shape   = (ref.height, ref.width)
    ref_nodata  = ref.nodata


# ── 3. Load all feature rasters and build prediction array ───
print("Loading feature rasters for prediction...")

FEATURE_ORDER = [
    "soc", "ph", "bulk_density", "clay", "cec",
    "slope", "twi", "ndvi", "rainfall", "temperature", "lulc"
]

feature_arrays = []
valid_mask     = None

for feat in FEATURE_ORDER:
    path = os.path.join(OUTPUT_DIR, f"reproj_{feat}.tif")
    with rasterio.open(path) as src:
        arr    = src.read(1).astype(np.float32)
        nodata = src.nodata
        arr    = np.where(arr == nodata, np.nan, arr)
    feature_arrays.append(arr.flatten())

    mask = ~np.isnan(arr.flatten())
    valid_mask = mask if valid_mask is None else (valid_mask & mask)

print(f"  Valid pixels: {valid_mask.sum():,} of {valid_mask.size:,}")

X_full = np.column_stack(feature_arrays)
X_valid = X_full[valid_mask]


# ── 4. Predict suitability index across full extent ───────────
print("Predicting suitability index (full spatial extent)...")
suitability_flat        = np.full(valid_mask.size, np.nan, dtype=np.float32)
suitability_flat[valid_mask] = model.predict(X_valid).astype(np.float32)
suitability_2d          = suitability_flat.reshape(ref_shape)
print(f"  Prediction range: {np.nanmin(suitability_2d):.3f} – {np.nanmax(suitability_2d):.3f}")


# ── 5. Classify into suitability classes ─────────────────────
print("Classifying into suitability zones...")
classified_2d = np.full(ref_shape, 0, dtype=np.uint8)

for class_id, info in SUITABILITY_CLASSES.items():
    lo, hi = info["range"]
    mask   = (suitability_2d >= lo) & (suitability_2d < hi) & ~np.isnan(suitability_2d)
    classified_2d[mask] = class_id

# Class 1 upper bound is inclusive
classified_2d[(suitability_2d == 1.0) & ~np.isnan(suitability_2d)] = 1


# ── 6. Export GeoTIFFs ────────────────────────────────────────
# Continuous suitability index
cont_meta = ref_meta.copy()
cont_meta.update({"dtype": "float32", "nodata": -9999, "count": 1})
cont_path = os.path.join(MAPS_DIR, "biochar_suitability_index.tif")
suitability_2d_out = np.where(np.isnan(suitability_2d), -9999, suitability_2d)

with rasterio.open(cont_path, "w", **cont_meta) as dst:
    dst.write(suitability_2d_out.astype(np.float32), 1)
print(f"  Continuous index saved: {cont_path}")

# Classified suitability map
cls_meta = ref_meta.copy()
cls_meta.update({"dtype": "uint8", "nodata": 0, "count": 1})
cls_path = os.path.join(MAPS_DIR, "biochar_suitability_classified.tif")

with rasterio.open(cls_path, "w", **cls_meta) as dst:
    dst.write(classified_2d, 1)
print(f"  Classified map saved: {cls_path}")


# ── 7. Area statistics by class ───────────────────────────────
pixel_area_km2 = (ref_meta["transform"].a / 1000) ** 2  # pixel width in km

area_stats = []
total_valid = (classified_2d > 0).sum()

for class_id, info in SUITABILITY_CLASSES.items():
    count       = (classified_2d == class_id).sum()
    area_km2    = count * pixel_area_km2
    area_pct    = (count / total_valid * 100) if total_valid > 0 else 0
    area_stats.append({
        "Class":       class_id,
        "Suitability": info["label"],
        "Pixels":      count,
        "Area (km²)":  round(area_km2, 1),
        "Area (%)":    round(area_pct, 2)
    })

area_df = pd.DataFrame(area_stats)
area_df.to_csv(os.path.join(OUTPUT_DIR, "suitability_area_statistics.csv"), index=False)

print("\nSuitability Area Statistics")
print("=" * 65)
print(area_df.to_string(index=False))
print("=" * 65)


# ── 8. Publication-ready map figure ───────────────────────────
print("\nGenerating publication-ready map figure...")

# Build RGB display array from classified raster
color_map = {
    class_id: matplotlib.colors.to_rgb(info["color"])
    for class_id, info in SUITABILITY_CLASSES.items()
}
rgb = np.zeros((*ref_shape, 3), dtype=np.float32)
for class_id, rgb_val in color_map.items():
    mask = classified_2d == class_id
    rgb[mask] = rgb_val

# Set nodata pixels to white
nodata_mask = classified_2d == 0
rgb[nodata_mask] = [1, 1, 1]

fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(rgb, extent=[
    ref_meta["transform"].c,
    ref_meta["transform"].c + ref_meta["transform"].a * ref_shape[1],
    ref_meta["transform"].f + ref_meta["transform"].e * ref_shape[0],
    ref_meta["transform"].f
])

# Legend patches
legend_patches = [
    mpatches.Patch(
        color=info["color"],
        label=f"Class {cid}: {info['label']}  "
              f"({area_df.loc[area_df['Class']==cid,'Area (%)'].values[0]:.1f}%)"
    )
    for cid, info in SUITABILITY_CLASSES.items()
]
ax.legend(
    handles=legend_patches,
    loc="lower left",
    fontsize=9,
    framealpha=0.9,
    title="Suitability class",
    title_fontsize=9
)

ax.set_title(
    "Biochar Suitability Map — Agricultural Soils of Pakistan\n"
    "GIS and Machine Learning Framework | Ghayas Haider Sajid\n"
    "Quaid-i-Azam University, Islamabad",
    fontsize=11, pad=12
)
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "biochar_suitability_map.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("  Map figure saved: outputs/figures/biochar_suitability_map.png")


# ── 9. Pie chart of area distribution ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
wedge_colors = [SUITABILITY_CLASSES[i]["color"] for i in range(1, 6)]
wedge_sizes  = area_df["Area (%)"].tolist()
wedge_labels = [f"{row['Suitability']}\n{row['Area (%)']:.1f}%"
                for _, row in area_df.iterrows()]

wedges, texts = ax.pie(
    wedge_sizes, labels=wedge_labels,
    colors=wedge_colors, startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 0.8}
)
for text in texts:
    text.set_fontsize(8)

ax.set_title(
    "Biochar Suitability — Area Distribution\nPakistan Agricultural Soils",
    fontsize=10, pad=10
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "suitability_area_pie.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("  Pie chart saved: outputs/figures/suitability_area_pie.png")

print("\nSuitability mapping complete.")
print("All outputs are in the outputs/ directory.")
