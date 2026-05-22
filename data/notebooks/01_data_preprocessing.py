# ============================================================
# Biochar Suitability Mapping — Ghayas Haider Sajid
# Quaid-i-Azam University, Islamabad, Pakistan
# MPhil Thesis | Under review: Soil and Tillage Research
# ============================================================
# Script 01: Data Preprocessing
# Description: Loads soil property rasters and remote sensing
#              data, computes DEM-derived terrain attributes,
#              normalises all input features, and assembles
#              the final multi-layer feature stack for ML.
# ============================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from scipy.ndimage import generic_filter
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = "data/"
OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Define input raster layers ────────────────────────────
# All rasters must share the same CRS, extent, and resolution
# after reprojection (handled below).
RASTER_INPUTS = {
    "soc":          os.path.join(DATA_DIR, "soilgrids_soc_250m.tif"),       # Soil organic carbon (g/kg)
    "ph":           os.path.join(DATA_DIR, "soilgrids_ph_250m.tif"),        # Soil pH (×10)
    "bulk_density": os.path.join(DATA_DIR, "soilgrids_bdod_250m.tif"),      # Bulk density (cg/cm³)
    "clay":         os.path.join(DATA_DIR, "soilgrids_clay_250m.tif"),      # Clay content (%)
    "cec":          os.path.join(DATA_DIR, "soilgrids_cec_250m.tif"),       # Cation exchange capacity
    "dem":          os.path.join(DATA_DIR, "srtm_dem_30m.tif"),             # Elevation (m)
    "ndvi":         os.path.join(DATA_DIR, "sentinel2_ndvi.tif"),           # NDVI (Sentinel-2)
    "lulc":         os.path.join(DATA_DIR, "lulc_pakistan_10m.tif"),        # Land use / land cover
    "rainfall":     os.path.join(DATA_DIR, "chirps_annual_rainfall.tif"),   # Annual rainfall (mm)
    "temperature":  os.path.join(DATA_DIR, "worldclim_tmean.tif"),          # Mean temperature (°C)
}

TARGET_CRS        = "EPSG:32642"   # UTM Zone 42N — covers most of Pakistan
TARGET_RESOLUTION = 250            # metres — matches SoilGrids base resolution
NODATA_VALUE      = -9999


# ── 2. Reproject and resample all rasters to common grid ─────
def reproject_raster(src_path, dst_path, target_crs, target_res, nodata=NODATA_VALUE):
    """
    Reprojects and resamples a raster to a target CRS and
    spatial resolution using bilinear resampling.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds,
            resolution=target_res
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
            "nodata":    nodata,
            "dtype":     "float32"
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
    print(f"  Reprojected: {os.path.basename(src_path)}")


print("Step 1: Reprojecting rasters to common grid...")
reprojected = {}
for name, path in RASTER_INPUTS.items():
    out_path = os.path.join(OUTPUT_DIR, f"reproj_{name}.tif")
    reproject_raster(path, out_path, TARGET_CRS, TARGET_RESOLUTION)
    reprojected[name] = out_path


# ── 3. Compute DEM-derived terrain attributes ─────────────────
def compute_slope(dem_array, cell_size=TARGET_RESOLUTION):
    """
    Computes slope in degrees from a DEM array using a
    3×3 neighbourhood finite difference approximation.
    """
    dz_dx = generic_filter(dem_array, lambda x: (x[5] - x[3]) / (2 * cell_size), size=3)
    dz_dy = generic_filter(dem_array, lambda x: (x[7] - x[1]) / (2 * cell_size), size=3)
    slope  = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    return slope.astype(np.float32)


def compute_twi(dem_array, cell_size=TARGET_RESOLUTION):
    """
    Computes Topographic Wetness Index (TWI = ln(a / tan(β)))
    where a = upslope contributing area (approximated here
    as a local slope proxy) and β = local slope angle.
    TWI captures soil moisture redistribution potential.
    """
    slope_rad = np.radians(compute_slope(dem_array, cell_size))
    slope_rad = np.where(slope_rad < 0.001, 0.001, slope_rad)  # avoid division by zero
    twi = np.log(cell_size / np.tan(slope_rad))
    return twi.astype(np.float32)


print("\nStep 2: Computing terrain attributes from DEM...")
with rasterio.open(reprojected["dem"]) as dem_src:
    dem_array = dem_src.read(1).astype(np.float32)
    dem_meta  = dem_src.meta.copy()
    nodata    = dem_src.nodata

dem_array = np.where(dem_array == nodata, np.nan, dem_array)

slope_array = compute_slope(dem_array)
twi_array   = compute_twi(dem_array)

for attr_name, attr_array in [("slope", slope_array), ("twi", twi_array)]:
    out_path = os.path.join(OUTPUT_DIR, f"reproj_{attr_name}.tif")
    with rasterio.open(out_path, "w", **dem_meta) as dst:
        dst.write(attr_array, 1)
    reprojected[attr_name] = out_path
    print(f"  Computed and saved: {attr_name}")


# ── 4. Stack all layers into a flat feature table ────────────
print("\nStep 3: Assembling feature stack...")

feature_arrays = {}
reference_meta = None

for name, path in reprojected.items():
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        arr = np.where(arr == src.nodata, np.nan, arr)
        feature_arrays[name] = arr
        if reference_meta is None:
            reference_meta = src.meta.copy()

# Flatten spatial arrays into rows, drop pixels with any NaN
n_rows, n_cols = next(iter(feature_arrays.values())).shape
df = pd.DataFrame({
    name: arr.flatten()
    for name, arr in feature_arrays.items()
})

df_clean = df.dropna().reset_index(drop=True)
print(f"  Total valid pixels: {len(df_clean):,} of {n_rows * n_cols:,}")


# ── 5. Normalise features using Min-Max scaling ──────────────
print("\nStep 4: Normalising feature values (Min-Max)...")

scaler       = MinMaxScaler()
feature_cols = [c for c in df_clean.columns if c != "lulc"]  # lulc is categorical
df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

# Save scaler parameters for reproducibility
scaler_params = pd.DataFrame({
    "feature": feature_cols,
    "min":     scaler.data_min_,
    "max":     scaler.data_max_
})
scaler_params.to_csv(os.path.join(OUTPUT_DIR, "scaler_parameters.csv"), index=False)
print("  Scaler parameters saved to outputs/scaler_parameters.csv")


# ── 6. Save processed feature table ─────────────────────────
output_csv = os.path.join(OUTPUT_DIR, "feature_stack.csv")
df_clean.to_csv(output_csv, index=False)
print(f"\nFeature stack saved: {output_csv}")
print(f"Shape: {df_clean.shape[0]} samples × {df_clean.shape[1]} features")
print("\nPreprocessing complete.")
