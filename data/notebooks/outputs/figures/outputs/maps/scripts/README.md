# Biochar Suitability Mapping
### GIS and Machine Learning Framework for Agricultural Soils

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/Paper-Under%20Review-orange)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

## Overview
This repository contains the complete code and methodology for a 
national-scale biochar suitability assessment across agricultural 
soils in Pakistan, combining GIS-based multi-criteria decision 
analysis (MCDA) with machine learning.

**Key result:** Bagging Regressor achieved R² = 0.96  
**Study area:** Pakistan (national scale)  
**Status:** Under review — *Soil and Tillage Research*

Methods
- GIS-based spatial analysis (ArcGIS Pro / QGIS)
- Analytical Hierarchy Process (AHP) for criterion weighting
- Machine learning: Random Forest, XGBoost, Decision Tree, Bagging Regressor
- SHAP (SHapley Additive Explanations) for model interpretability

Repository structure
    data/           Raw data sources and download instructions
    notebooks/      Analysis notebooks (run in order 01–05)
    scripts/        Reusable Python utilities
    outputs/        Figures, maps, and model results

How to reproduce
    git clone https://github.com/[ghayashaider1277-sys]/biochar-suitability-mapping
    cd biochar-suitability-mapping
    pip install -r requirements.txt

Then follow the notebooks in order.

Citation
*Under review. Citation will be added upon acceptance.*

Author
Ghayas Haider Sajid  
Quaid-i-Azam University, Islamabad, Pakistan  
