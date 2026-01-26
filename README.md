# Marine Pollution Prediction System

## Overview

A satellite-based machine learning system for predicting **marine pollution risk levels** using Copernicus Marine Service data. The system processes multiple oceanographic and biogeochemical variables to classify marine water quality into three categories (**LOW, MEDIUM, HIGH**) and provides **uncertainty quantification** for its predictions.

> **Important Note**
> This system estimates pollution risk using biogeochemical proxies (such as chlorophyll concentration and primary productivity) derived from satellite observations. It does not directly measure chemical or biological contaminants. The project is intended as a research-oriented prototype for environmental monitoring rather than a regulatory or operational pollution detection system.

---

## Quick Start

```bash
# Run the complete data processing and modeling pipeline
python run_pipeline.py

# Train the model and make predictions
python predict.py --train
python predict.py --interactive

# Perform uncertainty analysis
python src/analysis/uncertainty_analyzer.py --quick
```

---

## Model Performance

* **Test Accuracy:** 98.83%
* **Training Accuracy:** 99.97%
* **Cross-Validation Score:** 98.68% (±0.09%)
* **Number of Features:** 26 oceanographic variables
* **Sample Size:** 50,000 data points

> High accuracy is partly due to clearly separated, domain-informed proxy-based class definitions.

---

## Water Quality Classification

Classification thresholds are based on commonly used chlorophyll concentration ranges in marine and oceanographic studies.

| Category | Chlorophyll Range (mg/m³) | Description | Samples |
| -------- | ------------------------- | ----------- | ------- |
| LOW      | ≤ 1.0                     | Clean       | 33%     |
| MEDIUM   | 1.0 – 5.0                 | Moderate    | 33%     |
| HIGH     | > 5.0                     | Polluted    | 34%     |

---

## Feature Importance (Top 5)

* **Primary Productivity (PP):** 20.1%
* **Chlorophyll (CHL):** 16.7%
* **Light Attenuation (KD490):** 16.3%
* **Colored Dissolved Matter (CDM):** 12.9%
* **Particulate Backscattering (BBP):** 6.5%

---

## Data Sources

This project uses four datasets from the **Copernicus Marine Service**:

* **Chlorophyll** (`chlorophyll_full.nc`) — 21 variables
* **Light Attenuation** (`kd490_only.nc`) — KD490 coefficient
* **Optical Properties** (`optical_properties.nc`) — 5 variables
* **Primary Productivity** (`primary_productivity.nc`) — 3 variables

**Spatial Coverage:** Gulf of Mexico
**Geographic Bounds:** 18.02°N–29.98°N, 97.98°W–88.02°W
**Temporal Resolution:** Monthly composites

---

## Project Structure

```text
AI-Ocean-Water-Pollution-Predictor/
├── run_pipeline.py                     # Main data processing pipeline
├── predict.py                          # Model training & prediction
├── api.py                              # REST API interface
├── plot.py                             # Visualization utilities
├── main.py                             # Dashboard entry point
├── src/
│   ├── analysis/uncertainty_analyzer.py # Uncertainty quantification
│   ├── data/                           # Data processing modules
│   ├── models/                         # Model training logic
│   └── utils/                          # Configuration & logging
├── config/                             # YAML configuration files
├── data/                               # Data access instructions (raw data not included)
├── models/                             # Trained models (generated locally)
├── results/                            # Pipeline outputs
├── plots/                              # Figures and dashboards
├── logs/                               # Execution logs
└── requirements.txt                    # Python dependencies
```

---

## Usage Examples

### Interactive Prediction

```bash
python predict.py --interactive
```

Example session:

```text
Chlorophyll (CHL) in mg/m³: 0.25
Additional features: PP=300
→ Prediction: MEDIUM pollution (98.3% confidence)
```

### Model Training

```bash
python predict.py --train
```

Trains a new Random Forest model using the current processed dataset.

---

## Uncertainty Analysis

```bash
python src/analysis/uncertainty_analyzer.py
```

* Ensemble-based uncertainty estimation
* Bootstrap sampling
* Spatial uncertainty mapping
* **R² = 0.9629** for primary productivity prediction

---

## System Components

### 1. Data Pipeline (`run_pipeline.py`)

* Downloads Copernicus Marine Service data
* Aligns multi-source NetCDF datasets
* Generates training-ready features

### 2. Prediction System (`predict.py`)

* Random Forest classifier (200 trees)
* Interactive and batch prediction modes
* Model persistence and reuse

### 3. Uncertainty Quantification

* Ensemble predictions
* Feature importance variability
* Spatial uncertainty estimation

---

## Limitations

* **Proxy-based:** Pollution inferred from biogeochemical indicators
* **Regional:** Trained specifically on Gulf of Mexico data
* **Temporal:** Monthly resolution (not suitable for short-term events)
* **Coastal Complexity:** Higher uncertainty near shorelines

---

## Requirements

* Python 3.8+
* numpy, pandas, xarray, netCDF4
* scikit-learn, joblib
* tqdm

**System Requirements:** ≥ 8 GB RAM, ≥ 10 GB disk space

---

## Future Work

* Integration of additional satellite missions
* Deep learning–based spatiotemporal models
* Extension to global ocean coverage
* Near real-time monitoring capabilities

---

## License

This project is released under the MIT License.

You are free to use, modify, and distribute this software for research, educational, and commercial purposes, provided that the original copyright notice and license are included.
