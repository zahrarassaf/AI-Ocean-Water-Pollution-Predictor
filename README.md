# Satellite-Based Water Quality Assessment in the Gulf of Mexico

## Overview

This project implements a machine learning system for assessing water quality in the Gulf of Mexico using multi-temporal satellite observations. The focus is on building a reproducible, end-to-end pipeline that transforms raw Copernicus Marine Service NetCDF data into coarse water-quality classifications, exposed through a REST API and an interactive dashboard.

Rather than attempting to measure pollution directly, the system relies on established oceanographic proxies (primarily chlorophyll-related variables) and evaluates how reliably these indicators can be used for regional-scale water quality assessment.

---

## Study Area: Gulf of Mexico

The analysis focuses on the Gulf of Mexico, bounded by **18.02°N–29.98°N** latitude and **97.98°W–88.02°W** longitude.

* **Spatial resolution:** approximately 0.042° (~4.6 km grid)
* **Grid size:** 288 × 240 spatial points (69,120 locations per time step)
* **Temporal coverage:** January 2022 to November 2025 (47 monthly composites)
* **Regional characteristics:** includes the Mississippi River plume, Loop Current system, and both shallow shelf and deep-water regions

These features introduce strong spatial heterogeneity, making the region a useful but challenging test case for satellite-based water quality assessment.

---

## Data Sources

The analysis is based on four satellite-derived NetCDF datasets provided by the Copernicus Marine Service:

* Chlorophyll-a concentration (CHL)
* Diffuse attenuation coefficient at 490 nm (KD490)
* Primary productivity (PP)
* Secchi depth

Each dataset is provided on a different spatial grid (ranging from approximately 282×398 to 288×240), with non-uniform temporal coverage. Aligning these heterogeneous sources into a consistent dataset required explicit handling of spatial resolution, temporal alignment, and missing values.

---

## Spatiotemporal Processing and Feature Engineering

All input datasets are stored as three-dimensional tensors (time × latitude × longitude). A custom preprocessing routine was implemented to:

* Synchronize observations across time
* Interpolate variables onto a common spatial grid
* Handle missing values using context-aware environmental patterns

After alignment, the data were flattened into a sample-wise feature matrix. From the raw satellite variables, **26 oceanographic features** were extracted, including:

* Core indicators: chlorophyll concentration, primary productivity, and light attenuation
* Phytoplankton community structure (diatoms, dinoflagellates, and size-fractionated groups)
* Optical properties such as colored dissolved matter (CDM) and particulate backscattering (BBP)
* Uncertainty estimates associated with satellite measurements

The full preprocessing pipeline operates on approximately 2.3 million spatial samples; a representative subset of 100,000 samples was used for model training to balance coverage and computational cost.

---

## Model Development

Water quality classification is performed using a Random Forest classifier. The model was trained with the following configuration:

* 100 decision trees with depth constraints determined empirically
* Stratified 5-fold cross-validation
* Class weighting to account for imbalance between water quality categories

The classification task groups conditions into three levels based on chlorophyll thresholds commonly used in marine research:

* **LOW:** CHL ≤ 1.0 mg/m³ (oligotrophic conditions)
* **MEDIUM:** 1.0 < CHL ≤ 5.0 mg/m³ (mesotrophic conditions)
* **HIGH:** CHL > 5.0 mg/m³ (eutrophic conditions)

---

## Validation Approach

Model performance was evaluated using multiple complementary strategies to reduce optimistic bias and better reflect real-world behavior:

* **Stratified 5-fold cross-validation** to preserve class balance across spatial and temporal samples
* **Geographical hold-out testing** to assess generalization to unseen locations within the Gulf of Mexico
* **Consistency checks against established chlorophyll-based thresholds** commonly used in marine water quality studies
* **Error analysis** comparing near-coastal and open-ocean regions, where satellite signal reliability differs

---

## Performance and Interpretation

On held-out test data, the model achieved an overall accuracy of **98.56%**, with a mean cross-validation score of **98.34%**. Performance was stable across folds, indicating consistent behavior within the Gulf of Mexico dataset.

Feature importance analysis shows that chlorophyll concentration (26.3%) and primary productivity (18.7%) are the dominant predictors, which is consistent with established understanding of eutrophication-driven water quality dynamics. Most classification errors occur in near-coastal regions, where optical complexity and terrestrial influence reduce the reliability of satellite-derived signals.

---

## Technical Implementation Notes

Several implementation details were important for handling data volume and heterogeneity:

* **Data alignment:** Custom interpolation routines to reconcile differing grid resolutions (282×398 to 288×240)
* **Feature preservation:** Satellite uncertainty estimates retained as independent features
* **Computational scaling:** Parallelized NetCDF processing using Dask for memory-efficient execution
* **Model persistence:** Joblib serialization with associated metadata and feature descriptors

---

## System Architecture

The trained model is deployed as a containerized microservice system:

### API Service (FastAPI)

* REST endpoints for single and batch predictions
* Input validation using Pydantic models with scientific constraints
* Automatic fallback to a rule-based classifier if the ML model is unavailable
* Auto-generated documentation available at the `/docs` endpoint

### Interactive Dashboard (Streamlit)

* Real-time prediction interface with configurable parameters
* Batch CSV upload and summary statistics
* Visualization of prediction distributions and confidence scores
* Basic tracking of historical predictions

### Deployment

* Docker-based containerization
* Multi-service orchestration via Docker Compose
* Model artifacts mounted as persistent volumes

The API is exposed on port 8000, and the dashboard on port 8501.

---

## Appropriate Use and Limitations

This system is designed for regional-scale water quality assessment using satellite-derived proxies. Several limitations should be considered when interpreting results:

* **Proxy-based classification:** The model infers water quality from chlorophyll-related variables rather than direct pollution measurements.
* **Geographic specificity:** Training and validation are limited to the Gulf of Mexico; performance in other regions is not guaranteed.
* **Coastal complexity:** Shallow and near-shore areas are more prone to misclassification due to optical interference and sediment effects.
* **Temporal resolution:** Approximately monthly observations are suitable for seasonal analysis but not short-term event detection.

These constraints reflect broader challenges in satellite-based environmental monitoring rather than implementation deficiencies.

---

## Scope and Extensions

The project serves as:

* A technical case study in spatiotemporal satellite data processing
* An example of production-style deployment for environmental ML models
* A baseline system that can be extended with in-situ data, alternative models, or region-specific calibration

The emphasis throughout is on transparency, reproducibility, and realistic interpretation of model outputs rather than absolute pollution quantification.
