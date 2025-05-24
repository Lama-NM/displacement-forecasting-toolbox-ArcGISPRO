# Displacement Forecasting Toolbox for ArcGIS Pro

This Python-based toolbox enables the prediction of regular and irregular displacement time series through deep learning models, fully integrated into the ArcGIS Pro environment.

---

## Features and Input Parameters

- Dataset File: Upload your time series CSV.
- CNR P-SBAS Checkbox: Check if your dataset originates from G-TEP (CNR).
- Output Directory: Specify where results are saved (auto-creates `output/` folder).
- Geographic Bounds: Input min/max latitude and longitude to define the area of interest.
- Time Steps: Define the number of future steps to predict (positive integers only).
- Optional Hyperparameters:
  - Number of nodes in LSTM
  - Learning rate
  - Number of epochs
- Plot Learning Curves: Visualize training progress with RMSE curves.

---

## Model Selection Logic

Depending on the dataset type and number of prediction steps:
- Regular, 1 step: 1 LSTM layer
- Regular, >1 steps: 2 LSTM layers
- Irregular, 1 step: TimeGated LSTM
- Irregular, >1 steps: Temporal Fusion Transformer (TFT)

---

## Output

- A shapefile containing displacement points.
- A popup showing model RMSE.
- Optional learning curves (if checkbox is enabled).
- A Tkinter GUI for interactively visualizing predictions.

---

## Full User Manual

For figures and step-by-step screenshots, see the [User Manual (PDF)](docs/manual.pdf).

---

## Citation

> Moualla, L. (2025). *Displacement Forecasting Toolbox: A GIS-Integrated Deep Learning Framework*. GitHub Repository. https://doi.org/10.5281/zenodo.15502468


