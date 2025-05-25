# Displacement Forecasting Toolbox for ArcGIS Pro

This Python-based toolbox enables the prediction of regular and irregular displacement time series through deep learning models, fully integrated into the ArcGIS Pro environment.

---

## Features and Input Parameters

- Dataset File: Upload your time series CSV.
- CNR P-SBAS Checkbox: Check if your dataset originates from G-TEP (CNR).
- Output Directory: Specify where results are saved (auto-creates `output/` folder).
- Geographic Bounds: Input min/max latitude and longitude to define the area of interest.
- Time Steps: Define the number of future steps to predict (positive integers only).
- Optional Hyperparameters (automatically assigned if left empty):
    - Number of nodes in the first LSTM layer:
        Defaults to 50 for 1-step prediction in both regular and irregular datasets.
        Defaults to 100 for multi-step prediction in regular datasets, and 8 in irregular datasets.
    - Number of nodes in the second LSTM layer:
        Only applies to multi-step regular datasets. Defaults to 50 if unspecified.
    - Learning rate:
        Defaults to 0.001 for 1-step prediction in all cases.
        Defaults to 0.0001 for multi-step regular datasets; remains 0.001 for irregular.
     - Number of training epochs (epochs):
        Defaults to 15 for 1-step regular datasets and 35 for 1-step irregular.
        Defaults to 35 for multi-step regular datasets and 50 for multi-step irregular.
- (Optional) Plot Learning Curves: Visualize the model’s training and validation performance over epochs using RMSE 
   as the evaluation metric.

---

## Model Selection Logic

Depending on the dataset type and number of prediction steps:
- Regular, 1 step: LSTM model with 1 LSTM layer  
- Regular, >1 steps: LSTM model with 2 LSTM layers  
- Irregular, 1 step: TimeGated LSTM model
- Irregular, >1 steps: Temporal Fusion Transformer model

---

## Output

- A shapefile containing displacement points.
- A popup showing model RMSE.
- Optional learning curves (if checkbox is enabled).
- A Tkinter GUI for interactively visualizing predictions.

---

## Citation

> Moualla, L. (2025). *Displacement Forecasting Toolbox: A GIS-Integrated Deep Learning Framework*. GitHub Repository. https://doi.org/10.5281/zenodo.15502468


