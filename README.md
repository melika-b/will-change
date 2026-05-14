# Energy Demand & CO₂ Emission Prediction

This repository contains two Jupyter notebooks that demonstrate data preprocessing, feature engineering, model training, and evaluation for predicting energy demand (`En_Requests (MJ)`) and CO₂ emissions (`co2_emission (kg)`) using environmental features (temperature, humidity, solar hours, wind speed).
It uses data of an office building located in a hot-arid climate.

## Files

- **1_utils.ipynb**  
  Single‑target regression (Linear Regression)  
  - Loads and cleans data (`load_and_clean_data`)  
  - Prepares features and scales them (`prepare_features_target`)  
  - Trains a linear regression model  
  - Evaluates using MAE, MSE, RMSE, R², MAPE  
  - Plots actual vs predicted values (with optional Persian text support)

- **2_utils.ipynb**  
  Multi‑target regression (Random Forest)  
  - Uses the same cleaning function  
  - Prepares two targets: `en_requests (mj)` and `co2_emission (kg)`  
  - Trains a `RandomForestRegressor` for both outputs simultaneously  
  - Evaluates each target separately  
  - Includes a `plot_multi_target` function to visualise predictions for both targets

Both notebooks share helper functions for model saving/loading (`save_model`, `load_model`) and Persian text formatting (optional).

## Dependencies

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn joblib

For Persian text support (optional):
pip install arabic-reshaper python-bidi


Usage
Place your data file (realityvalues.csv) in D:\paper\C\data\exc\ (or modify the path in the notebooks).
Run the cells sequentially.
The notebooks will print evaluation metrics and display plots.


Notes
The data file path is hardcoded; you may need to adjust it.
Outliers are handled using the IQR method (3×IQR).
Missing values are forward/backward filled.
Feature scaling uses StandardScaler.


Author
[Narges Barzanouni]
