# =============================================================================
# model.py
# House Price Predictor — Model Training Script
#
# Purpose : Load dataset, train & compare 3 regression models, save the best.
# Output  : model.pkl and scaler.pkl in the same backend/ folder.
# Usage   : python model.py
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "house_price_regression_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]
TARGET       = "House_Price"
TEST_SIZE    = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Read the CSV dataset and return a DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[Data]  Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def train() -> tuple:
    """Train candidate models, pick the best by R2, persist artefacts."""
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    candidates = {
        "LinearRegression" : LinearRegression(),
        "RandomForest"     : RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting" : GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    best_model, best_r2, best_name = None, -np.inf, ""

    print("\n{:<24} {:>8}  {:>12}  {:>12}".format("Model", "R2", "MAE", "RMSE"))
    print("-" * 62)
    for name, mdl in candidates.items():
        mdl.fit(X_train_sc, y_train)
        preds = mdl.predict(X_test_sc)
        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name:<24} {r2:>8.4f}  {mae:>12,.0f}  {rmse:>12,.0f}")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, mdl, name

    print("-" * 62)
    print(f"\nBest model : {best_name}  (R2 = {best_r2:.4f})\n")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    return best_model, scaler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
