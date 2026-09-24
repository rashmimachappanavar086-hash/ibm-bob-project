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

    # =============================================================================
# app.py
# House Price Predictor — Flask REST API Backend
#
# Endpoints:
#   GET  /health              Liveness check
#   POST /predict             Predict price from JSON body
#   GET  /feature-importance  Feature importance scores from trained model
#   GET  /stats               Dataset descriptive statistics
#
# Usage : python app.py
#         API runs on http://127.0.0.1:5000
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "house_price_regression_dataset.csv")

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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


def load_artefacts():
    """Load the trained model and scaler from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    model, scaler = load_artefacts()
    print("[INFO] Model and scaler loaded successfully.")
except FileNotFoundError:
    model, scaler = None, None
    print("[WARN] model.pkl / scaler.pkl not found — run model.py first.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict house price.

    Request body (JSON):
        {
            "Square_Footage": 2000,
            "Num_Bedrooms": 3,
            "Num_Bathrooms": 2,
            "Year_Built": 2005,
            "Lot_Size": 2.5,
            "Garage_Size": 1,
            "Neighborhood_Quality": 7
        }

    Response (JSON):
        {"predicted_price": 542318.45}
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run model.py first."}), 503

    data = request.get_json(force=True)

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row    = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
        row_sc = scaler.transform(row)
        price  = float(model.predict(row_sc)[0])
        return jsonify({"predicted_price": round(price, 2)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance scores sorted descending."""
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_).tolist()
    else:
        return jsonify({"error": "Model does not expose feature importance."}), 400

    result = sorted(
        [{"feature": f, "importance": round(s, 4)} for f, s in zip(FEATURES, scores)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    """Return descriptive statistics of the dataset."""
    try:
        df   = pd.read_csv(DATA_PATH)
        desc = df.describe().round(2).to_dict()
        return jsonify(desc)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # =============================================================================
# app.py
# House Price Predictor — Flask REST API Backend
#
# Endpoints:
#   GET  /health              Liveness check
#   POST /predict             Predict price from JSON body
#   GET  /feature-importance  Feature importance scores from trained model
#   GET  /stats               Dataset descriptive statistics
#
# Usage : python app.py
#         API runs on http://127.0.0.1:5000
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "house_price_regression_dataset.csv")

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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


def load_artefacts():
    """Load the trained model and scaler from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    model, scaler = load_artefacts()
    print("[INFO] Model and scaler loaded successfully.")
except FileNotFoundError:
    model, scaler = None, None
    print("[WARN] model.pkl / scaler.pkl not found — run model.py first.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict house price.

    Request body (JSON):
        {
            "Square_Footage": 2000,
            "Num_Bedrooms": 3,
            "Num_Bathrooms": 2,
            "Year_Built": 2005,
            "Lot_Size": 2.5,
            "Garage_Size": 1,
            "Neighborhood_Quality": 7
        }

    Response (JSON):
        {"predicted_price": 542318.45}
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run model.py first."}), 503

    data = request.get_json(force=True)

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row    = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
        row_sc = scaler.transform(row)
        price  = float(model.predict(row_sc)[0])
        return jsonify({"predicted_price": round(price, 2)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Return feature importance scores sorted descending."""
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_).tolist()
    else:
        return jsonify({"error": "Model does not expose feature importance."}), 400

    result = sorted(
        [{"feature": f, "importance": round(s, 4)} for f, s in zip(FEATURES, scores)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    """Return descriptive statistics of the dataset."""
    try:
        df   = pd.read_csv(DATA_PATH)
        desc = df.describe().round(2).to_dict()
        return jsonify(desc)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # =============================================================================
# requirements.txt
# House Price Predictor — Python dependencies
#
# Install all dependencies with:
#   pip install -r requirements.txt
#
# Python version: 3.10 or higher recommended
# =============================================================================

# ---------------------------------------------------------------------------
# Data Science & Machine Learning
# ---------------------------------------------------------------------------
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11            # required internally by scikit-learn

# ---------------------------------------------------------------------------
# Backend — Flask REST API
# ---------------------------------------------------------------------------
flask>=3.0
flask-cors>=4.0

# ---------------------------------------------------------------------------
# Frontend — Streamlit UI
# ---------------------------------------------------------------------------
streamlit>=1.35
plotly>=5.20
requests>=2.31

# ---------------------------------------------------------------------------
# Optional — needed for trendline="ols" in Plotly scatter charts
# ---------------------------------------------------------------------------
statsmodels>=0.14