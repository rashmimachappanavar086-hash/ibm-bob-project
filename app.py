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
