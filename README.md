# House Price Predictor

A full-stack machine-learning web application that predicts residential house
prices using regression models trained on a 1,000-property dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [Frontend Pages](#frontend-pages)
8. [Model Details](#model-details)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project demonstrates an end-to-end ML pipeline:

- **Data** — 1,000 synthetic house records with 7 features and 1 price target
- **Model** — Three regression algorithms trained and compared; best saved automatically
- **Backend** — Flask REST API serves predictions and statistics
- **Frontend** — Streamlit multi-page app with interactive Plotly charts

---

## Tech Stack

| Layer       | Technology                          | Purpose                        |
|-------------|-------------------------------------|--------------------------------|
| ML / Data   | scikit-learn, pandas, numpy         | Training, preprocessing        |
| Backend     | Flask, flask-cors                   | REST API on port 5000          |
| Frontend    | Streamlit, Plotly, requests         | Interactive UI on port 8501    |
| Serialisation | pickle                            | Model and scaler persistence   |

---

## Project Structure

```
house_price_project/
│
├── data/
│   └── house_price_regression_dataset.csv   <- raw dataset (1,000 rows)
│
├── backend/
│   ├── model.py        <- train models, save best as model.pkl
│   ├── app.py          <- Flask REST API
│   ├── model.pkl       <- trained model (generated)
│   └── scaler.pkl      <- StandardScaler (generated)
│
├── frontend/
│   └── streamlit_app.py  <- 4-page Streamlit UI
│
├── requirements.txt    <- all Python dependencies
├── README.md           <- this file
└── project_report.md   <- detailed technical report
```

---

## Dataset

**File:** `data/house_price_regression_dataset.csv`

| Column                | Type    | Description                      |
|-----------------------|---------|----------------------------------|
| Square_Footage        | int     | Interior area in sq ft           |
| Num_Bedrooms          | int     | Number of bedrooms (1–6)         |
| Num_Bathrooms         | int     | Number of bathrooms (1–5)        |
| Year_Built            | int     | Year the house was constructed   |
| Lot_Size              | float   | Lot area in acres                |
| Garage_Size           | int     | Number of parking spaces (0–3)   |
| Neighborhood_Quality  | int     | Quality score 1 (low) – 10 (high)|
| House_Price           | float   | Target — sale price in USD       |

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On this machine use Miniconda Python to avoid DLL issues:
> `C:\Users\Rashmi HM\miniconda3\python.exe`

---

### Step 2 — Train the model

```bash
cd backend
python model.py
```

Expected output:
```
Model          R2        MAE          RMSE
LinearReg...   0.9984    8,175        10,071
RandomFor...   0.9939    16,106       19,853
GradBoost...   0.9970    11,402       14,016

Best model : LinearRegression  (R2 = 0.9984)
Saved model  -> ...backend/model.pkl
Saved scaler -> ...backend/scaler.pkl
```

---

### Step 3 — Start the Flask API  (Terminal 1)

```bash
cd backend
python app.py
```

API available at: **http://127.0.0.1:5000**

---

### Step 4 — Start the Streamlit frontend  (Terminal 2)

```bash
cd frontend
streamlit run streamlit_app.py
```

UI available at: **http://localhost:8501**

---

## API Reference

### GET /health

Check if the API and model are loaded.

**Response:**
```json
{"status": "ok", "model_loaded": true}
```

---

### POST /predict

Predict the price of a house.

**Request body:**
```json
{
  "Square_Footage": 2000,
  "Num_Bedrooms": 3,
  "Num_Bathrooms": 2,
  "Year_Built": 2005,
  "Lot_Size": 2.5,
  "Garage_Size": 1,
  "Neighborhood_Quality": 7
}
```

**Response:**
```json
{"predicted_price": 542318.45}
```

**cURL example:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"Square_Footage\":2000,\"Num_Bedrooms\":3,\"Num_Bathrooms\":2,\"Year_Built\":2005,\"Lot_Size\":2.5,\"Garage_Size\":1,\"Neighborhood_Quality\":7}"
```

---

### GET /feature-importance

Returns feature importance scores ranked highest to lowest.

**Response:**
```json
[
  {"feature": "Square_Footage", "importance": 0.4821},
  {"feature": "Neighborhood_Quality", "importance": 0.2134},
  ...
]
```

---

### GET /stats

Returns descriptive statistics (count, mean, std, min, max, etc.) for all dataset columns.

---

## Frontend Pages

| Page                  | Description                                                       |
|-----------------------|-------------------------------------------------------------------|
| Predict Price         | Sliders for all 7 features, calls API, shows price + gauge chart |
| EDA Dashboard         | KPI cards, histogram, scatter plot, boxplot, heatmap              |
| Feature Importance    | Horizontal bar chart of model feature weights                     |
| Dataset Stats         | Raw data preview, describe(), missing values, data types          |

---

## Model Details

Three scikit-learn regression models are trained and compared:

| Model                | Description                                                  |
|----------------------|--------------------------------------------------------------|
| LinearRegression     | Ordinary Least Squares — fast and highly interpretable       |
| RandomForestRegressor | Ensemble of 100 decision trees — handles non-linearity      |
| GradientBoosting     | 200 boosted trees — strong accuracy, slightly slower         |

Selection criterion: highest **R² score** on a 20% held-out test set.

All features are standardised with `StandardScaler` before training and prediction.

---

## Troubleshooting

| Problem                                  | Solution                                                        |
|------------------------------------------|-----------------------------------------------------------------|
| `http://localhost:8501` does not open    | Run `streamlit run frontend/streamlit_app.py` first            |
| "Cannot reach Flask API" on Predict page | Start `python backend/app.py` in a separate terminal           |
| `ModuleNotFoundError: sklearn`           | Use Miniconda Python: `miniconda3\python.exe`                  |
| `DLL load failed`                        | System Python 3.13 has blocked DLLs — use Miniconda Python    |
| `model.pkl not found`                    | Run `python backend/model.py` to train and save the model      |
| Feature Importance shows empty           | Only available after Flask API is running                       |
