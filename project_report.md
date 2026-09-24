# Project Report — House Price Predictor

**Project Title:** House Price Prediction using Machine Learning  
**Technology Stack:** Python, scikit-learn, Flask, Streamlit, Plotly  
**Dataset:** House Price Regression Dataset (1,000 records)  
**Date:** 2024

---

## 1. Executive Summary

This project builds a complete end-to-end machine learning application to predict
residential house prices based on property features. It covers the full pipeline
from raw data ingestion through model training, REST API deployment, and an
interactive web frontend. The best trained model (Linear Regression) achieves an
R² score of **0.9984** — meaning it explains 99.84% of the variance in house prices,
with a Mean Absolute Error of approximately **USD 8,175**.

---

## 2. Problem Statement

Accurately estimating the market value of a residential property is a complex
problem that depends on many interacting factors. Estate agents, buyers, and
investors need a reliable, fast, and accessible tool to get a price estimate
given the physical and locational characteristics of a house.

**Objective:** Build an ML regression model that can predict house prices from
seven property attributes, and expose it through a user-friendly web application.

---

## 3. Dataset Description

| Attribute             | Description                              | Range              |
|-----------------------|------------------------------------------|--------------------|
| Square_Footage        | Interior living area in square feet      | 500 – 6,000        |
| Num_Bedrooms          | Number of bedrooms                       | 1 – 6              |
| Num_Bathrooms         | Number of bathrooms                      | 1 – 5              |
| Year_Built            | Year of construction                     | 1950 – 2024        |
| Lot_Size              | Lot area in acres                        | 0.1 – 10.0         |
| Garage_Size           | Number of garage parking spaces          | 0 – 3              |
| Neighborhood_Quality  | Area quality score (subjective rating)   | 1 (low) – 10 (high)|
| House_Price (target)  | Sale price in US Dollars                 | ~$50,000 – $1.5M   |

- **Total records:** 1,000  
- **Missing values:** None  
- **Data types:** 6 integer columns, 1 float column (Lot_Size), 1 float target (House_Price)

---

## 4. Exploratory Data Analysis

Key findings from the EDA Dashboard:

- **Price Distribution:** Roughly unimodal with a right skew; most properties are
  priced between $200,000 and $800,000.
- **Square Footage correlation:** Strong positive linear relationship with House_Price
  (highest single-feature correlation).
- **Neighborhood Quality correlation:** Strong positive correlation — quality score 10
  properties command significantly higher prices than score 1 properties.
- **Bedrooms vs Price:** Median price increases consistently with bedroom count;
  spread (IQR) also grows with more bedrooms.
- **Year Built:** Newer properties tend to command slightly higher prices, though
  this effect is weaker than square footage or neighborhood quality.
- **No multicollinearity issues** were detected in the correlation heatmap that
  would require feature removal.

---

## 5. Data Preprocessing

| Step                     | Method                     | Rationale                             |
|--------------------------|----------------------------|---------------------------------------|
| Train / test split       | 80% train, 20% test        | Standard holdout for unbiased evaluation |
| Feature scaling          | StandardScaler (z-score)   | Ensures equal feature contribution to linear models |
| No encoding needed       | All features are numeric   | No categorical variables present     |
| No imputation needed     | Zero missing values        | Dataset is complete                   |

The fitted `StandardScaler` is saved alongside the model so that incoming
prediction requests are transformed with the same parameters.

---

## 6. Model Selection

Three candidate models were trained and evaluated:

| Model                | R² Score | MAE (USD)  | RMSE (USD) |
|----------------------|----------|------------|------------|
| LinearRegression     | **0.9984**| **8,175** | **10,071** |
| GradientBoosting     | 0.9970   | 11,402     | 14,016     |
| RandomForestRegressor| 0.9939   | 16,106     | 19,853     |

**Winner: Linear Regression**

The dataset exhibits a strong linear relationship between features and target price.
Linear Regression achieves the best R² with the lowest error metrics, outperforming
both ensemble methods on this particular dataset. The model is also the most
interpretable and fastest to serve predictions.

**Selection criterion:** Highest R² score on the 20% held-out test set.

---

## 7. System Architecture

```
┌─────────────────────────┐       HTTP POST /predict        ┌─────────────────────┐
│   Streamlit Frontend    │  ──────────────────────────────> │   Flask Backend     │
│   (port 8501)           │                                  │   (port 5000)       │
│                         │  <──────────────────────────────  │                     │
│  Page 1: Predict Price  │       {"predicted_price": ...}   │  /predict           │
│  Page 2: EDA Dashboard  │                                  │  /feature-importance│
│  Page 3: Feature Imp.   │  ──── GET /feature-importance ─> │  /health            │
│  Page 4: Dataset Stats  │                                  │  /stats             │
└─────────────────────────┘                                  └──────────┬──────────┘
         |                                                              |
  Reads CSV directly                                            Loads model.pkl
  for EDA pages                                                 and scaler.pkl
         |                                                              |
         └──────────────────────────────────────────────────────────────┘
                               data/house_price_regression_dataset.csv
```

---

## 8. Backend — Flask API

The backend is a lightweight Flask application (`app.py`) that:

1. Loads `model.pkl` and `scaler.pkl` at startup into memory.
2. Exposes four HTTP endpoints:

| Endpoint              | Method | Description                                |
|-----------------------|--------|--------------------------------------------|
| `/health`             | GET    | Returns API status and model load state    |
| `/predict`            | POST   | Accepts 7 features, returns predicted price|
| `/feature-importance` | GET    | Returns feature weights sorted by impact  |
| `/stats`              | GET    | Returns dataset descriptive statistics     |

3. Uses `flask-cors` to allow cross-origin requests from the Streamlit frontend.
4. Returns all responses as JSON.

**Prediction flow:**
```
JSON request → validate fields → build DataFrame → StandardScaler.transform()
  → model.predict() → round to 2 decimals → JSON response
```

---

## 9. Frontend — Streamlit Application

The frontend is a 4-page Streamlit application (`streamlit_app.py`) with:

### Page 1 — Predict Price
- Seven input controls (sliders + selectbox) covering all model features
- Calls `POST /predict` on button click
- Displays predicted price in a Plotly gauge chart (green/amber/red bands)
- Shows an input summary table

### Page 2 — EDA Dashboard
- 4 KPI metric cards (total properties, avg price, avg sq ft, avg bedrooms)
- Price distribution histogram
- Price vs Square Footage scatter plot (colour-coded by bedrooms)
- Price by Bedrooms box plot
- Price vs Neighborhood Quality scatter with OLS trendline
- Full correlation heatmap (Plotly imshow)

### Page 3 — Feature Importance
- Fetches live data from `GET /feature-importance`
- Horizontal bar chart ranked by importance score
- Sortable data table of scores

### Page 4 — Dataset Statistics
- Raw data table (first 50 rows)
- Descriptive statistics (describe())
- Missing value audit
- Data type audit

---

## 10. Results

| Metric                         | Value          |
|--------------------------------|----------------|
| Best model                     | Linear Regression |
| R² score (test set)            | 0.9984         |
| Mean Absolute Error            | USD 8,175      |
| Root Mean Squared Error        | USD 10,071     |
| Training set size              | 800 records    |
| Test set size                  | 200 records    |
| Average prediction (2000 sqft, 3 bed, 2 bath, 2005, 2.5ac, 1 garage, quality 7) | ~USD 470,759 |

---

## 11. Challenges and Solutions

| Challenge                          | Solution Applied                                    |
|------------------------------------|-----------------------------------------------------|
| System Python 3.13 DLL block       | Used Miniconda Python 3.10 environment              |
| Feature name warning from scaler   | Pass features as `pd.DataFrame` with column names  |
| Relative data path from frontend   | Used `pathlib.Path(__file__).resolve().parent.parent` |
| CORS blocking frontend API calls   | Added `flask-cors` with `CORS(app)`                |

---

## 12. Conclusion

The House Price Predictor successfully demonstrates a production-style ML pipeline:

- **Data exploration** confirms strong linear relationships — justifying Linear Regression.
- **Model comparison** ensures the best algorithm is selected automatically.
- **REST API** cleanly decouples the ML logic from the UI.
- **Streamlit frontend** provides an interactive, no-code interface for end users.

The application can be extended with:
- Additional models (XGBoost, Ridge/Lasso)
- Real estate API integration for live market data
- User authentication and prediction history logging
- Docker containerisation for deployment

---

## 13. File Inventory

| File                                                   | Lines | Purpose                         |
|--------------------------------------------------------|-------|---------------------------------|
| `backend/model.py`                                     | ~88   | Model training and persistence  |
| `backend/app.py`                                       | ~130  | Flask REST API                  |
| `frontend/streamlit_app.py`                            | ~230  | Streamlit multi-page UI         |
| `data/house_price_regression_dataset.csv`              | 1001  | Raw dataset                     |
| `requirements.txt`                                     | ~34   | Python dependencies             |
| `README.md`                                            | ~190  | Setup and usage guide           |
| `project_report.md`                                    | ~210  | This report                     |
| `backend/model.pkl`                                    | —     | Serialised trained model        |
| `backend/scaler.pkl`                                   | —     | Serialised StandardScaler       |
