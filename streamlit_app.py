# =============================================================================
# streamlit_app.py
# House Price Predictor — Streamlit Frontend
#
# Pages:
#   Page 1 — Predict Price       : Input sliders -> Flask API -> price + gauge
#   Page 2 — EDA Dashboard       : KPI cards, charts, correlation heatmap
#   Page 3 — Feature Importance  : Bar chart of model feature weights
#   Page 4 — Dataset Stats       : Raw data preview and descriptive statistics
#
# Usage : streamlit run streamlit_app.py
#         UI opens at http://localhost:8501
#
# Requires the Flask backend (app.py) to be running on port 5000 for
# the Predict and Feature Importance pages to work.
# =============================================================================

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE  = "http://127.0.0.1:5000"
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house_price_regression_dataset.csv"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the dataset from disk."""
    return pd.read_csv(DATA_PATH)


def api_predict(payload: dict) -> dict:
    """POST to /predict and return the JSON response."""
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach the Flask API. Is it running on port 5000?"}


def api_feature_importance() -> list:
    """GET /feature-importance and return list of dicts."""
    try:
        r = requests.get(f"{API_BASE}/feature-importance", timeout=5)
        return r.json()
    except requests.exceptions.ConnectionError:
        return []


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/000000/home.png", width=64)
st.sidebar.title("House Price Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Predict Price", "📊 EDA Dashboard", "📈 Feature Importance", "📋 Dataset Stats"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Backend: Flask  |  Frontend: Streamlit  |  ML: scikit-learn")


# ===========================================================================
# PAGE 1 — PREDICT PRICE
# ===========================================================================
if page == "🏠 Predict Price":
    st.title("🏠 House Price Predictor")
    st.markdown(
        "Adjust the sliders to describe the property, then click **Predict** "
        "to get an AI-estimated price."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        square_footage       = st.slider("Square Footage (sq ft)",    500,  6000, 2000, step=50)
        num_bedrooms         = st.slider("Number of Bedrooms",          1,     6,    3)
        num_bathrooms        = st.slider("Number of Bathrooms",         1,     5,    2)
        year_built           = st.slider("Year Built",               1950,  2024, 2000)

    with col2:
        lot_size             = st.slider("Lot Size (acres)",           0.1,  10.0,  2.0, step=0.1)
        garage_size          = st.selectbox("Garage Size (cars)",     [0, 1, 2, 3])
        neighborhood_quality = st.slider("Neighborhood Quality (1–10)", 1,   10,    5)

    st.markdown("---")

    if st.button("Predict Price", use_container_width=True):
        payload = {
            "Square_Footage":       square_footage,
            "Num_Bedrooms":         num_bedrooms,
            "Num_Bathrooms":        num_bathrooms,
            "Year_Built":           year_built,
            "Lot_Size":             lot_size,
            "Garage_Size":          garage_size,
            "Neighborhood_Quality": neighborhood_quality,
        }

        with st.spinner("Getting prediction from model..."):
            result = api_predict(payload)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            price = result["predicted_price"]
            st.success(f"### Estimated House Price:  ${price:,.2f}")

            # Price gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = price,
                title = {"text": "Predicted Price (USD)", "font": {"size": 16}},
                gauge = {
                    "axis" : {"range": [50_000, 1_500_000]},
                    "bar"  : {"color": "#3b82d4"},
                    "steps": [
                        {"range": [50_000,   400_000], "color": "#d1fae5"},
                        {"range": [400_000,  800_000], "color": "#fef3c7"},
                        {"range": [800_000, 1_500_000],"color": "#fee2e2"},
                    ],
                },
                number = {"prefix": "$", "valueformat": ",.0f"},
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Input summary table
            st.markdown("#### Your Input Summary")
            summary = pd.DataFrame([payload]).T.rename(columns={0: "Value"})
            st.dataframe(summary, use_container_width=True)


# ===========================================================================
# PAGE 2 — EDA DASHBOARD
# ===========================================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    df = load_data()

    # KPI metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Properties",   f"{len(df):,}")
    k2.metric("Avg House Price",    f"${df['House_Price'].mean():,.0f}")
    k3.metric("Avg Square Footage", f"{df['Square_Footage'].mean():,.0f} sq ft")
    k4.metric("Avg Bedrooms",       f"{df['Num_Bedrooms'].mean():.1f}")

    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("House Price Distribution")
        fig = px.histogram(
            df, x="House_Price", nbins=40,
            color_discrete_sequence=["#3b82d4"],
            labels={"House_Price": "House Price (USD)"},
        )
        fig.update_layout(bargap=0.05, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price vs Square Footage")
        fig = px.scatter(
            df, x="Square_Footage", y="House_Price",
            color="Num_Bedrooms",
            color_continuous_scale="Blues",
            opacity=0.7,
            labels={"Square_Footage": "Square Footage", "House_Price": "Price (USD)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Price by Number of Bedrooms")
        fig = px.box(
            df, x="Num_Bedrooms", y="House_Price",
            color="Num_Bedrooms",
            labels={"House_Price": "Price (USD)", "Num_Bedrooms": "Bedrooms"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Price vs Neighborhood Quality")
        fig = px.scatter(
            df, x="Neighborhood_Quality", y="House_Price",
            trendline="ols",
            color_discrete_sequence=["#7c5cd8"],
            labels={
                "Neighborhood_Quality": "Neighborhood Quality (1–10)",
                "House_Price": "Price (USD)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = df.corr(numeric_only=True).round(2)
    fig  = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 3 — FEATURE IMPORTANCE
# ===========================================================================
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    st.markdown(
        "Shows which input features have the greatest influence on the "
        "model's price predictions. Data comes live from the Flask API."
    )
    st.markdown("---")

    data = api_feature_importance()

    if not data or (isinstance(data, dict) and "error" in data):
        st.error("Could not load feature importance. Make sure the Flask API is running on port 5000.")
    else:
        fi_df = pd.DataFrame(data)

        fig = px.bar(
            fi_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Importance Score", "feature": "Feature"},
            title="Feature Importance (higher = stronger influence on price)",
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Importance Scores Table")
        st.dataframe(fi_df, use_container_width=True)


# ===========================================================================
# PAGE 4 — DATASET STATS
# ===========================================================================
elif page == "📋 Dataset Stats":
    st.title("📋 Dataset Statistics")
    df = load_data()

    st.markdown("### Dataset Preview  (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### Missing Values Check")
    missing         = df.isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Count"]
    st.dataframe(missing, use_container_width=True)

    st.markdown("### Data Types")
    dtypes         = df.dtypes.reset_index()
    dtypes.columns = ["Feature", "Data Type"]
    st.dataframe(dtypes, use_container_width=True)
