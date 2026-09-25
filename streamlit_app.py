# ============================================================
# HOUSE PRICE PREDICTOR - STREAMLIT APP
# ============================================================

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# IMPORTANT:
# Path(__file__).resolve().parent gives the folder where
# streamlit_app.py is actually located.
#
# This works both locally and on Streamlit Cloud.

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "house_price_regression_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]

TARGET = "House_Price"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 36px;
        font-weight: 700;
        color: #1b5e20;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    """Load the house price dataset."""

    if not DATA_PATH.exists():
        return None

    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load trained machine learning model."""

    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():
    """Load trained scaler."""

    if not SCALER_PATH.exists():
        return None

    try:
        return joblib.load(SCALER_PATH)
    except Exception:
        return None


# ============================================================
# INITIALIZE
# ============================================================

df = load_data()
model = load_model()
scaler = load_scaler()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 House Price Predictor")

st.sidebar.markdown("### Navigate")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Predict Price",
        "📊 EDA Dashboard",
        "📈 Feature Importance",
        "📋 Dataset Stats"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Backend:** Streamlit

    **ML:** Scikit-learn

    **Model:** House Price Regression
    """
)


# ============================================================
# FILE STATUS
# ============================================================

with st.sidebar.expander("🔍 File Status"):

    st.write(
        "Dataset:",
        "✅ Found" if DATA_PATH.exists() else "❌ Missing"
    )

    st.write(
        "Model:",
        "✅ Found" if MODEL_PATH.exists() else "❌ Missing"
    )

    st.write(
        "Scaler:",
        "✅ Found" if SCALER_PATH.exists() else "❌ Missing"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏠 House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based House Price Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK FILES
# ============================================================

if df is None:

    st.error(
        "❌ Dataset not found: "
        f"{DATA_PATH}"
    )

    st.info(
        "Make sure house_price_regression_dataset.csv "
        "is present in the same folder as streamlit_app.py."
    )


if model is None:

    st.error(
        "❌ Model file not found: "
        f"{MODEL_PATH}"
    )

    st.info(
        "Make sure model.pkl is present in the same folder "
        "as streamlit_app.py."
    )


if scaler is None:

    st.warning(
        "⚠️ Scaler file not found: "
        f"{SCALER_PATH}"
    )

    st.info(
        "If your model requires a scaler, upload scaler.pkl "
        "to the same folder."
    )


# ============================================================
# PAGE 1 - PREDICT PRICE
# ============================================================

if page == "🏠 Predict Price":

    st.header("🏠 Predict House Price")

    st.write(
        "Enter the house details below to estimate its price."
    )

    if model is not None:

        col1, col2 = st.columns(2)

        with col1:

            square_footage = st.number_input(
                "Square Footage",
                min_value=100.0,
                max_value=10000.0,
                value=1500.0,
                step=50.0
            )

            bedrooms = st.number_input(
                "Number of Bedrooms",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )

            bathrooms = st.number_input(
                "Number of Bathrooms",
                min_value=1.0,
                max_value=20.0,
                value=2.0,
                step=0.5
            )

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2026,
                value=2010,
                step=1
            )

        with col2:

            lot_size = st.number_input(
                "Lot Size",
                min_value=100.0,
                max_value=100000.0,
                value=5000.0,
                step=100.0
            )

            garage_size = st.number_input(
                "Garage Size",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=1.0
            )

            neighborhood_quality = st.number_input(
                "Neighborhood Quality",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=1.0
            )

        st.markdown("---")

        predict_button = st.button(
            "💰 Predict House Price",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            try:

                # Create input dataframe
                input_data = pd.DataFrame(
                    [[
                        square_footage,
                        bedrooms,
                        bathrooms,
                        year_built,
                        lot_size,
                        garage_size,
                        neighborhood_quality
                    ]],
                    columns=FEATURES
                )

                # ==================================================
                # SCALING
                # ==================================================

                if scaler is not None:

                    input_scaled = scaler.transform(input_data)

                else:

                    input_scaled = input_data

                # ==================================================
                # PREDICTION
                # ==================================================

                prediction = model.predict(input_scaled)

                predicted_price = float(
                    np.asarray(prediction).flatten()[0]
                )

                # ==================================================
                # DISPLAY RESULT
                # ==================================================

                st.markdown(
                    f"""
                    <div class="prediction-box">

                    <div>
                    🏠 Estimated House Price
                    </div>

                    <div class="prediction-value">
                    ₹{predicted_price:,.2f}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "House price prediction completed successfully."
                )

                # ==================================================
                # INPUT SUMMARY
                # ==================================================

                st.subheader("📋 Input Summary")

                display_data = pd.DataFrame(
                    {
                        "Feature": FEATURES,
                        "Value": [
                            square_footage,
                            bedrooms,
                            bathrooms,
                            year_built,
                            lot_size,
                            garage_size,
                            neighborhood_quality
                        ]
                    }
                )

                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:

                st.error(
                    "❌ Prediction failed."
                )

                st.exception(e)


# ============================================================
# PAGE 2 - EDA DASHBOARD
# ============================================================

elif page == "📊 EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")

    if df is None:

        st.warning("Dataset is not available.")

    else:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📈 Numerical Summary")

        st.dataframe(
            df.describe(),
            use_container_width=True
        )

        st.markdown("---")

        # ==================================================
        # HOUSE PRICE DISTRIBUTION
        # ==================================================

        if TARGET in df.columns:

            st.subheader("💰 House Price Distribution")

            st.bar_chart(
                df[TARGET].value_counts().sort_index()
            )

        # ==================================================
        # CORRELATION
        # ==================================================

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if not numeric_df.empty:

            st.subheader("🔗 Correlation Matrix")

            correlation = numeric_df.corr()

            st.dataframe(
                correlation.round(3),
                use_container_width=True
            )


# ============================================================
# PAGE 3 - FEATURE IMPORTANCE
# ============================================================

elif page == "📈 Feature Importance":

    st.header("📈 Feature Importance")

    if model is None:

        st.warning(
            "Model is not available."
        )

    else:

        try:

            # ==================================================
            # TREE-BASED MODELS
            # ==================================================

            if hasattr(model, "feature_importances_"):

                importance = model.feature_importances_

                importance_df = pd.DataFrame(
                    {
                        "Feature": FEATURES,
                        "Importance": importance
                    }
                ).sort_values(
                    "Importance",
                    ascending=False
                )

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    importance_df.set_index("Feature")
                )

            # ==================================================
            # LINEAR MODELS
            # ==================================================

            elif hasattr(model, "coef_"):

                coefficients = np.asarray(
                    model.coef_
                ).flatten()

                importance_df = pd.DataFrame(
                    {
                        "Feature": FEATURES,
                        "Coefficient": coefficients,
                        "Absolute Importance": np.abs(
                            coefficients
                        )
                    }
                ).sort_values(
                    "Absolute Importance",
                    ascending=False
                )

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    importance_df.set_index(
                        "Feature"
                    )["Absolute Importance"]
                )

            else:

                st.info(
                    "Feature importance is not directly available "
                    "for this model type."
                )

        except Exception as e:

            st.error(
                "Could not calculate feature importance."
            )

            st.exception(e)


# ============================================================
# PAGE 4 - DATASET STATISTICS
# ============================================================

elif page == "📋 Dataset Stats":

    st.header("📋 Dataset Statistics")

    if df is None:

        st.warning(
            "Dataset is not available."
        )

    else:

        # ==================================================
        # BASIC METRICS
        # ==================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

        st.markdown("---")

        # ==================================================
        # COLUMN INFORMATION
        # ==================================================

        st.subheader("Column Information")

        column_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
                "Missing Values": [
                    int(df[col].isnull().sum())
                    for col in df.columns
                ],
                "Unique Values": [
                    int(df[col].nunique())
                    for col in df.columns
                ]
            }
        )

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # ==================================================
        # TARGET STATISTICS
        # ==================================================

        if TARGET in df.columns:

            st.subheader(
                "💰 House Price Statistics"
            )

            target = df[TARGET]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Minimum",
                    f"₹{target.min():,.2f}"
                )

            with c2:
                st.metric(
                    "Maximum",
                    f"₹{target.max():,.2f}"
                )

            with c3:
                st.metric(
                    "Average",
                    f"₹{target.mean():,.2f}"
                )

            with c4:
                st.metric(
                    "Median",
                    f"₹{target.median():,.2f}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🏠 House Price Predictor | "
    "Machine Learning & Data Analytics Project"
)

      
          

