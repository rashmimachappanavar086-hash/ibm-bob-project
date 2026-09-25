
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
# ============================================================
# PATH CONFIGURATION
# ============================================================
# streamlit_app.py is inside:
# house_price_project/frontend/
# ================================
# PATH CONFIGURATION
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "house_price_regression_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
# ================================
# PATH CONFIGURATION
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "house_price_regression_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)
SCALER_PATH = os.path.join(
    ROOT_DIR,
    "scaler.pkl"
)
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
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)
# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found: {SCALER_PATH}")
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
df = load_data()
model, scaler = load_model()
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🏠 House Price Predictor")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Predict Price",
        "📊 EDA Dashboard",
        "📈 Feature Importance",
        "📋 Dataset Stats"
    ]
)
st.sidebar.markdown("---")
st.sidebar.write("**Backend:** Streamlit")
st.sidebar.write("**ML:** Scikit-learn")
st.sidebar.write("**Model:** House Price Regression")
# ============================================================
# PAGE 1 — PREDICT PRICE
# ============================================================
if page == "🏠 Predict Price":
    st.title("🏠 House Price Predictor")
    st.write(
        "Enter the house details below to estimate its price."
    )
    if model is None or scaler is None:
        st.error("Model or scaler could not be loaded.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        square_footage = st.slider(
            "Square Footage (sq ft)",
            min_value=500,
            max_value=5000,
            value=2000,
            step=50
        )
        num_bedrooms = st.slider(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )
        num_bathrooms = st.slider(
            "Number of Bathrooms",
            min_value=1,
            max_value=8,
            value=2
        )
        year_built = st.slider(
            "Year Built",
            min_value=1950,
            max_value=2025,
            value=2000
        )
    with col2:
        lot_size = st.slider(
            "Lot Size (acres)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1
        )
        garage_size = st.selectbox(
            "Garage Size (cars)",
            [0, 1, 2, 3, 4]
        )
        neighborhood_quality = st.slider(
            "Neighborhood Quality (1-10)",
            min_value=1,
            max_value=10,
            value=5
        )
    st.markdown("---")
    if st.button(
        "💰 Predict Price",
        use_container_width=True
    ):
        try:
            input_data = pd.DataFrame(
                [[
                    square_footage,
                    num_bedrooms,
                    num_bathrooms,
                    year_built,
                    lot_size,
                    garage_size,
                    neighborhood_quality
                ]],
                columns=FEATURES
            )
            # Scale input
            input_scaled = scaler.transform(input_data)
            # Predict
            prediction = model.predict(input_scaled)[0]
            st.success(
                f"🏠 Estimated House Price: ₹{prediction:,.2f}"
            )
        except Exception as e:
            st.error(
                f"Prediction error: {str(e)}"
            )
# ============================================================
# PAGE 2 — EDA DASHBOARD
# ============================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    if df.empty:
        st.warning("Dataset is not available.")
        st.stop()
    st.subheader("Dataset Preview")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )
    st.subheader("House Price Distribution")
    if TARGET in df.columns:
        fig, ax = plt.subplots()
        ax.hist(
            df[TARGET].dropna(),
            bins=30
        )
        ax.set_xlabel("House Price")
        ax.set_ylabel("Number of Houses")
        ax.set_title("House Price Distribution")
        st.pyplot(fig)
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(
        include="number"
    )
    if not numeric_df.empty:
        correlation = numeric_df.corr()
        st.dataframe(
            correlation.style.background_gradient(
                cmap="Blues"
            ),
            use_container_width=True
        )
# ============================================================
# PAGE 3 — FEATURE IMPORTANCE
# ============================================================
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    if model is None:
        st.warning("Model is not available.")
        st.stop()
    importance = None
    # Standard tree-based models
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    # Some models expose coefficients
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_)
        if len(importance.shape) > 1:
            importance = importance[0]
    if importance is not None:
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importance
        })
        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )
        st.dataframe(
            importance_df,
            use_container_width=True
        )
        fig, ax = plt.subplots()
        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info(
            "Feature importance is not directly available "
            "for this model."
        )
# ============================================================
# PAGE 4 — DATASET STATS
# ============================================================
elif page == "📋 Dataset Stats":
    st.title("📋 Dataset Statistics")
    if df.empty:
        st.warning("Dataset is not available.")
        st.stop()
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
    st.subheader("Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })
    st.dataframe(
        info_df,
        use_container_width=True
    )
    st.subheader("Statistical Summary")
    st.dataframe(
        df.describe(),
        use_container_width=True
    )
