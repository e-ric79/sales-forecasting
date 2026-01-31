import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📊 Retail Sales Forecasting Dashboard")
st.markdown(
    """
    This application forecasts **monthly retail sales** using historical data.
    Designed to support **inventory planning, budgeting, and staffing decisions**.
    """
)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=3,
    max_value=12,
    value=6
)

seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode",
    options=["multiplicative", "additive"],
    help="Multiplicative works best when sales grow over time."
)

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Sales CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file containing `Order Date` and `Sales` columns.")
    st.stop()

# --------------------------------------------------
# Data Loading & Validation
# --------------------------------------------------
try:
    df = pd.read_csv(uploaded_file, encoding="latin1")
except Exception as e:
    st.error("Unable to read file. Please upload a valid CSV.")
    st.stop()

required_cols = {"Order Date", "Sales"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain `Order Date` and `Sales` columns.")
    st.stop()

# --------------------------------------------------
# Data Preparation
# --------------------------------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df = df.dropna(subset=["Order Date", "Sales"])
df = df.sort_values("Order Date")

monthly_sales = (
    df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
    .reset_index()
)

monthly_sales["Order Date"] = monthly_sales["Order Date"].dt.to_timestamp()

# --------------------------------------------------
# Executive Summary
# --------------------------------------------------
st.subheader("📌 Executive Summary")

latest_month = monthly_sales.iloc[-1]
avg_sales = monthly_sales["Sales"].mean()
growth_rate = (
    (monthly_sales["Sales"].iloc[-1] /
     monthly_sales["Sales"].iloc[-13] - 1) * 100
    if len(monthly_sales) > 12 else np.nan
)

col1, col2, col3 = st.columns(3)

col1.metric("Latest Monthly Sales", f"${latest_month['Sales']:,.0f}")
col2.metric("Average Monthly Sales", f"${avg_sales:,.0f}")
col3.metric("YoY Growth", f"{growth_rate:.1f}%" if not np.isnan(growth_rate) else "N/A")

# --------------------------------------------------
# Historical Trend
# --------------------------------------------------
st.subheader("📈 Historical Monthly Sales")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o",
    linewidth=2
)
ax.set_xlabel("Date")
ax.set_ylabel("Sales ($)")
ax.grid(alpha=0.3)
st.pyplot(fig)

# --------------------------------------------------
# Prophet Modeling
# --------------------------------------------------
st.subheader("🤖 Forecasting Model")

df_prophet = monthly_sales.rename(
    columns={"Order Date": "ds", "Sales": "y"}
)

model = Prophet(
    seasonality_mode=seasonality_mode,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

with st.spinner("Training forecasting model..."):
    model.fit(df_prophet)

# --------------------------------------------------
# Forecast
# --------------------------------------------------
future = model.make_future_dataframe(
    periods=forecast_horizon,
    freq="MS"
)

forecast = model.predict(future)

st.subheader("🔮 Sales Forecast")

fig2 = model.plot(forecast)
plt.xlabel("Date")
plt.ylabel("Sales ($)")
st.pyplot(fig2)

# --------------------------------------------------
# Forecast Table
# --------------------------------------------------
st.subheader("📊 Forecast Details")

forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_horizon)
forecast_table.columns = [
    "Date",
    "Expected Sales",
    "Conservative Estimate",
    "Aggressive Estimate"
]

st.dataframe(
    forecast_table.style.format("${:,.0f}")
)

# --------------------------------------------------
# Business Interpretation
# --------------------------------------------------
st.subheader("💼 Business Interpretation")

st.markdown(
    f"""
    - Forecast generated for the next **{forecast_horizon} months**
    - Confidence intervals support **conservative vs aggressive planning**
    - Best used for:
        - Inventory ordering
        - Revenue planning
        - Staffing decisions
    """
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Built using Python, Prophet, and Streamlit | Designed for real-world retail forecasting"
)
