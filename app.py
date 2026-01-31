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
    help="Multiplicative works best when sales grow over time (percentage-based peaks)."
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
    st.error(f"CSV must contain columns named: {required_cols}")
    st.stop()

# --------------------------------------------------
# Data Preparation
# --------------------------------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df = df.dropna(subset=["Order Date", "Sales"])
df = df.sort_values("Order Date")

# Aggregate to Monthly
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
    (monthly_sales["Sales"].iloc[-1] / monthly_sales["Sales"].iloc[-13] - 1) * 100
    if len(monthly_sales) > 12 else np.nan
)

col1, col2, col3 = st.columns(3)
col1.metric("Latest Monthly Sales", f"${latest_month['Sales']:,.0f}")
col2.metric("Average Monthly Sales", f"${avg_sales:,.0f}")
col3.metric("YoY Growth", f"{growth_rate:.1f}%" if not np.isnan(growth_rate) else "N/A")

# --------------------------------------------------
# Prophet Modeling
# --------------------------------------------------
df_prophet = monthly_sales.rename(columns={"Order Date": "ds", "Sales": "y"})

model = Prophet(
    seasonality_mode=seasonality_mode,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

with st.spinner("🤖 Training model and calculating accuracy..."):
    model.fit(df_prophet)
    
    # Forecast
    future = model.make_future_dataframe(periods=forecast_horizon, freq="MS")
    forecast = model.predict(future)

# --------------------------------------------------
# Model Accuracy (MAPE)
# --------------------------------------------------
st.subheader("🎯 Model Performance")

# Compare actuals with the model's fitted values (in-sample)
actuals = df_prophet['y']
predictions = forecast.iloc[:len(df_prophet)]['yhat']
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
accuracy = 100 - mape

acc_col1, acc_col2 = st.columns(2)
acc_col1.metric("Model Accuracy", f"{accuracy:.1f}%")
acc_col2.metric("Mean Absolute % Error", f"{mape:.1f}%")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Forecast Plot", "🕵️ Trend Components", "📄 Data Table"])

with tab1:
    st.subheader("🔮 Sales Forecast")
    fig_forecast = model.plot(forecast)
    plt.xlabel("Date")
    plt.ylabel("Sales ($)")
    st.pyplot(fig_forecast)

with tab2:
    st.subheader("🔍 Breakdown of Trends")
    st.write("This shows the long-term trend and the yearly seasonal patterns (e.g., holiday peaks).")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

with tab3:
    st.subheader("📊 Forecast Details")
    
    # Clean up table for display
    forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_horizon)
    forecast_table.columns = ["Date", "Expected Sales", "Lower Bound", "Upper Bound"]
    
    st.dataframe(forecast_table.style.format({
        "Expected Sales": "${:,.0f}",
        "Lower Bound": "${:,.0f}",
        "Upper Bound": "${:,.0f}"
    }))

    # CSV Download Button
    csv = forecast_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name='retail_forecast.csv',
        mime='text/csv',
    )

# --------------------------------------------------
# Business Interpretation
# --------------------------------------------------
st.markdown("---")
st.subheader("💼 Business Strategy Guide")
st.info(f"""
- **Aggressive Growth:** Plan inventory based on the **Upper Bound** (${forecast_table['Upper Bound'].iloc[-1]:,.0f}).
- **Risk Mitigation:** Use the **Lower Bound** (${forecast_table['Lower Bound'].iloc[-1]:,.0f}) for emergency cash-flow budgeting.
- **Seasonality:** Check the 'Trend Components' tab to identify which months require extra staffing.
""")

st.caption("Built with Python & Prophet | Data-driven decision support for Retail.")