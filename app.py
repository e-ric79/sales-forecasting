import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Forecaster",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Retail Sales Forecaster")
st.markdown("Upload any sales CSV. This app automatically detects your data structure.")

# --------------------------------------------------
# File Upload & Flexible Data Loading
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    # Load raw data
    raw_df = pd.read_csv(uploaded_file, encoding="latin1")
    all_cols = raw_df.columns.tolist()

    # --- Sidebar: Dynamic Mapping ---
    st.sidebar.header("🛠 Configuration")
    
    st.sidebar.subheader("Column Mapping")
    # Smart default detection
    def_date = next((c for c in all_cols if 'date' in c.lower() or 'time' in c.lower()), all_cols[0])
    def_sales = next((c for c in all_cols if 'sale' in c.lower() or 'rev' in c.lower()), all_cols[-1])
    
    date_col = st.sidebar.selectbox("Date Column", all_cols, index=all_cols.index(def_date))
    sales_col = st.sidebar.selectbox("Sales Column", all_cols, index=all_cols.index(def_sales))
    
    # Optional Category Filter
    cat_col = st.sidebar.selectbox("Category Column (Optional)", ["None"] + all_cols)

    # --- Data Cleaning & Filtering ---
    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])
    
    if cat_col != "None":
        unique_cats = df[cat_col].unique().tolist()
        selected_cat = st.sidebar.multiselect("Filter by Category", unique_cats)
        if selected_cat:
            df = df[df[cat_col].isin(selected_cat)]

    # Aggregate to Monthly
    monthly_sales = df.groupby(df[date_col].dt.to_period("M"))[sales_col].sum().reset_index()
    monthly_sales[date_col] = monthly_sales[date_col].dt.to_timestamp()
    
    # Prepare for Prophet
    df_prophet = monthly_sales.rename(columns={date_col: "ds", sales_col: "y"})

    # --------------------------------------------------
    # Forecasting Logic
    # --------------------------------------------------
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 3, 24, 6)
    
    model = Prophet(yearly_seasonality=True, interval_width=0.95)
    model.add_country_holidays(country_name='US') # Adds Holiday Effects
    
    with st.spinner("🔮 Generating Forecast..."):
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_horizon, freq="MS")
        forecast = model.predict(future)

    # --------------------------------------------------
    # Results & Visualization
    # --------------------------------------------------
    # 1. Executive Metrics
    st.subheader("📌 Performance Summary")
    actuals = df_prophet['y']
    preds = forecast.iloc[:len(df_prophet)]['yhat']
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Average Monthly Sales", f"${actuals.mean():,.0f}")
    m2.metric("Model Accuracy", f"{100-mape:.1f}%")
    m3.metric("Projected Total (Forecast Period)", f"${forecast['yhat'].tail(forecast_horizon).sum():,.0f}")

    # 2. Tabs for visualization
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Plot", "🔍 Trend Analysis", "📥 Export Data"])
    
    with tab1:
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        
    with tab2:
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
        
    with tab3:
        # Prepare table
        res_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
        res_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        st.dataframe(res_table.style.format("${:,.0f}", subset=['Forecast', 'Lower Bound', 'Upper Bound']))
        
        csv = res_table.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

else:
    st.info("Please upload a CSV file to begin.")