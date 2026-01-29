📊 Sales Forecasting for Retail Businesses

Time Series Forecasting | Business Analytics

Overview

This project builds a monthly sales forecasting system for a retail business using historical sales data. The goal was to help decision-makers plan inventory, staffing, and budgets by predicting sales 6 months into the future while accounting for uncertainty and seasonality.

Problem

Retail businesses often make operational decisions without reliable forecasts, leading to:

Overstocking → wasted capital and storage costs

Understocking → lost sales and customer dissatisfaction

Accurate forecasting is essential for balancing supply and demand.

Objective

Forecast monthly sales 6 months ahead

Identify seasonal and long-term trends

Provide confidence intervals for conservative and aggressive planning

Use models effective on small datasets

Data

Superstore retail sales dataset

~48 months of historical data

Daily transactions aggregated into monthly sales

Approach

Data Preparation

Cleaned and transformed dates

Aggregated sales to monthly totals

Exploratory Analysis

Visualized sales trends

Performed seasonal decomposition (trend, seasonality, residuals)

Modeling

Evaluated Prophet vs ML approaches (XGBoost)

Selected Prophet for its strong seasonality handling and interpretability

Trained and validated using a train/test split

Forecasting

Retrained on full dataset

Generated 6-month forward forecast with uncertainty bounds

Key Insights

Strong Nov–Dec seasonality (~60% sales spike)

Significant Jan–Feb decline

Consistent ~15% year-over-year growth

Unexpected March surge, likely tied to Q1 activity

Model Performance

MAE: ~$12,000

RMSE: ~$15,000

R²: ~0.57

Prophet outperformed more complex models by effectively capturing recurring seasonal patterns.

Business Impact

Enables proactive inventory and staffing decisions

Supports conservative vs aggressive budgeting using forecast ranges

A single accurate forecast identified a potential $35k+ revenue upside by preventing stock-outs during peak demand

Tech Stack

Python | pandas | NumPy | Prophet | Matplotlib | statsmodels | scikit-learn

Key Takeaways

Simpler, domain-specific models can outperform complex ML

Seasonality is critical in retail forecasting

Forecast uncertainty is as important as point predictions

Strong business context turns models into impact
