# Streamlit front end

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ratnapura Rainfall Forecast", layout="wide")

bundle = pickle.load(open("model.pkl", "rb"))
model, features = bundle["model"], bundle["features"]
monthly = pd.read_csv("monthly.csv", index_col=0, parse_dates=True)
results = pd.read_csv("results.csv", index_col=0)
backtest = pd.read_csv("backtest.csv", index_col=0, parse_dates=True)


def next_month_row(m):
    """Build the feature row for the month after the last one in the data."""
    nxt = m.index[-1] + pd.DateOffset(months=1)
    row = {}
    for col in ["rain_mm", "temp_mean", "wind_max"]:
        for lag in [1, 2, 3, 12]:
            row[f"{col}_lag{lag}"] = m[col].iloc[-lag]
    row["rain_roll3"] = m["rain_mm"].iloc[-3:].mean()
    row["month_sin"] = np.sin(2 * np.pi * nxt.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * nxt.month / 12)
    return nxt, pd.DataFrame([row])[features]


st.title("Ratnapura Monthly Rainfall Forecast")
st.caption("Random Forest on lagged weather features. Data: Open-Meteo reanalysis, 2010 to 2025.")

tab1, tab2, tab3 = st.tabs(["Forecast", "Model comparison", "Data"])

with tab1:
    nxt, X_next = next_month_row(monthly)
    pred = max(float(model.predict(X_next)[0]), 0)
    same_month_last_year = monthly["rain_mm"].iloc[-12]

    c1, c2 = st.columns(2)
    c1.metric(f"Predicted rainfall for {nxt:%B %Y}", f"{pred:.0f} mm",
              delta=f"{pred - same_month_last_year:+.0f} mm vs last year",
              delta_color="off")
    c2.metric("Same month last year", f"{same_month_last_year:.0f} mm")

    fig, ax = plt.subplots(figsize=(10, 4))
    recent = monthly["rain_mm"].iloc[-36:]
    ax.plot(recent.index, recent, label="Actual", color="black")
    ax.scatter([nxt], [pred], color="red", zorder=5, label="Forecast")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Test-set results (2022 to 2025, lower RMSE is better)")
    st.dataframe(results.round(3), use_container_width=True)

    options = [c for c in backtest.columns if c != "actual"]
    defaults = [m for m in ["Random Forest", "CatBoost", "ANN", "SARIMA"] if m in options]
    chosen = st.multiselect("Models to plot", options, default=defaults)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(backtest.index, backtest["actual"], color="black", linewidth=2, label="Actual")
    for c in chosen:
        ax.plot(backtest.index, backtest[c], alpha=0.8, label=c)
    ax.set_ylabel("Rainfall (mm)")
    ax.legend(loc="upper center", ncol=4, fontsize=8)
    st.pyplot(fig)

with tab3:
    st.dataframe(monthly.round(2), use_container_width=True)
    st.line_chart(monthly["rain_mm"])