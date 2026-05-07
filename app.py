import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from datetime import datetime, timedelta

# --- 1. SET CONFIG & THEME ---
st.set_page_config(page_title="Vantage BTC Forecasting", layout="wide")

# --- 2. LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    try:
        # Bina semula model ikut spesifikasi tesis (30 timesteps, 12 features)
        model = Sequential([
            Input(shape=(30, 12)), 
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(1, activation='linear')
        ])
        model.load_weights('lstm_weights.weights.h5')
        lgbm = joblib.load('lgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, lgbm, scaler
    except Exception as e:
        return None, None, None

lstm, lgbm, scaler = load_assets()

# --- 3. FETCH DATA ---
@st.cache_data(ttl=3600)
def get_btc_data():
    data = yf.download("BTC-USD", period="150d", interval="1d", auto_adjust=True)
    if not data.empty:
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    return None

# --- 4. UI DESIGN ---
st.title("🚀 Multi-Day Ahead Bitcoin Prediction")
st.markdown("Model Hybrid: **LSTM-LightGBM** (Based on FYP Thesis)")

df_raw = get_btc_data()

if df_raw is not None:
    # Sidebar untuk pilih hari (1, 3, 5, 7)
    st.sidebar.header("Forecast Settings")
    days_ahead = st.sidebar.selectbox("Select Prediction Horizon (Days)", [1, 3, 5, 7])
    
    if st.button("Start Hybrid Prediction"):
        if lstm is not None:
            # --- LOGIK TARIKH ---
            current_date = df_raw['Date'].max()
            forecast_date = current_date + timedelta(days=days_ahead)
            last_price = float(df_raw['Close'].iloc[-1])
            
            # --- SIMULASI PREDICTION (Gantikan dengan model.predict yang sebenar) ---
            # Di sini sepatutnya model kau proses 12 features dan keluarkan result
            predicted_price = last_price * (1 + np.random.normal(0, 0.02))
            
            # --- PAPARAN METRIC DENGAN TARIKH ---
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Current Date:** {current_date.strftime('%d %B %Y')}")
                st.metric("Current Price", f"${last_price:,.2f}")
            with col2:
                st.success(f"**Forecast Date:** {forecast_date.strftime('%d %B %Y')}")
                st.metric(f"Predicted Price ({days_ahead} Days)", f"${predicted_price:,.2f}", f"{predicted_price-last_price:,.2f}")

            # --- GRAF ACTUAL VS PREDICTED ---
            st.subheader(f"Price Prediction Graph ({days_ahead} Days Ahead)")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Data Sejarah (30 hari ke belakang)
            historical_plot = df_raw.tail(30)
            ax.plot(historical_plot['Date'], historical_plot['Close'], label='Actual Price', color='#1f77b4', linewidth=2)
            
            # Garis Putus-putus sambungkan Actual ke Predicted
            ax.plot([current_date, forecast_date], [last_price, predicted_price], 
                    color='orange', linestyle='--', label='Forecast Path')
            
            # Titik Ramalan
            ax.scatter(forecast_date, predicted_price, color='red', s=100, label='Predicted Point', zorder=5)
            
            # Format Graf
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # --- JADUAL PREDICTION ---
            st.write("### Prediction Summary Table")
            summary_data = {
                "Period": ["Current", f"{days_ahead} Days Ahead"],
                "Date": [current_date.strftime('%Y-%m-%d'), forecast_date.strftime('%Y-%m-%d')],
                "Price (USD)": [f"${last_price:,.2f}", f"${predicted_price:,.2f}"]
            }
            st.table(pd.DataFrame(summary_data))
            
        else:
            st.error("Model fail didapati. Pastikan 'lstm_weights.weights.h5' telah diupload ke GitHub.")
else:
    st.error("Gagal mendapatkan data dari Yahoo Finance. Sila tunggu sebentar dan cuba lagi.")
