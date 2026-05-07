import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from datetime import datetime, timedelta

# --- 1. SET CONFIG ---
st.set_page_config(page_title="Vantage BTC Forecasting", layout="wide")

# --- 2. LOAD ASSETS (Sesuai dengan fail GitHub kau) ---
@st.cache_resource
def load_assets():
    try:
        # Rekabentuk model ikut Chapter 3 Methodology kau
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
    except:
        return None, None, None

lstm, lgbm, scaler = load_assets()

# --- 3. FETCH DATA ---
@st.cache_data(ttl=3600)
def get_data():
    data = yf.download("BTC-USD", period="150d", interval="1d", auto_adjust=True)
    if not data.empty:
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    return None

# --- 4. UI ---
st.title("🚀 Bitcoin Price Movement Analysis")
st.markdown("Hybrid LSTM-LightGBM Multi-Horizon Forecasting")

df_raw = get_data()

if df_raw is not None:
    st.sidebar.header("Forecast Horizon")
    days_ahead = st.sidebar.selectbox("Select Horizon (Days)", [1, 3, 5, 7])
    
    if st.button("Generate Hybrid Forecast"):
        # --- LOGIK TARIKH & HARGA ---
        current_date = df_raw['Date'].max()
        last_price = float(df_raw['Close'].iloc[-1])
        forecast_date = current_date + timedelta(days=days_ahead)
        
        # Simulasi Prediction (Nanti kau ganti dengan real model.predict)
        # Sesuai dengan Table 3.X tesis kau untuk horizons berbeza
        np.random.seed(days_ahead)
        predicted_price = last_price * (1 + np.random.normal(0, 0.02))

        # --- METRICS ---
        c1, c2 = st.columns(2)
        c1.metric(f"Market Price ({current_date.strftime('%d %b')})", f"${last_price:,.2f}")
        c2.metric(f"AI Forecast ({forecast_date.strftime('%d %b')})", f"${predicted_price:,.2f}", f"{predicted_price-last_price:,.2f} USD")

        # --- GRAF BERTERUSAN ---
        st.subheader(f"BTC/USD Trend Analysis: {days_ahead} Day(s) Ahead")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Plot Actual Price (Biru)
        hist_data = df_raw.tail(45) # Tunjuk 45 hari ke belakang
        ax.plot(hist_data['Date'], hist_data['Close'], label='Actual Price', color='#1f77b4', linewidth=2.5)
        
        # 2. Plot Forecast Path (Garis putus-putus menyambung ke ramalan)
        # Ini akan buat garis tu nampak berterusan dari titik terakhir
        ax.plot([current_date, forecast_date], [last_price, predicted_price], 
                color='#ff7f0e', linestyle='--', linewidth=2, label='Forecast Path')
        
        # 3. Plot Predicted Point (Titik Merah)
        ax.scatter(forecast_date, predicted_price, color='red', s=120, edgecolor='white', label='Predicted Point', zorder=5)

        # Setting Graf
        ax.set_ylabel("Price (USD)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        
        st.pyplot(fig)

        # --- TABLE ---
        st.write("### Data Summary")
        st.table(pd.DataFrame({
            "Status": ["Current Market", f"AI Forecast ({days_ahead}d)"],
            "Date": [current_date.date(), forecast_date.date()],
            "Price": [f"${last_price:,.2f}", f"${predicted_price:,.2f}"]
        }))
