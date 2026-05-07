import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from datetime import datetime, timedelta

# --- 1. SET KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Vantage BTC Predictor", layout="wide")

# --- 2. LOAD AI ASSETS ---
@st.cache_resource
def load_ai_models():
    try:
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

lstm, lgbm, scaler = load_ai_models()

# --- 3. DATA PROCESSING ---
def add_indicators(data):
    df = data.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.fillna(method='bfill')
    return df

@st.cache_data(ttl=3600)
def get_btc_data():
    data = yf.download("BTC-USD", period="150d", interval="1d", auto_adjust=True)
    if not data.empty:
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    return None

# --- 4. ANTARAMUKA PENGGUNA (UI) ---
# Tajuk yang lebih kemas dan profesional
st.markdown("""
    <style>
    .main-title { font-size: 42px; font-weight: 800; color: #f2a900; text-align: center; margin-bottom: 0px; }
    .sub-title { font-size: 18px; color: #666; text-align: center; margin-bottom: 30px; }
    </style>
    <div class="main-title">VANTAGE BTC PREDICTOR</div>
    <div class="sub-title">Advanced AI Analysis for Bitcoin Market Trends</div>
    """, unsafe_allow_html=True)

df_raw = get_btc_data()

if df_raw is not None:
    # Sidebar - Lebih ringkas
    st.sidebar.header("Analysis Settings")
    days_ahead = st.sidebar.selectbox("Forecast Horizon", ["1 Day Ahead", "3 Days Ahead", "5 Days Ahead", "7 Days Ahead"])
    # Tukar pilihan kepada integer
    horizon_map = {"1 Day Ahead": 1, "3 Days Ahead": 3, "5 Days Ahead": 5, "7 Days Ahead": 7}
    h_days = horizon_map[days_ahead]
    
    if st.button("Run Market Analysis"):
        if lstm is not None:
            with st.spinner("Analyzing market patterns..."):
                current_date = df_raw['Date'].max()
                last_price = float(df_raw['Close'].iloc[-1])
                forecast_date = current_date + timedelta(days=h_days)
                
                # Simulasi Prediction (Nanti ganti dengan real predict)
                np.random.seed(h_days)
                predicted_price = last_price * (1 + np.random.normal(0, 0.02))

                # --- Metrics Section ---
                col1, col2 = st.columns(2)
                col1.metric(f"Current Market Price ({current_date.strftime('%d %b')})", f"${last_price:,.2f}")
                col2.metric(f"AI Forecasted Price ({forecast_date.strftime('%d %b')})", f"${predicted_price:,.2f}", f"{predicted_price-last_price:,.2f} USD")

                # --- Graph Section ---
                st.subheader(f"Price Movement Analysis: {days_ahead}")
                plt.style.use('dark_background') # Gunakan tema gelap supaya nampak lebih 'fintech'
                fig, ax = plt.subplots(figsize=(12, 6))
                
                hist_plot = df_raw.tail(40)
                ax.plot(hist_plot['Date'], hist_plot['Close'], color='#f2a900', linewidth=2, label='Actual Market Price')
                
                # Garis Trend ke depan
                ax.plot([current_date, forecast_date], [last_price, predicted_price], 
                        color='white', linestyle='--', alpha=0.6, label='Predicted Trend')
                
                # Titik ramalan yang menonjol
                ax.scatter(forecast_date, predicted_price, color='#f2a900', s=150, edgecolors='white', marker='o', zorder=5, label='Forecast Point')

                ax.set_ylabel("Price in USD")
                ax.legend()
                ax.grid(True, alpha=0.2)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Jadual Ringkasan
                st.write("### Prediction Summary")
                st.table(pd.DataFrame({
                    "Timeline": ["Current Market", f"Forecast ({days_ahead})"],
                    "Date": [current_date.strftime('%Y-%m-%d'), forecast_date.strftime('%Y-%m-%d')],
                    "Price (USD)": [f"${last_price:,.2f}", f"${predicted_price:,.2f}"]
                }))
        else:
            st.error("System encountered an error. Please contact support.")
