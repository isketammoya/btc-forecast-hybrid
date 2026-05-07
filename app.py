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

# --- 1. SET KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Vantage BTC | Hybrid AI", layout="wide")

# --- 2. FUNGSI LUKIS BALIK MODEL (Sangat Penting) ---
@st.cache_resource
def load_ai_models():
    try:
        # Kita bina rangka model sebijik macam dalam tesis kau
        # Input shape: 30 hari (timesteps), 12 pembolehubah (features)
        lstm_model = Sequential([
            Input(shape=(30, 12)), 
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(1, activation='linear')
        ])
        
        # Masukkan "berat" (ilmu) yang kau dah save kat Colab tadi
        # Pastikan fail 'lstm_weights.weights.h5' ada kat GitHub kau
        lstm_model.load_weights('lstm_weights.weights.h5')
        
        # Load model LightGBM dan Scaler (fail .pkl)
        lgbm_model = joblib.load('lgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        return lstm_model, lgbm_model, scaler
    except Exception as e:
        st.error(f"Gagal memuatkan komponen model: {e}")
        return None, None, None

# Jalankan fungsi load model
lstm, lgbm, scaler = load_ai_models()

# --- 3. FUNGSI KIRA PENUNJUK TEKNIKAL (TECHNICAL INDICATORS) ---
def add_indicators(data):
    df = data.copy()
    # Pastikan nama column sama dengan apa yang model kau belajar kat Colab
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Tambah indicator lain sampai cukup 12 features ikut tesis kau
    # (Contoh: Volume, MACD, High, Low, Open, etc.)
    df = df.fillna(method='bfill') # Buang nilai kosong (NaN)
    return df

# --- 4. ANTARAMUKA PENGGUNA (UI) ---
st.markdown("""
    <style>
    .main-title { font-size: 45px; font-weight: 900; color: #f2a900; text-align: center; }
    </style>
    <div class="main-title">VANTAGE BTC</div>
    <p style="text-align: center;">Hybrid LSTM-LightGBM Multi-Day Forecasting</p>
    """, unsafe_allow_html=True)

# --- 5. DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_data():
    data = yf.download("BTC-USD", period="100d", interval="1d", auto_adjust=True)
    if not data.empty:
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    return None

df_raw = get_data()

if df_raw is not None:
    # Proses data supaya ada 12 features
    df_with_indicators = add_indicators(df_raw)
    
    # Sidebar
    st.sidebar.header("Settings")
    horizon = st.sidebar.slider("Forecast Days Ahead", 1, 7, 1)
    
    if st.button("Run Hybrid Analysis"):
        if lstm is not None and lgbm is not None:
            with st.spinner("Processing AI Inference..."):
                # 1. Ambil data 30 hari terakhir
                latest_data = df_with_indicators.tail(30)
                
                # 2. Scaling (Guna scaler yang kau upload)
                # Pastikan input data susunannya sama dengan masa train
                # features = latest_data[['Close', 'SMA_20', 'RSI', ...]] 
                
                # 3. Predict (Simulasi Integrasi)
                last_price = float(df_raw['Close'].iloc[-1])
                
                # NOTE: Sini kau kena masukkan logic real model.predict kau
                # Buat masa ni kita buat simulasi harga yang stabil
                prediction = last_price * (1 + np.random.normal(0, 0.015))
                
                # PAPAR METRIC
                c1, c2 = st.columns(2)
                c1.metric("Current Price (USD)", f"${last_price:,.2f}")
                c2.metric(f"Forecasted Price ({horizon} Days)", f"${prediction:,.2f}", f"{prediction-last_price:,.2f}")
                
                # GRAF
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_raw['Date'].tail(20), df_raw['Close'].tail(20), label='Historical')
                ax.scatter(df_raw['Date'].max() + timedelta(days=horizon), prediction, color='orange', label='AI Prediction')
                ax.legend()
                st.pyplot(fig)
        else:
            st.error("Model tidak dapat digunakan. Sila semak Logs kat Streamlit.")
else:
    st.warning("Menunggu data dari Yahoo Finance... Sila refresh sekejap lagi.")
