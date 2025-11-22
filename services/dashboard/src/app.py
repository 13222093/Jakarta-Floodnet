import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Jakarta FloodNet", page_icon="🌊", layout="wide")

# --- FIX URL UNTUK LOCALHOST ---
# Default ke localhost:8000 jika tidak ada ENV
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #262730; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌊 Jakarta FloodNet Command Center")
st.markdown(f"### AI-Powered Early Warning System (Connected to: `{API_URL}`)")

# Cek Koneksi
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.success("🟢 API Connection: Online")
    st.sidebar.json(health['models'])
except:
    st.sidebar.error("🔴 API Connection: Offline")
    st.sidebar.info(f"Pastikan API jalan di {API_URL}")

tab1, tab2 = st.tabs(["📊 Live Monitoring", "👁️ CCTV Intelligence"])

with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("🎛️ Sensor Control")
        mode = st.radio("Mode Input", ["Manual Slider", "Auto Simulation ⚡"])
        if mode == "Manual Slider":
            rain = st.slider("Curah Hujan (mm/jam)", 0.0, 100.0, 15.5)
            water = st.slider("TMA Manggarai (cm)", 50.0, 1000.0, 650.0)
            delay = 0
        else:
            st.info("Simulating sensor data...")
            rain = np.random.gamma(2, 2) * 5
            water = np.random.normal(700, 100)
            delay = 2
            
    with col2:
        metrics_container = st.container()
        chart_container = st.empty()
        
        try:
            payload = {"rainfall_mm": rain, "water_level_cm": water, "location_id": "MANGGARAI_01"}
            start_ts = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()
            latency = round((time.time() - start_ts) * 1000, 2)

            with metrics_container:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🌧️ Rainfall", f"{rain:.1f} mm")
                m2.metric("🌊 TMA Saat Ini", f"{water:.0f} cm")
                delta_color = "inverse" if result['risk_level'] == "BAHAYA" else "normal"
                m3.metric("🔮 Prediksi TMA", f"{result['prediction_cm']:.0f} cm", 
                         delta=f"{result['prediction_cm'] - water:.1f} cm", delta_color=delta_color)
                m4.markdown(f"### Status: {result['risk_level']}")

            if 'history' not in st.session_state:
                st.session_state.history = {'time': [], 'actual': [], 'pred': []}
            st.session_state.history['time'].append(pd.Timestamp.now())
            st.session_state.history['actual'].append(water)
            st.session_state.history['pred'].append(result['prediction_cm'])
            
            if len(st.session_state.history['time']) > 50:
                for key in st.session_state.history: st.session_state.history[key].pop(0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.history['time'], y=st.session_state.history['actual'], name='Actual'))
            fig.add_trace(go.Scatter(x=st.session_state.history['time'], y=st.session_state.history['pred'], name='Predicted', line=dict(dash='dot')))
            fig.add_hline(y=150, line_dash="dash", line_color="red")
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            chart_container.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Waiting for API... ({str(e)})")

    if mode == "Auto Simulation ⚡":
        time.sleep(delay)
        st.rerun()

with tab2:
    st.subheader("👁️ Visual Flood Verification")
    uploaded_file = st.file_uploader("Upload CCTV Image", type=['jpg', 'png'])
    if uploaded_file and st.button("Analyze Image"):
        try:
            files = {"file": uploaded_file.getvalue()}
            res = requests.post(f"{API_URL}/verify-visual", files=files)
            st.json(res.json())
        except Exception as e:
            st.error(f"Error: {e}")