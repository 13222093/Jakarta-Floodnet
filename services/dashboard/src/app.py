import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Jakarta FloodNet",
    page_icon="🌊",
    layout="wide"
)

# URL API (Ambil dari Env Docker atau Default Localhost)
API_URL = os.getenv("API_URL", "http://api_gateway:8000")

# Custom CSS biar tampilan makin sangar
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== HEADER =====
st.title("🌊 Jakarta FloodNet Command Center")
st.markdown("### AI-Powered Early Warning System")

# Cek Koneksi API
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    status_icon = "🟢" if health['status'] == 'active' else "🔴"
    st.sidebar.success(f"{status_icon} API Connection: Connected")
    st.sidebar.json(health['models'])
except:
    st.sidebar.error("🔴 API Connection: Offline")
    st.sidebar.info("Pastikan container api_gateway menyala!")

# ===== TABS =====
tab1, tab2 = st.tabs(["📊 Live Monitoring", "👁️ CCTV Intelligence"])

# ---------------------------------------------------------------------
# TAB 1: LIVE MONITORING (FORECASTING)
# ---------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("🎛️ Sensor Control")
        # Input Manual / Simulasi
        mode = st.radio("Mode Input", ["Manual Slider", "Auto Simulation ⚡"])
        
        if mode == "Manual Slider":
            rain = st.slider("Curah Hujan (mm/jam)", 0.0, 100.0, 15.5)
            water = st.slider("TMA Manggarai (cm)", 50.0, 1000.0, 650.0)
            delay = 0
        else:
            # Simulasi Data Acak (Biar grafik gerak)
            st.info("Simulating sensor data stream...")
            rain = np.random.gamma(2, 2) * 5  # Distribusi hujan realistis
            water = np.random.normal(700, 100) # Fluktuasi TMA
            delay = 2 # Refresh tiap 2 detik
            
    with col2:
        # Placeholder untuk update real-time
        metrics_container = st.container()
        chart_container = st.empty()

        # Logic Request ke API
        payload = {
            "rainfall_mm": rain,
            "water_level_cm": water,
            "location_id": "MANGGARAI_01"
        }
        
        try:
            # PANGGIL API KITA!
            start_ts = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()
            latency = round((time.time() - start_ts) * 1000, 2)

            # Tampilkan Metrics Utama
            with metrics_container:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🌧️ Rainfall", f"{payload['rainfall_mm']:.1f} mm")
                m2.metric("🌊 TMA Saat Ini", f"{payload['water_level_cm']:.0f} cm")
                
                # Warna Alert
                risk = result['risk_level']
                delta_color = "normal"
                if risk == "BAHAYA": delta_color = "inverse"
                
                m3.metric("🔮 Prediksi TMA (+1h)", f"{result['prediction_cm']:.0f} cm", 
                         delta=f"{result['prediction_cm'] - payload['water_level_cm']:.1f} cm",
                         delta_color=delta_color)
                
                status_color = "green" if risk == "AMAN" else "red"
                m4.markdown(f"### Status: :{status_color}[{risk}]")
            
            # Tampilkan Pesan Alert
            if risk != "AMAN":
                st.error(f"🚨 ALERT: {result['alert_message']}")
            else:
                st.success(f"✅ {result['alert_message']}")

            # Simpan History untuk Grafik (Session State)
            if 'history' not in st.session_state:
                st.session_state.history = {'time': [], 'actual': [], 'pred': []}
            
            # Update Data Grafik
            st.session_state.history['time'].append(pd.Timestamp.now())
            st.session_state.history['actual'].append(payload['water_level_cm'])
            st.session_state.history['pred'].append(result['prediction_cm'])
            
            # Batasi cuma 50 data terakhir biar enteng
            if len(st.session_state.history['time']) > 50:
                for key in st.session_state.history:
                    st.session_state.history[key].pop(0)
            
            # Gambar Grafik Keren pakai Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.history['time'], y=st.session_state.history['actual'],
                                    mode='lines+markers', name='Actual TMA'))
            fig.add_trace(go.Scatter(x=st.session_state.history['time'], y=st.session_state.history['pred'],
                                    mode='lines', name='Predicted (+1h)', line=dict(dash='dot')))
            
            # Garis Batas Bahaya
            fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Threshold Siaga")
            
            fig.update_layout(title="Real-time Water Level Monitoring", height=400, template="plotly_dark")
            chart_container.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"⏱️ API Latency: {latency} ms | Model Status: {result['status']}")

        except Exception as e:
            st.warning(f"Waiting for API... ({str(e)})")

    # Auto Refresh Logic (Looping curang ala Streamlit)
    if mode == "Auto Simulation ⚡":
        time.sleep(delay)
        st.rerun()

# ---------------------------------------------------------------------
# TAB 2: CCTV INTELLIGENCE (VISUAL VERIFICATION)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("👁️ Visual Flood Verification")
    
    col_upload, col_result = st.columns(2)
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload CCTV/Drone Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image 🔍"):
                with st.spinner("Processing with YOLOv8 + Color Analysis..."):
                    try:
                        # Kirim file ke API
                        files = {"file": uploaded_file.getvalue()}
                        res = requests.post(f"{API_URL}/verify-visual", files=files)
                        vision_result = res.json()
                        
                        # Tampilkan Hasil di kolom sebelah
                        with col_result:
                            st.write("### 🎯 Analysis Result")
                            
                            # Tampilkan Metrics Visual
                            prob = vision_result.get('flood_probability', 0)
                            st.progress(prob, text=f"Flood Probability: {prob*100:.1f}%")
                            
                            if vision_result.get('is_flooded'):
                                st.error("🚨 FLOOD DETECTED")
                            else:
                                st.success("✅ NO FLOOD DETECTED")
                                
                            st.json(vision_result)
                            
                    except Exception as e:
                        st.error(f"Error connecting to API: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 Jakarta FloodNet Team | Built for Hackathon")