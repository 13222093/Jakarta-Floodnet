"""
Jakarta FloodNet Dashboard
===========================
Entry Point untuk Dashboard Monitoring Banjir berbasis Streamlit

Author: Jakarta FloodNet Team
Date: November 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent))

from components import (
    api_client,
    sidebar_info,
    connection_status_badge,
    error_message,
    inject_responsive_css
)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Jakarta FloodNet Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="auto",  # Changed to auto for better mobile UX
    menu_items={
        'Get Help': 'https://github.com/Jakarta-FloodNet',
        'Report a bug': 'https://github.com/Jakarta-FloodNet/issues',
        'About': """
        # Jakarta FloodNet 🌊
        
        Sistem Monitoring dan Prediksi Banjir berbasis AI
        
        **Fitur Utama:**
        - Prediksi ketinggian air dengan LSTM
        - Verifikasi visual dengan YOLO
        - Real-time monitoring
        
        **Teknologi:**
        - Frontend: Streamlit
        - Backend: FastAPI
        - ML: TensorFlow, PyTorch
        """
    }
)

# ==================== CUSTOM CSS ====================
inject_responsive_css()

st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 8px;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Responsive header */
    @media (max-width: 768px) {
        .main h1 {
            font-size: 1.5rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# 🌊 Jakarta FloodNet")
    st.markdown("### Sistem Monitoring Banjir AI")
    st.markdown("---")
    
    # Check API connection
    st.markdown("#### 🔌 Status Koneksi")
    health = api_client.check_health()
    
    if health['success']:
        connection_status_badge(True)
        st.caption(f"✓ {health['message']}")
    else:
        connection_status_badge(False)
        st.caption(f"✗ {health['message']}")
    
    st.markdown("---")
    
    # Navigation info
    st.markdown("""
        #### 📱 Navigasi
        Gunakan menu di atas untuk:
        - 📊 **Dashboard Utama**: Monitoring real-time
        - 🔮 **Prediksi Banjir**: Kalkulator LSTM
        - 👁️ **Verifikasi Visual**: Analisis gambar YOLO
    """)
    
    # Add sidebar info
    sidebar_info()

# ==================== MAIN PAGE ====================
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    ">
        <h1 style="margin: 0; font-size: 3rem; font-weight: 800;">
            🌊 Jakarta FloodNet
        </h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.5rem; opacity: 0.95;">
            Sistem Monitoring dan Prediksi Banjir Berbasis AI
        </p>
    </div>
""", unsafe_allow_html=True)

# Welcome section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            height: 100%;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">�</div>
            <h3 style="margin: 0;">Prediksi LSTM</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">
                Forecasting ketinggian air berdasarkan data curah hujan
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            height: 100%;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👁️</div>
            <h3 style="margin: 0;">Verifikasi Visual</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">
                Deteksi kondisi banjir menggunakan YOLO computer vision
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            height: 100%;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="margin: 0;">Real-time Monitor</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">
                Monitoring status dan tren ketinggian air secara real-time
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick Start Guide
st.markdown("## 🚀 Panduan Cepat")

st.markdown("""
### Cara Menggunakan Dashboard

1. **📊 Dashboard Utama**
   - Lihat status terkini sistem monitoring
   - Monitor tren ketinggian air
   - Cek health status semua komponen

2. **🔮 Prediksi Banjir (LSTM)**
   - Masukkan data curah hujan Bogor dan Jakarta
   - Klik tombol "Hitung Prediksi"
   - Dapatkan prediksi level air dan tingkat risiko

3. **👁️ Verifikasi Visual (YOLO)**
   - Upload gambar kondisi sungai/jalan
   - Klik "Analisis Gambar"
   - Lihat hasil deteksi visual kondisi banjir

### 🔧 Prasyarat
- Pastikan API Gateway berjalan di `http://localhost:8000`
- Model LSTM dan YOLO sudah ter-load di backend
- Koneksi internet stabil untuk request ke API

### ⚡ Tips
- Gunakan mode **wide** untuk tampilan optimal
- Refresh halaman jika koneksi terputus
- Cek sidebar untuk status koneksi real-time
""")

st.markdown("---")

# System Status Overview
if health['success']:
    st.success("✅ **Sistem Siap Digunakan** - Semua komponen terhubung dengan baik")
    
    # Try to get system status
    status = api_client.get_system_status()
    if status['success']:
        st.markdown("### 🎯 Status Komponen")
        col1, col2, col3 = st.columns(3)
        
        status_data = status.get('data', {})
        
        with col1:
            st.info("**API Gateway**  \n✓ Online")
        with col2:
            lstm_status = status_data.get('lstm_model', 'unknown')
            st.info(f"**LSTM Model**  \n{'✓' if lstm_status == 'loaded' else '⚠'} {lstm_status.title()}")
        with col3:
            yolo_status = status_data.get('yolo_model', 'unknown')
            st.info(f"**YOLO Model**  \n{'✓' if yolo_status == 'loaded' else '⚠'} {yolo_status.title()}")
else:
    error_message(
        "API Gateway tidak dapat dijangkau",
        f"Detail: {health.get('message', 'Unknown error')}\n\n"
        "Pastikan backend API Gateway sudah berjalan di http://localhost:8000"
    )
    
    st.markdown("""
        ### 🔧 Troubleshooting
        
        1. Cek apakah API Gateway sudah berjalan:
           ```bash
           cd services/api_gateway/src
           python main.py
           ```
        
        2. Verifikasi port 8000 tidak digunakan aplikasi lain
        
        3. Cek log API Gateway untuk error messages
    """)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p><strong>Jakarta FloodNet</strong> - AI-Powered Flood Monitoring System</p>
        <p>© 2025 Jakarta FloodNet Team. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)