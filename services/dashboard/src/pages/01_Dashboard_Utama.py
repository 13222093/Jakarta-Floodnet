"""
Halaman Dashboard Utama
=======================
Monitoring status terkini sistem Jakarta FloodNet
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components import (
    api_client,
    page_header,
    status_card,
    metric_card,
    connection_status_badge,
    flood_level_indicator,
    create_water_level_chart,
    error_message,
    success_message,
    warning_message,
    sidebar_info,
    inject_responsive_css,
    COLORS
)

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Dashboard Utama - Jakarta FloodNet",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# Inject responsive CSS
inject_responsive_css()

# Sidebar
with st.sidebar:
    sidebar_info()
    
    # Connection status
    st.markdown("---")
    st.markdown("#### 🔌 Status Koneksi")
    health = api_client.check_health()
    connection_status_badge(health['success'])

# ==================== MAIN CONTENT ====================
page_header(
    "Dashboard Utama",
    "Monitoring Status Real-time Sistem Jakarta FloodNet",
    "📊"
)

# Auto-refresh toggle
col_refresh1, col_refresh2 = st.columns([3, 1])
with col_refresh2:
    auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)
    
if auto_refresh:
    st.info("⏱️ Auto refresh aktif - halaman akan di-refresh setiap 30 detik")

# ==================== STATUS OVERVIEW ====================
st.markdown("### 🎯 Status Sistem")

if not health['success']:
    error_message(
        "API Gateway tidak dapat dijangkau",
        health.get('message', 'Unknown error')
    )
    st.stop()

success_message("API Gateway terhubung dan siap digunakan")

# Get system status
system_status = api_client.get_system_status()

if system_status['success']:
    status_data = system_status.get('data', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div style="background: {COLORS['success']}15; border: 2px solid {COLORS['success']}; 
                        padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2.5rem;">🚀</div>
                <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">API Gateway</div>
                <div style="color: {COLORS['success']}; font-weight: 600; margin-top: 0.25rem;">✓ Online</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        lstm_status = status_data.get('lstm_model', 'unknown')
        is_loaded = lstm_status == 'loaded'
        color = COLORS['success'] if is_loaded else COLORS['warning']
        icon = "🔮" if is_loaded else "⚠️"
        
        st.markdown(f"""
            <div style="background: {color}15; border: 2px solid {color}; 
                        padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2.5rem;">{icon}</div>
                <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">LSTM Model</div>
                <div style="color: {color}; font-weight: 600; margin-top: 0.25rem;">{lstm_status.title()}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        yolo_status = status_data.get('yolo_model', 'unknown')
        is_loaded = yolo_status == 'loaded'
        color = COLORS['success'] if is_loaded else COLORS['warning']
        icon = "👁️" if is_loaded else "⚠️"
        
        st.markdown(f"""
            <div style="background: {color}15; border: 2px solid {color}; 
                        padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2.5rem;">{icon}</div>
                <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">YOLO Model</div>
                <div style="color: {color}; font-weight: 600; margin-top: 0.25rem;">{yolo_status.title()}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
            <div style="background: {COLORS['info']}15; border: 2px solid {COLORS['info']}; 
                        padding: 1.5rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2.5rem;">⏰</div>
                <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">Waktu Server</div>
                <div style="color: {COLORS['info']}; font-weight: 600; margin-top: 0.25rem;">{current_time}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ==================== MONITORING DATA ====================
st.markdown("### 📈 Data Monitoring")

# Generate dummy data untuk demo
# Dalam implementasi nyata, ambil dari API history
timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
water_levels = [
    120 + np.random.normal(0, 15) + (10 * np.sin(i/4)) 
    for i in range(24)
]

# Ensure water levels are realistic
water_levels = [max(50, min(250, level)) for level in water_levels]

# Current metrics
current_water_level = water_levels[-1]
previous_water_level = water_levels[-2]
delta_water = current_water_level - previous_water_level

# Responsive columns - 4 on desktop, 2 on tablet, 1 on mobile
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    metric_card(
        "💧 Ketinggian Air Terkini",
        f"{current_water_level:.1f} cm",
        f"{delta_water:+.1f} cm",
        "inverse" if delta_water > 0 else "normal"
    )

with col2:
    avg_level = np.mean(water_levels[-6:])
    metric_card(
        "📊 Rata-rata 6 Jam",
        f"{avg_level:.1f} cm"
    )

with col3:
    max_level = max(water_levels)
    metric_card(
        "📈 Level Tertinggi",
        f"{max_level:.1f} cm"
    )

with col4:
    min_level = min(water_levels)
    metric_card(
        "📉 Level Terendah",
        f"{min_level:.1f} cm"
    )

st.markdown("---")

# ==================== FLOOD LEVEL INDICATOR ====================
# Responsive layout - stacked on mobile, side-by-side on desktop
use_mobile_layout = st.session_state.get('mobile_view', False)

if use_mobile_layout:
    # Mobile: Stack vertically
    st.markdown("#### 🌊 Status Banjir")
    flood_level_indicator(current_water_level, threshold_siaga=150, threshold_bahaya=200)
    
    # Additional info
    if current_water_level < 150:
        st.success("✅ **KONDISI AMAN**  \nKetinggian air dalam batas normal")
    elif current_water_level < 200:
        st.warning("⚠️ **SIAGA BANJIR**  \nWaspada! Ketinggian air meningkat")
    else:
        st.error("🚨 **BAHAYA BANJIR**  \nEvakuasi! Ketinggian air sangat tinggi")
    
    st.markdown("#### 📊 Grafik Tren Ketinggian Air (24 Jam Terakhir)")
    
    # Create chart data
    chart_data = {
        'timestamps': timestamps,
        'water_levels': water_levels
    }
    
    fig = create_water_level_chart(chart_data)
    st.plotly_chart(fig, use_container_width=True)
else:
    # Desktop: Side by side
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Grafik Tren Ketinggian Air (24 Jam Terakhir)")
        
        # Create chart data
        chart_data = {
            'timestamps': timestamps,
            'water_levels': water_levels
        }
        
        fig = create_water_level_chart(chart_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌊 Status Banjir")
        flood_level_indicator(current_water_level, threshold_siaga=150, threshold_bahaya=200)
    
        # Additional info
        if current_water_level < 150:
            st.success("✅ **KONDISI AMAN**  \nKetinggian air dalam batas normal")
        elif current_water_level < 200:
            st.warning("⚠️ **SIAGA BANJIR**  \nWaspada! Ketinggian air meningkat")
        else:
            st.error("🚨 **BAHAYA BANJIR**  \nEvakuasi! Ketinggian air sangat tinggi")

st.markdown("---")

# ==================== RECENT ALERTS ====================
st.markdown("### 🚨 Notifikasi Terbaru")

# Dummy alerts
if current_water_level > 150:
    st.warning(f"""
        **⚠️ Peringatan Siaga**  
        Ketinggian air mencapai {current_water_level:.1f} cm (batas: 150 cm)  
        Waktu: {datetime.now().strftime("%d %B %Y, %H:%M")}
    """)

if delta_water > 10:
    st.info(f"""
        **📊 Peningkatan Signifikan**  
        Ketinggian air naik {delta_water:.1f} cm dalam 1 jam terakhir  
        Waktu: {datetime.now().strftime("%d %B %Y, %H:%M")}
    """)

st.success("""
    **✅ Sistem Berjalan Normal**  
    Semua komponen monitoring aktif dan berfungsi baik  
    Terakhir dicek: {}
""".format(datetime.now().strftime("%d %B %Y, %H:%M")))

st.markdown("---")

# ==================== QUICK ACTIONS ====================
st.markdown("### ⚡ Aksi Cepat")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔮 **Buat Prediksi Baru**", use_container_width=True):
        st.switch_page("pages/02_Prediksi_Banjir.py")

with col2:
    if st.button("👁️ **Verifikasi Visual**", use_container_width=True):
        st.switch_page("pages/03_Verifikasi_Visual.py")

with col3:
    if st.button("🔄 **Refresh Data**", use_container_width=True):
        st.rerun()

# ==================== AUTO REFRESH ====================
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
