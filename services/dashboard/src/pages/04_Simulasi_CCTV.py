"""
Halaman Simulasi CCTV Multi-Channel
====================================
Simulasi monitoring CCTV dengan deteksi banjir otomatis menggunakan YOLO
"""

import streamlit as st
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components import (
    api_client,
    page_header,
    status_card,
    connection_status_badge,
    error_message,
    success_message,
    warning_message,
    sidebar_info,
    inject_responsive_css,
    render_icon,
    COLORS
)

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Simulasi CCTV - Jakarta FloodNet",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="auto"
)

# Inject responsive CSS
inject_responsive_css()

# ==================== CONSTANTS ====================
CCTV_LOCATIONS = [
    {"name": "📍 Pintu Air Manggarai (Pusat)", "location": "Manggarai, Jakarta Selatan"},
    {"name": "🌊 Bendung Katulampa (Hulu)", "location": "Bogor, Jawa Barat"},
    {"name": "🏙️ Pintu Air Karet (Barat)", "location": "Tanah Abang, Jakarta Pusat"},
    {"name": "🌉 Pos Pantau Sunter (Utara)", "location": "Sunter, Jakarta Utara"}
]

VIDEO_FILES = [
    "cctv_manggarai.mp4",
    "cctv_katulampa.mp4", 
    "cctv_karet.mp4",
    "cctv_sunter.mp4"
]

IMAGE_FILES = [
    "cctv_manggarai.png", 
    "cctv_katulampa.png", 
    "cctv_karet.png", 
    "cctv_sunter.png"
]

# Path to videos directory
VIDEOS_DIR = Path(__file__).parent.parent / "assets" / "videos"

# ==================== SESSION STATE ====================
if 'cctv_index' not in st.session_state:
    st.session_state.cctv_index = 0
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = True

# ==================== SIDEBAR ====================
with st.sidebar:
    sidebar_info()
    
    # Connection status
    st.markdown("---")
    st.markdown("#### 🔌 Status Koneksi API")
    health = api_client.check_health()
    connection_status_badge(health['success'])
    
    st.markdown("---")
    st.markdown("#### ⚙️ Kontrol CCTV")
    
    # Detection interval
    detection_interval = st.slider(
        "🔄 Interval Deteksi (detik)", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Seberapa sering mengirim frame ke YOLO untuk deteksi"
    )
    
    # Auto loop toggle
    auto_loop = st.checkbox("🔁 Loop Otomatis", value=True)
    
    # Play/Pause toggle
    if st.button("⏯️ Play/Pause"):
        st.session_state.is_playing = not st.session_state.is_playing
        
    st.markdown("---")
    st.markdown("#### 📊 Status Deteksi")
    
    if st.session_state.detection_result:
        result = st.session_state.detection_result
        if result.get('success'):
            data = result.get('data', {})
            
            # Fix: Parse backend response correctly
            flood_detected = (
                data.get('is_flooded', False) or 
                data.get('flood_detected', False)
            )
            
            # Backend returns 'flood_probability' not 'confidence' 
            confidence = data.get('flood_probability', data.get('confidence', 0))
            
            if flood_detected:
                st.error(f"🚨 **BANJIR TERDETEKSI**\n\nTingkat Keyakinan adanya Banjir: {confidence:.1%}")
            else:
                st.success(f"✅ **KONDISI NORMAL**\n\nTingkat Keyakinan adanya Banjir: {confidence:.1%}")
        else:
            st.warning("⚠️ Gagal deteksi")
    else:
        st.info("⏳ Belum ada deteksi")

# ==================== MAIN CONTENT ====================
page_header(
    "Simulasi CCTV Multi-Channel",
    "Monitoring Real-time dengan Deteksi Banjir Otomatis",
    "cctv"
)

# Check API connection
if not health['success']:
    warning_message("API Gateway tidak terhubung - Deteksi otomatis dinonaktifkan")

st.markdown("---")

# ==================== CHANNEL NAVIGATION ====================
st.markdown("### 📺 Kontrol Channel CCTV")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅️ **Channel Sebelumnya**", use_container_width=True):
        st.session_state.cctv_index = (st.session_state.cctv_index - 1) % len(CCTV_LOCATIONS)

with col2:
    current_location = CCTV_LOCATIONS[st.session_state.cctv_index]
    st.markdown(f"""
        <div style="
            background: {COLORS['info']}15;
            border: 2px solid {COLORS['info']};
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        ">
            <div style="font-size: 1.5rem; font-weight: 700; color: {COLORS['info']};">
                {current_location['name']}
            </div>
            <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">
                📍 {current_location['location']}
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("➡️ **Channel Berikutnya**", use_container_width=True):
        st.session_state.cctv_index = (st.session_state.cctv_index + 1) % len(CCTV_LOCATIONS)

st.markdown("---")

# ==================== VIDEO PLAYER ====================
st.markdown("### 📹 Live Feed CCTV")

# Video placeholder and status
video_placeholder = st.empty()
status_placeholder = st.empty()

# Get current video file
current_video = VIDEO_FILES[st.session_state.cctv_index]
video_path = VIDEOS_DIR / current_video

# Check if video file exists, otherwise create a placeholder
if not video_path.exists():
    with status_placeholder.container():
        st.warning(f"⚠️ **File video tidak ditemukan**: {current_video}")
        st.info("""
            📁 **Cara menambahkan video CCTV:**
            
            1. Siapkan file video (.mp4) untuk simulasi
            2. Simpan di folder: `services/dashboard/src/assets/videos/`
            3. Nama file sesuai dengan: `cctv_manggarai.mp4`, `cctv_katulampa.mp4`, dll
            4. Refresh halaman untuk mulai simulasi
        """)
        
        # Create placeholder image
        with video_placeholder.container():
            st.markdown(f"""
                <div style="
                    background: #f3f4f6;
                    border: 2px dashed #d1d5db;
                    padding: 3rem;
                    border-radius: 12px;
                    text-align: center;
                    min-height: 300px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
                    <div style="font-size: 1.5rem; color: #6b7280;">
                        CCTV Placeholder
                    </div>
                    <div style="font-size: 1rem; color: #9ca3af; margin-top: 0.5rem;">
                        {current_location['name']} - {current_location['location']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

else:
    # Video player logic
    if st.session_state.is_playing:
        try:
            # Display video using native Streamlit player (Much smoother)
            with video_placeholder.container():
                st.video(str(video_path), autoplay=True, loop=True, muted=True)
            
            # One-Shot Detection Logic
            current_time = time.time()
            if (health['success'] and 
                (st.session_state.detection_result is None or 
                 st.session_state.last_detection_time == 0 or
                 current_time - st.session_state.last_detection_time > 2.0)): # Debounce
                
                with status_placeholder.container():
                    with st.spinner("🔍 Menganalisis kondisi visual..."):
                        try:
                            # Get image for current index
                            current_image = IMAGE_FILES[st.session_state.cctv_index]
                            img_path = VIDEOS_DIR.parent / current_image
                            
                            if img_path.exists():
                                with open(img_path, "rb") as f:
                                    frame_bytes = f.read()
                                
                                # Send to YOLO API
                                detection_result = api_client.verify_visual(
                                    frame_bytes, 
                                    f"cctv_analysis_{current_time}.jpg"
                                )
                                st.session_state.detection_result = detection_result
                                st.session_state.last_detection_time = current_time
                                st.rerun()
                            else:
                                st.warning(f"Image asset missing: {current_image}")
                                
                        except Exception as e:
                            st.error(f"Detection Error: {str(e)}")
            
            # Display Status Metrics
            if st.session_state.detection_result:
                result = st.session_state.detection_result
                if result.get('success'):
                    data = result.get('data', {})
                    flood_status = "🚨 BANJIR" if (
                        data.get('is_flooded', False) or 
                        data.get('flood_detected', False)
                    ) else "✅ NORMAL"
                    
                    with status_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📍 Lokasi", current_location['name'])
                        with col2:
                            st.metric("🔍 Status AI", flood_status)

        except Exception as e:
            error_message("Error dalam pemrosesan video", str(e))
    
    else:
        with video_placeholder.container():
            st.markdown(f"""
                <div style="
                    background: #f9fafb;
                    border: 2px solid {COLORS['warning']};
                    padding: 3rem;
                    border-radius: 12px;
                    text-align: center;
                    min-height: 300px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">⏸️</div>
                    <div style="font-size: 1.5rem; color: {COLORS['warning']};">
                        Video Dijeda
                    </div>
                    <div style="font-size: 1rem; color: #6b7280; margin-top: 0.5rem;">
                        Klik Play/Pause di sidebar untuk melanjutkan
                    </div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ==================== DETECTION HISTORY ====================
st.markdown("### 📊 Riwayat Deteksi")

if st.session_state.detection_result:
    result = st.session_state.detection_result
    
    if result.get('success'):
        data = result.get('data', {})
        
        # DEBUG: Show raw response
        with st.expander("🔍 Debug: Raw API Response", expanded=True):
            st.json(data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 Detail Deteksi Terakhir")
            
            detection_time = datetime.fromtimestamp(st.session_state.last_detection_time)
            st.write(f"🕐 **Waktu**: {detection_time.strftime('%H:%M:%S')}")
            st.write(f"📍 **Lokasi**: {current_location['name']}")
            
            # Fix: Parse backend response correctly
            flood_detected = (
                data.get('is_flooded', False) or 
                data.get('flood_detected', False)
            )
            confidence = data.get('flood_probability', data.get('confidence', 0))
            
            if flood_detected:
                st.error(f"🚨 **Status**: BANJIR TERDETEKSI")
                st.error(f"📊 **Confidence of flood**: {confidence:.1%}")
            else:
                st.success(f"✅ **Status**: KONDISI NORMAL")
                st.success(f"📊 **Confidence of flood**: {confidence:.1%}")
        
        with col2:
            st.markdown("#### 🔧 Detail Teknis")
            
            objects_detected = data.get('objects_detected', [])
            bounding_boxes = data.get('bounding_boxes', [])
            
            st.write(f"🎯 **Objek Terdeteksi**: {len(objects_detected)}")
            st.write(f"📐 **Bounding Boxes**: {len(bounding_boxes)}")
            
            if objects_detected:
                st.write("🏷️ **Kategori Objek**:")
                for obj in objects_detected:
                    st.write(f"   • {obj}")
    else:
        st.error(f"❌ **Error Deteksi**: {result.get('message', 'Unknown error')}")
else:
    st.info("ℹ️ Belum ada hasil deteksi. Video akan otomatis mengirim frame ke AI untuk analisis.")

# ==================== FOOTER INFO ====================
st.markdown("---")
st.info("""
    **📋 Informasi Sistem CCTV**
    
    - **🎯 Model Deteksi**: YOLOv8 untuk deteksi objek banjir
    - **📹 Format Video**: MP4 (resolusi optimal: 640x480 atau 1280x720)
    - **⚡ Deteksi Real-time**: Frame dikirim ke API sesuai interval yang ditentukan
    - **🔄 Auto Loop**: Video akan berulang otomatis untuk simulasi kontinyu
    
    **💡 Tips Penggunaan**:
    - Gunakan video dengan durasi 10-30 detik untuk simulasi optimal
    - Interval deteksi 3-5 detik memberikan keseimbangan antara akurasi dan performa
    - Pastikan API Gateway aktif untuk fitur deteksi otomatis
""")