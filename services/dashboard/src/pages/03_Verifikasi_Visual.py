"""
Halaman Verifikasi Visual (YOLO)
=================================
Analisis gambar untuk deteksi kondisi banjir menggunakan YOLO
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import io

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
    info_message,
    loading_spinner,
    sidebar_info,
    inject_responsive_css,
    COLORS
)

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Verifikasi Visual - Jakarta FloodNet",
    page_icon="👁️",
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
    
    if not health['success']:
        st.error("⚠️ API offline - verifikasi tidak dapat dilakukan")

# ==================== MAIN CONTENT ====================
page_header(
    "Verifikasi Visual (YOLO)",
    "Analisis Gambar Kondisi Banjir Menggunakan Computer Vision",
    "👁️"
)

# Check API connection
if not health['success']:
    error_message(
        "API Gateway tidak terhubung",
        "Pastikan backend API sudah berjalan di http://localhost:8000"
    )
    st.stop()

# ==================== INSTRUCTION ====================
with st.expander("📖 Cara Menggunakan", expanded=False):
    st.markdown("""
        ### Panduan Penggunaan Verifikasi Visual
        
        1. **Upload Gambar**
           - Klik tombol "Browse files" atau drag & drop
           - Format: JPG, PNG, JPEG
           - Ukuran: Maksimal 10MB
           - Resolusi: Optimal 640x640 atau lebih
        
        2. **Klik "Analisis Gambar"**
           - Sistem akan mengirim gambar ke API Gateway
           - YOLO model akan mendeteksi objek terkait banjir
           - Hasil deteksi akan ditampilkan
        
        3. **Interpretasi Hasil**
           - **Flood Indicators**: Jumlah objek terkait banjir
           - **Flood Probability**: Kemungkinan banjir (0-100%)
           - **Water Coverage**: Persentase area tergenang
           - **Severity Level**: Tingkat keparahan banjir
        
        ### Tips untuk Hasil Optimal
        - Gunakan gambar yang jelas dan terang
        - Hindari gambar yang blur atau gelap
        - Foto area yang menunjukkan jalan, sungai, atau bangunan
        - Capture kondisi real-time untuk akurasi tinggi
        
        ### Catatan
        - Model: YOLOv8n untuk deteksi objek
        - Deteksi: Water, vehicles, buildings, flood signs
        - Processing time: ~1-3 detik per gambar
    """)

st.markdown("---")

# ==================== IMAGE UPLOAD ====================
st.markdown("### 📤 Upload Gambar")

# Responsive layout for upload section
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Pilih gambar kondisi lapangan",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar jalan, sungai, atau area yang ingin dianalisis",
        accept_multiple_files=False
    )

with col2:
    st.markdown("""
        **📋 Syarat:**
        - JPG, PNG
        - Max 10MB
        - Optimal 640x640px
        - Terang & jelas
    """)

# Display uploaded image
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        st.markdown("---")
        st.markdown("#### 🖼️ Preview Gambar")
        
        # Responsive image preview (stacked on mobile, side-by-side on desktop)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption="Gambar yang diupload", use_container_width=True)
        
        with col2:
            st.markdown("**📊 Info:**")
            st.caption(f"📏 {image.size[0]} x {image.size[1]} px")
            st.caption(f"📁 {image.format}")
            st.caption(f"🎨 {image.mode}")
            
            # File size
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)  # Reset to start
            st.caption(f"💾 {file_size / 1024:.1f} KB")
        
        st.markdown("---")
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            analyze_button = st.button(
                "👁️ **Analisis Gambar**",
                use_container_width=True,
                type="primary",
                help="Klik untuk menganalisis gambar dengan YOLO model"
            )
        
        # ==================== ANALYSIS RESULTS ====================
        if analyze_button:
            with loading_spinner("👁️ Menganalisis gambar..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Call API
                result = api_client.verify_image(uploaded_file)
            
            st.markdown("---")
            st.markdown("### 🎯 Hasil Analisis Visual")
            
            if result['success']:
                data = result['data']
                
                # Extract analysis info
                flood_indicators = data.get('flood_indicators', 0)
                total_detections = data.get('total_detections', 0)
                flood_probability = data.get('flood_probability', 0) * 100  # Convert to percentage
                severity_level = data.get('severity_level', 'unknown')
                water_coverage = data.get('water_coverage', 0)
                detections = data.get('detections', [])
                
                # Display key metrics (responsive: 2x2 grid on mobile, 1x4 on desktop)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div style="
                            background: {COLORS['info']}15;
                            border: 3px solid {COLORS['info']};
                            padding: 1.25rem;
                            border-radius: 12px;
                            text-align: center;
                        ">
                            <div style="font-size: 2rem;">🔍</div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
                                Total Deteksi
                            </div>
                            <div style="font-size: 1.75rem; font-weight: 800; color: {COLORS['info']}; margin-top: 0.5rem;">
                                {total_detections}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    indicator_color = COLORS['danger'] if flood_indicators > 5 else (COLORS['warning'] if flood_indicators > 2 else COLORS['success'])
                    
                    st.markdown(f"""
                        <div style="
                            background: {indicator_color}15;
                            border: 3px solid {indicator_color};
                            padding: 1.25rem;
                            border-radius: 12px;
                            text-align: center;
                        ">
                            <div style="font-size: 2rem;">🚨</div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
                                Flood Indicators
                            </div>
                            <div style="font-size: 1.75rem; font-weight: 800; color: {indicator_color}; margin-top: 0.5rem;">
                                {flood_indicators}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    prob_color = COLORS['danger'] if flood_probability > 70 else (COLORS['warning'] if flood_probability > 40 else COLORS['safe'])
                    
                    st.markdown(f"""
                        <div style="
                            background: {prob_color}15;
                            border: 3px solid {prob_color};
                            padding: 1.25rem;
                            border-radius: 12px;
                            text-align: center;
                        ">
                            <div style="font-size: 2rem;">📊</div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
                                Flood Probability
                            </div>
                            <div style="font-size: 1.75rem; font-weight: 800; color: {prob_color}; margin-top: 0.5rem;">
                                {flood_probability:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Determine severity color
                    severity_colors = {
                        'no_flood': COLORS['safe'],
                        'minor_flood': COLORS['warning'],
                        'moderate_flood': COLORS['warning'],
                        'major_flood': COLORS['danger'],
                        'severe_flood': COLORS['danger']
                    }
                    severity_color = severity_colors.get(severity_level, COLORS['info'])
                    
                    st.markdown(f"""
                        <div style="
                            background: {severity_color}15;
                            border: 3px solid {severity_color};
                            padding: 1.25rem;
                            border-radius: 12px;
                            text-align: center;
                        ">
                            <div style="font-size: 2rem;">⚡</div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
                                Severity
                            </div>
                            <div style="font-size: 1.125rem; font-weight: 800; color: {severity_color}; margin-top: 0.5rem;">
                                {severity_level.replace('_', ' ').title()}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Analysis summary
                st.markdown("#### 📋 Kesimpulan Analisis")
                
                if flood_probability < 30:
                    success_message("✅ **TIDAK TERDETEKSI BANJIR** - Kondisi area tampak normal tanpa tanda-tanda banjir signifikan")
                elif flood_probability < 70:
                    warning_message("⚠️ **POTENSI BANJIR TERDETEKSI** - Terdapat indikasi genangan atau kondisi yang perlu diwaspadai")
                else:
                    st.error("🚨 **BANJIR TERDETEKSI** - Kondisi banjir teridentifikasi dengan confidence tinggi. Evakuasi mungkin diperlukan!")
                
                # Water coverage info
                if water_coverage > 0:
                    st.info(f"💧 **Water Coverage**: {water_coverage:.1f}% dari area gambar terdeteksi sebagai air atau genangan")
                
                # Detailed detections
                if detections and len(detections) > 0:
                    st.markdown("---")
                    st.markdown("#### 🔍 Detail Deteksi Objek")
                    
                    # Create table of detections
                    detection_data = []
                    for i, det in enumerate(detections, 1):
                        detection_data.append({
                            "No": i,
                            "Objek": det.get('class_name', 'Unknown'),
                            "Confidence": f"{det.get('confidence', 0) * 100:.1f}%",
                            "Area": f"{det.get('area', 0):.0f} px²"
                        })
                    
                    st.table(detection_data)
                
                # Full API response
                with st.expander("📊 Data Lengkap (JSON)"):
                    st.json(data)
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 **Analisis Gambar Lain**", use_container_width=True):
                        st.rerun()
                
                with col2:
                    if st.button("🔮 **Buat Prediksi LSTM**", use_container_width=True):
                        st.switch_page("pages/02_Prediksi_Banjir.py")
                
                with col3:
                    if st.button("📊 **Kembali ke Dashboard**", use_container_width=True):
                        st.switch_page("pages/01_Dashboard_Utama.py")
            
            else:
                error_message(
                    "Gagal menganalisis gambar",
                    result.get('message', 'Unknown error')
                )
                
                st.markdown("#### 🔧 Troubleshooting")
                st.markdown("""
                    - Pastikan API Gateway berjalan
                    - Cek format dan ukuran gambar
                    - Verifikasi model YOLO sudah ter-load
                    - Coba upload ulang gambar
                """)
    
    except Exception as e:
        error_message("Error membaca gambar", str(e))

else:
    # No image uploaded yet
    st.markdown("---")
    info_message("📤 Silakan upload gambar terlebih dahulu untuk memulai analisis")
    
    # Sample images info
    st.markdown("#### 📷 Contoh Gambar yang Baik")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            **✅ Gambar Jalan**
            - Tampilan jelas jalan raya
            - Visible vehicles & traffic
            - Kondisi pencahayaan baik
        """)
    
    with col2:
        st.markdown("""
            **✅ Gambar Sungai**
            - Area sungai terlihat jelas
            - Water level visible
            - Landmark untuk referensi
        """)
    
    with col3:
        st.markdown("""
            **✅ Area Pemukiman**
            - Bangunan dan infrastruktur
            - Street conditions
            - Drainage systems
        """)

# ==================== FOOTER INFO ====================
st.markdown("---")
st.info("""
    **ℹ️ Informasi Model**
    
    - **Model**: YOLOv8n (Nano) - Fast & Lightweight
    - **Deteksi**: Water, vehicles, buildings, infrastructure, flood indicators
    - **Processing**: ~1-3 seconds per image
    - **Accuracy**: Real-time object detection with high precision
    - **Integration**: Seamless verification with LSTM predictions
""")
