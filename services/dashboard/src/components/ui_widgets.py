"""
UI Widgets dan Komponen Reusable
Jakarta FloodNet Dashboard

EMOJI TO LUCIDE ICON MAPPING:
==============================
Kategori Status:
- 🌊 (Banjir/Air)     -> 'waves'
- 🌧️ (Hujan)          -> 'cloud-rain'  
- ✅ (Aman/Sukses)     -> 'shield-check'
- ⚠️ (Bahaya/Warning)  -> 'triangle-alert'
- 🚨 (Darurat/Sirine)  -> 'siren'
- ❌ (Error)           -> 'x-circle'
- ℹ️ (Info)           -> 'info'

Kategori UI/Navigasi:
- 🏠 (Home)           -> 'layout-dashboard'
- 🤖 (AI/Model)       -> 'brain-circuit'
- 📷 (Kamera/Visual)  -> 'camera' atau 'scan-eye'  
- 📈 (Grafik/Tren)    -> 'activity'
- 📊 (Dashboard)      -> 'activity' atau 'bar-chart-3'
- 🔮 (Prediksi)       -> 'brain-circuit'
- 👁️ (Mata/Visual)    -> 'scan-eye'
- 🟢 (Connected)      -> 'wifi'
- 🔴 (Disconnected)   -> 'wifi-off'
- ⏱️ (Timer/Waktu)    -> 'timer'

Usage:
render_icon('waves', size=24, color='#3b82f6', margin_right=8)
"""

import streamlit as st
from typing import Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Color scheme konsisten
COLORS = {
    'safe': '#10b981',      # Green - Aman
    'warning': '#f59e0b',   # Yellow - Waspada
    'danger': '#ef4444',    # Red - Bahaya
    'info': '#3b82f6',      # Blue - Info
    'success': '#22c55e',   # Light Green
    'error': '#dc2626'      # Dark Red
}

def render_icon(icon_name: str, size: int = 24, color: Optional[str] = None, 
                margin_right: int = 8) -> str:
    """
    Merender Lucide Icon tanpa library tambahan via CDN.
    
    Args:
        icon_name: Nama icon Lucide (contoh: 'waves', 'triangle-alert')
        size: Ukuran icon dalam pixel (default: 24)
        color: Warna hex (opsional, default mengikuti tema)
        margin_right: Margin kanan dalam pixel (default: 8)
    
    Returns:
        HTML string untuk icon SVG
    
    Note:
        Menggunakan CDN jsdelivr untuk load icon SVG.
        Untuk mengubah warna, gunakan CSS filter.
    """
    url = f"https://cdn.jsdelivr.net/npm/lucide-static@0.344.0/icons/{icon_name}.svg"
    
    # CSS Style untuk adjustment
    style_parts = [
        f"width:{size}px",
        f"height:{size}px", 
        f"margin-right:{margin_right}px",
        "vertical-align:middle"
    ]
    
    # Add color filter if specified
    if color:
        # Convert hex to filter for SVG colorization
        if color.startswith('#'):
            # Basic filter mapping for common colors
            color_filters = {
                '#ffffff': 'invert(1)',  # White
                '#000000': 'invert(0)',  # Black
                '#ef4444': 'invert(17%) sepia(93%) saturate(7471%) hue-rotate(356deg) brightness(91%) contrast(134%)',  # Red
                '#f59e0b': 'invert(64%) sepia(88%) saturate(2231%) hue-rotate(2deg) brightness(105%) contrast(101%)',  # Yellow
                '#10b981': 'invert(64%) sepia(42%) saturate(1352%) hue-rotate(127deg) brightness(95%) contrast(80%)',   # Green
                '#3b82f6': 'invert(45%) sepia(62%) saturate(2865%) hue-rotate(213deg) brightness(103%) contrast(101%)'  # Blue
            }
            filter_value = color_filters.get(color, 'none')
            if filter_value != 'none':
                style_parts.append(f"filter:{filter_value}")
    
    style = "; ".join(style_parts)
    
    # Render HTML Image
    return f'<img src="{url}" style="{style}" alt="{icon_name}-icon">'

def status_card(title: str, value: str, status: str = 'info', icon_name: str = "activity"):
    """
    Menampilkan kartu status dengan warna sesuai tingkat bahaya
    
    Args:
        title: Judul kartu
        value: Nilai yang ditampilkan
        status: Tipe status ('safe', 'warning', 'danger', 'info')
        icon_name: Nama Lucide icon (contoh: 'activity', 'waves', 'triangle-alert')
    """
    color = COLORS.get(status, COLORS['info'])
    icon_html = render_icon(icon_name, size=32, color=color)
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
            border-left: 4px solid {color};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span>{icon_html}</span>
                <div>
                    <div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;">
                        {title}
                    </div>
                    <div style="font-size: 1.875rem; font-weight: 700; color: {color};">
                        {value}
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: Optional[str] = None, 
                delta_color: str = "normal"):
    """
    Kartu metrik dengan delta (perubahan)
    
    Args:
        label: Label metrik
        value: Nilai utama
        delta: Perubahan (optional)
        delta_color: Warna delta ('normal', 'inverse', 'off')
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def error_message(message: str, details: Optional[str] = None):
    """
    Menampilkan pesan error dengan styling konsistent
    
    Args:
        message: Pesan error utama
        details: Detail tambahan (optional)
    """
    error_icon = render_icon('x-circle', size=20, color="#ffffff")
    st.markdown(f"""
        <div style="background: {COLORS['error']}15; border-left: 4px solid {COLORS['error']}; padding: 1rem; border-radius: 4px; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLORS['error']};">
                {error_icon} <strong>{message}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if details:
        with st.expander("Detail Error"):
            st.code(details)


def success_message(message: str):
    """Menampilkan pesan sukses"""
    success_icon = render_icon('shield-check', size=20, color="#ffffff")
    st.markdown(f"""
        <div style="background: {COLORS['success']}15; border-left: 4px solid {COLORS['success']}; padding: 1rem; border-radius: 4px; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLORS['success']};">
                {success_icon} <strong>{message}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)


def warning_message(message: str):
    """Menampilkan pesan warning"""
    warning_icon = render_icon('triangle-alert', size=20, color="#ffffff")
    st.markdown(f"""
        <div style="background: {COLORS['warning']}15; border-left: 4px solid {COLORS['warning']}; padding: 1rem; border-radius: 4px; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLORS['warning']};">
                {warning_icon} <strong>{message}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)


def info_message(message: str):
    """Menampilkan pesan info"""
    info_icon = render_icon('info', size=20, color="#ffffff")
    st.markdown(f"""
        <div style="background: {COLORS['info']}15; border-left: 4px solid {COLORS['info']}; padding: 1rem; border-radius: 4px; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: {COLORS['info']};">
                {info_icon} <strong>{message}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)


def connection_status_badge(is_connected: bool):
    """
    Badge status koneksi ke API
    
    Args:
        is_connected: Status koneksi (True/False)
    """
    if is_connected:
        connected_icon = render_icon('wifi', size=16, color='#10b981')
        st.markdown(f"""
            <div style="
                display: inline-block;
                background-color: #10b98120;
                color: #10b981;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            ">
                {connected_icon} API Terhubung
            </div>
        """, unsafe_allow_html=True)
    else:
        disconnected_icon = render_icon('wifi-off', size=16, color='#ef4444')
        st.markdown(f"""
            <div style="
                display: inline-block;
                background-color: #ef444420;
                color: #ef4444;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            ">
                {disconnected_icon} API Terputus
            </div>
        """, unsafe_allow_html=True)


def flood_level_indicator(water_level: float, threshold_siaga: float = 150.0,
                          threshold_bahaya: float = 200.0):
    """
    Indikator visual level banjir
    
    Args:
        water_level: Ketinggian air dalam cm
        threshold_siaga: Batas siaga (default 150cm)
        threshold_bahaya: Batas bahaya (default 200cm)
    """
    # Tentukan status
    if water_level < threshold_siaga:
        status = 'safe'
        status_text = 'AMAN'
        icon_html = render_icon('shield-check', size=48, color=COLORS['safe'])
    elif water_level < threshold_bahaya:
        status = 'warning'
        status_text = 'SIAGA'
        icon_html = render_icon('triangle-alert', size=48, color=COLORS['warning'])
    else:
        status = 'danger'
        status_text = 'BAHAYA'
        icon_html = render_icon('siren', size=48, color=COLORS['danger'])
    
    color = COLORS[status]
    
    # Progress bar visual
    progress = min(water_level / threshold_bahaya, 1.0)
    
    st.markdown(f"""
        <div style="
            background: white;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="margin-bottom: 0.5rem;">{icon_html}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin-top: 0.5rem;">
                    {status_text}
                </div>
            </div>
            <div style="
                background: #f3f4f6;
                border-radius: 9999px;
                height: 24px;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    height: 100%;
                    width: {progress * 100}%;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="text-align: center; margin-top: 1rem; font-size: 1.25rem; font-weight: 600; color: {color}">
                {water_level:.1f} cm
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_water_level_chart(data: Dict[str, Any]):
    """
    Membuat grafik tren ketinggian air
    
    Args:
        data: Dictionary dengan keys 'timestamps' dan 'water_levels'
    """
    fig = go.Figure()
    
    timestamps = data.get('timestamps', [])
    water_levels = data.get('water_levels', [])
    
    # Line chart utama
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=water_levels,
        mode='lines+markers',
        name='Ketinggian Air',
        line=dict(color=COLORS['info'], width=3),
        marker=dict(size=8)
    ))
    
    # Threshold lines
    fig.add_hline(y=150, line_dash="dash", line_color=COLORS['warning'],
                  annotation_text="Siaga (150 cm)")
    fig.add_hline(y=200, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="Bahaya (200 cm)")
    
    fig.update_layout(
        title="Tren Ketinggian Air",
        xaxis_title="Waktu",
        yaxis_title="Ketinggian (cm)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_rainfall_chart(hujan_bogor: float, hujan_jakarta: float):
    """
    Membuat grafik perbandingan curah hujan
    
    Args:
        hujan_bogor: Curah hujan di Bogor (mm)
        hujan_jakarta: Curah hujan di Jakarta (mm)
    """
    fig = go.Figure(data=[
        go.Bar(
            x=['Bogor', 'Jakarta'],
            y=[hujan_bogor, hujan_jakarta],
            marker_color=[COLORS['info'], COLORS['success']],
            text=[f'{hujan_bogor:.1f} mm', f'{hujan_jakarta:.1f} mm'],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Curah Hujan Saat Ini",
        yaxis_title="Curah Hujan (mm)",
        height=350,
        template='plotly_white'
    )
    
    return fig


def loading_spinner(message: str = "Memproses..."):
    """Context manager untuk loading spinner"""
    return st.spinner(message)


def sidebar_info():
    """Informasi sidebar standar"""
    st.sidebar.markdown("---")
    
    # Generate icons for sidebar
    waves_icon = render_icon('waves', size=20, color='#3b82f6')
    brain_icon = render_icon('brain-circuit', size=16, color='#ffffff')
    eye_icon = render_icon('scan-eye', size=16, color='#ffffff')
    chart_icon = render_icon('activity', size=16, color='#ffffff')
    
    st.sidebar.markdown(f"""
        ### {waves_icon} Jakarta FloodNet
        **Sistem Monitoring Banjir AI**
        
        #### Fitur:
        - {brain_icon} Prediksi LSTM
        - {eye_icon} Verifikasi Visual YOLO
        - {chart_icon} Real-time Monitoring
        
        ---
        *Powered by AI & ML*
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon_name: str = "waves"):
    """
    Header standar untuk setiap halaman
    
    Args:
        title: Judul halaman
        subtitle: Subtitle (optional)
        icon_name: Nama Lucide icon (default: 'waves')
    """
    header_icon = render_icon(icon_name, size=40, color='#ffffff')
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        ">
            <h1 style="margin: 0; font-size: 2.5rem; display: flex; align-items: center; gap: 1rem;">
                {header_icon} {title}
            </h1>
            {f'<p style="margin: 0.5rem 0 0 0; font-size: 1.125rem; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def format_timestamp(timestamp: Optional[str] = None) -> str:
    """
    Format timestamp untuk display
    
    Args:
        timestamp: String timestamp atau None untuk waktu sekarang
    
    Returns:
        Formatted timestamp string
    """
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%d %B %Y, %H:%M:%S")
        except:
            return timestamp
    else:
        return datetime.now().strftime("%d %B %Y, %H:%M:%S")


def inject_responsive_css():
    """
    Inject responsive CSS untuk mobile, tablet, dan desktop
    """
    st.markdown("""
        <style>
        /* Base Responsive Styles */
        @media (max-width: 768px) {
            /* Mobile styles */
            .main .block-container {
                padding: 1rem !important;
                max-width: 100% !important;
            }
            
            /* Smaller font sizes on mobile */
            h1 { font-size: 1.75rem !important; }
            h2 { font-size: 1.5rem !important; }
            h3 { font-size: 1.25rem !important; }
            
            /* Stack columns on mobile */
            .row-widget.stHorizontalBlock {
                flex-direction: column !important;
            }
            
            /* Full width buttons */
            .stButton button {
                width: 100% !important;
                margin-bottom: 0.5rem;
            }
            
            /* Adjust metrics */
            .stMetric {
                padding: 0.75rem !important;
            }
            
            /* Hide sidebar by default on mobile */
            section[data-testid="stSidebar"] {
                width: 0px !important;
            }
        }
        
        @media (min-width: 769px) and (max-width: 1024px) {
            /* Tablet styles */
            .main .block-container {
                padding: 1.5rem !important;
                max-width: 95% !important;
            }
            
            h1 { font-size: 2rem !important; }
            h2 { font-size: 1.75rem !important; }
        }
        
        @media (min-width: 1025px) {
            /* Desktop styles */
            .main .block-container {
                padding: 2rem !important;
                max-width: 1400px !important;
            }
        }
        
        /* Responsive Images */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Responsive Tables */
        .dataframe {
            width: 100% !important;
            overflow-x: auto !important;
        }
        
        /* Responsive Cards */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            gap: 1rem !important;
        }
        
        /* Touch-friendly buttons */
        @media (hover: none) and (pointer: coarse) {
            .stButton button {
                min-height: 3rem !important;
                font-size: 1.1rem !important;
            }
        }
        
        /* Responsive sidebar */
        @media (max-width: 768px) {
            section[data-testid="stSidebar"][aria-expanded="true"] {
                width: 80vw !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
