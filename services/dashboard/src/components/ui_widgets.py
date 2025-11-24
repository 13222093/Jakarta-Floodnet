"""
UI Widgets dan Komponen Reusable
Jakarta FloodNet Dashboard
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

def status_card(title: str, value: str, status: str = 'info', icon: str = "📊"):
    """
    Menampilkan kartu status dengan warna sesuai tingkat bahaya
    
    Args:
        title: Judul kartu
        value: Nilai yang ditampilkan
        status: Tipe status ('safe', 'warning', 'danger', 'info')
        icon: Emoji icon
    """
    color = COLORS.get(status, COLORS['info'])
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
            border-left: 4px solid {color};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">{icon}</span>
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
    Menampilkan pesan error dengan styling konsisten
    
    Args:
        message: Pesan error utama
        details: Detail tambahan (optional)
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("Detail Error"):
            st.code(details)


def success_message(message: str):
    """Menampilkan pesan sukses"""
    st.success(f"✅ {message}")


def warning_message(message: str):
    """Menampilkan pesan warning"""
    st.warning(f"⚠️ {message}")


def info_message(message: str):
    """Menampilkan pesan info"""
    st.info(f"ℹ️ {message}")


def connection_status_badge(is_connected: bool):
    """
    Badge status koneksi ke API
    
    Args:
        is_connected: Status koneksi (True/False)
    """
    if is_connected:
        st.markdown("""
            <div style="
                display: inline-block;
                background-color: #10b98120;
                color: #10b981;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            ">
                🟢 API Terhubung
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="
                display: inline-block;
                background-color: #ef444420;
                color: #ef4444;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            ">
                🔴 API Terputus
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
        icon = '✅'
    elif water_level < threshold_bahaya:
        status = 'warning'
        status_text = 'SIAGA'
        icon = '⚠️'
    else:
        status = 'danger'
        status_text = 'BAHAYA'
        icon = '🚨'
    
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
                <span style="font-size: 3rem;">{icon}</span>
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
            <div style="text-align: center; margin-top: 1rem; font-size: 1.25rem; font-weight: 600;">
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
    st.sidebar.markdown("""
        ### 🌊 Jakarta FloodNet
        **Sistem Monitoring Banjir AI**
        
        #### Fitur:
        - 🔮 Prediksi LSTM
        - 👁️ Verifikasi Visual YOLO
        - 📊 Real-time Monitoring
        
        ---
        *Powered by AI & ML*
    """)


def page_header(title: str, subtitle: str = "", icon: str = "🌊"):
    """
    Header standar untuk setiap halaman
    
    Args:
        title: Judul halaman
        subtitle: Subtitle (optional)
        icon: Icon emoji
    """
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        ">
            <h1 style="margin: 0; font-size: 2.5rem;">
                {icon} {title}
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
