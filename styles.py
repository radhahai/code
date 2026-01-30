"""
styles.py - CSS Styles and Theme

This module contains all CSS styling for the Streamlit application.
"""


# =============================================================================
# MAIN APPLICATION STYLES
# =============================================================================

APP_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    .stApp {
        background: #0b1220;
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }

    .app-header {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 8px 22px rgba(2, 6, 23, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.6);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    div.stButton > button {
        background: #1d4ed8;
        color: white;
        border: none;
        padding: 0.7rem 1.2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        box-shadow: 0 8px 16px rgba(2, 6, 23, 0.4);
        transition: background 0.2s ease, transform 0.2s ease;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }
    
    div.stButton > button:active {
        transform: translateY(0);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        border: 1px solid transparent;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        background: rgba(99, 102, 241, 0.12);
    }
    
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(99, 102, 241, 0.6) !important;
        box-shadow: none;
    }

    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px !important;
        border-left-width: 4px !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1d4ed8, #6366f1) !important;
    }
    
    .custom-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    
    .dataframe {
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 10px !important;
    }
    
    .disclaimer-box {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15), rgba(234, 179, 8, 0.05));
        border: 1px solid rgba(234, 179, 8, 0.4);
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #fbbf24;
    }
    
    .performance-badge {
        padding: 10px 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        color: white;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.05));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .feature-card h4 {
        color: #f8fafc;
        margin: 10px 0;
    }
    
    .feature-card p {
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
"""


# =============================================================================
# HEADER HTML
# =============================================================================

APP_HEADER = """
<div class="app-header">🏗️ Seismic Analysis Tool</div>
"""

APP_SUBTITLE = """
<div class="app-subtitle">Advanced MDOF Shear Building Simulation for Structural Engineering Education</div>
"""


# =============================================================================
# DISCLAIMER FOR SYNTHETIC DATA
# =============================================================================

SYNTHETIC_DATA_DISCLAIMER = """
<div class="disclaimer-box">
    ⚠️ <strong>Educational Disclaimer:</strong> All ground motions in this tool are 
    <strong>synthetically generated</strong> to match target spectral characteristics. 
    They are NOT actual recorded earthquake data. Use for educational purposes only.
</div>
"""


# =============================================================================
# WELCOME SCREEN CONTENT
# =============================================================================

def get_feature_card(title: str, description: str, color: str) -> str:
    """Generate HTML for a feature card."""
    return f"""
    <div style="background: linear-gradient(135deg, rgba({color}, 0.2), rgba({color}, 0.05));
                border: 1px solid rgba({color}, 0.3); border-radius: 12px; padding: 20px; text-align: center;">
        <h4 style="color: #f8fafc; margin: 10px 0;">{title}</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">{description}</p>
    </div>
    """


WELCOME_FEATURES = [
    ("11 Ground Motion Types", "Synthetic motions matching historical spectral characteristics", "59, 130, 246"),
    ("7 Control Devices", "Base isolation, TMD, viscous dampers & more", "139, 92, 246"),
    ("4 Analysis Types", "Linear, nonlinear, hysteresis & pushover", "34, 197, 94"),
    ("Full Reporting", "Response spectra, fragility & recommendations", "249, 115, 22"),
]
