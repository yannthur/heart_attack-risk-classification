import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import torch
import plotly.graph_objects as go
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --font-display: 'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;

    /* Light palette */
    --bg-primary:     #F7F4EF;
    --bg-secondary:   #FFFFFF;
    --bg-card:        #FFFFFF;
    --bg-input:       #F0EDE8;
    --text-primary:   #1A1410;
    --text-secondary: #6B6459;
    --text-muted:     #9E9286;
    --accent:         #C0392B;
    --accent-light:   #E8564A;
    --accent-glow:    rgba(192,57,43,0.18);
    --success:        #1A7A4A;
    --success-bg:     #EBF7F0;
    --warning:        #B35C00;
    --warning-bg:     #FFF4E6;
    --border:         #E2DDD7;
    --shadow-sm:      0 2px 8px rgba(26,20,16,0.06);
    --shadow-md:      0 8px 32px rgba(26,20,16,0.10);
    --shadow-lg:      0 20px 60px rgba(26,20,16,0.14);
    --radius-sm:      8px;
    --radius-md:      14px;
    --radius-lg:      22px;
    --radius-xl:      32px;
}

[data-theme="dark"], .dark-mode {
    --bg-primary:     #0F0D0B;
    --bg-secondary:   #181411;
    --bg-card:        #1E1A16;
    --bg-input:       #252018;
    --text-primary:   #F5F0EA;
    --text-secondary: #A89D91;
    --text-muted:     #6E6358;
    --accent:         #E8564A;
    --accent-light:   #FF7A6E;
    --accent-glow:    rgba(232,86,74,0.22);
    --success:        #3DD68C;
    --success-bg:     rgba(61,214,140,0.12);
    --warning:        #FFB347;
    --warning-bg:     rgba(255,179,71,0.12);
    --border:         #2A2420;
    --shadow-sm:      0 2px 8px rgba(0,0,0,0.3);
    --shadow-md:      0 8px 32px rgba(0,0,0,0.4);
    --shadow-lg:      0 20px 60px rgba(0,0,0,0.5);
}

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: var(--font-body) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    transition: background-color 0.4s ease, color 0.4s ease;
}

.stApp { max-width: 1440px; margin: 0 auto; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* ── LABELS ── */
label[data-testid="stWidgetLabel"],
.stSelectbox label,
.stNumberInput label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ── BUTTONS ── */
div.stButton > button,
div.stFormSubmitButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #FFFFFF !important;
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 1rem 2rem !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 6px 24px var(--accent-glow) !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: var(--accent-light) !important;
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 12px 36px var(--accent-glow) !important;
}
div.stButton > button:active,
div.stFormSubmitButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: var(--font-display) !important; font-size: 1.7rem !important; font-weight: 700 !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: none !important;
}

/* ── EXPANDERS ── */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 0.75rem !important;
    overflow: hidden !important;
}
summary {
    padding: 1rem 1.5rem !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    cursor: pointer !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; }

/* ── PLOTLY CHARTS ── */
.js-plotly-plot .plotly { border-radius: var(--radius-lg) !important; }

/* ── CUSTOM COMPONENTS ── */

.cp-hero {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
    padding: 2.5rem 3rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}
.cp-hero::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 300px; height: 300px;
    background: radial-gradient(circle at top right, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.cp-hero-icon {
    font-size: 3.5rem;
    line-height: 1;
    filter: drop-shadow(0 4px 12px var(--accent-glow));
    animation: heartbeat 1.8s ease-in-out infinite;
}
@keyframes heartbeat {
    0%,100% { transform: scale(1); }
    14% { transform: scale(1.12); }
    28% { transform: scale(1); }
    42% { transform: scale(1.08); }
    56% { transform: scale(1); }
}
.cp-hero-title {
    font-family: var(--font-display);
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.05;
    margin: 0;
    letter-spacing: -0.02em;
}
.cp-hero-title span { color: var(--accent); }
.cp-hero-sub {
    font-family: var(--font-body);
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0.4rem 0 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}
.cp-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 99px;
    padding: 0.25rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.7rem;
    width: fit-content;
}

.cp-section-title {
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 1.2rem;
    display: inline-block;
}

.cp-form-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.5rem;
    transition: box-shadow 0.3s ease;
}
.cp-form-card:hover { box-shadow: var(--shadow-md); }

.cp-model-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.8rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1);
    position: relative;
    overflow: hidden;
}
.cp-model-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 60%, var(--accent-glow) 100%);
    pointer-events: none;
}
.cp-model-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
.cp-model-card .label {
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.cp-model-card .value {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
}
.cp-model-card .icon { font-size: 2rem; margin-bottom: 0.8rem; }

.cp-result-high {
    background: linear-gradient(135deg, #C0392B 0%, #922B21 100%);
    color: #FFFFFF;
    border-radius: var(--radius-lg);
    padding: 2.5rem 3rem;
    text-align: center;
    box-shadow: 0 16px 48px rgba(192,57,43,0.35);
    animation: emergePulse 2.5s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}
.cp-result-high::after {
    content: '⚠';
    position: absolute;
    right: -20px; top: -20px;
    font-size: 8rem;
    opacity: 0.08;
}
@keyframes emergePulse {
    0%,100% { box-shadow: 0 16px 48px rgba(192,57,43,0.35); }
    50% { box-shadow: 0 20px 64px rgba(192,57,43,0.55); }
}
.cp-result-low {
    background: linear-gradient(135deg, #1A7A4A 0%, #0E5C36 100%);
    color: #FFFFFF;
    border-radius: var(--radius-lg);
    padding: 2.5rem 3rem;
    text-align: center;
    box-shadow: 0 16px 48px rgba(26,122,74,0.3);
    position: relative;
    overflow: hidden;
}
.cp-result-low::after {
    content: '✓';
    position: absolute;
    right: -10px; top: -20px;
    font-size: 8rem;
    opacity: 0.08;
}
.cp-result-title {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.cp-result-sub { font-size: 0.95rem; opacity: 0.85; font-weight: 300; }

.cp-reco-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.cp-reco-title {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
}
.cp-reco-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.92rem;
    color: var(--text-secondary);
}
.cp-reco-item:last-child { border-bottom: none; }
.cp-reco-emoji { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.05rem; }

.cp-sidebar-section {
    margin-bottom: 1.8rem;
    padding-bottom: 1.8rem;
    border-bottom: 1px solid var(--border);
}
.cp-sidebar-section:last-child { border-bottom: none; margin-bottom: 0; }
.cp-sidebar-label {
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}
.cp-model-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.9rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.cp-model-pill.ok {
    background: var(--success-bg);
    color: var(--success);
    border: 1px solid rgba(26,122,74,0.2);
}
.cp-model-pill.ok::before { content: '●'; font-size: 0.6rem; animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.cp-stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-input);
    color: var(--text-secondary);
    border-radius: 99px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

.cp-theme-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-input);
    border-radius: 99px;
    padding: 0.4rem 0.4rem 0.4rem 1rem;
    margin-bottom: 1rem;
}
.cp-theme-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
}

.cp-footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.82rem;
    letter-spacing: 0.02em;
}
.cp-footer strong { color: var(--accent); font-family: var(--font-display); }

.cp-guide-card {
    display: flex;
    gap: 1.2rem;
    align-items: flex-start;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.7rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.cp-guide-card:hover { transform: translateX(4px); box-shadow: var(--shadow-sm); }
.cp-guide-icon { font-size: 1.5rem; flex-shrink: 0; margin-top: 0.1rem; }
.cp-guide-term {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.cp-guide-desc { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; }

/* Animate in sections */
@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
.animate-in { animation: fadeSlideUp 0.5s ease forwards; }
.animate-in-1 { animation-delay: 0.05s; opacity:0; }
.animate-in-2 { animation-delay: 0.12s; opacity:0; }
.animate-in-3 { animation-delay: 0.19s; opacity:0; }

/* Number input arrows styling */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    opacity: 0.5;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── THEME TOGGLE IN SESSION STATE ─────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # Default: dark

# Apply theme class via JS
theme_class = "dark-mode" if st.session_state.dark_mode else ""
st.markdown(f"""
<script>
document.documentElement.setAttribute('data-theme', '{"dark" if st.session_state.dark_mode else "light"}');
</script>
""", unsafe_allow_html=True)

# Override bg for current theme dynamically
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    html, body, .stApp { background-color: #0F0D0B !important; color: #F5F0EA !important; }
    section[data-testid="stSidebar"] { background: #181411 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #1E1A16 !important; border-color: #2A2420 !important; }
    .stTabs [data-baseweb="tab"] { color: #A89D91 !important; }
    details { background: #1E1A16 !important; border-color: #2A2420 !important; }
    [data-testid="stMetric"] { background: #1E1A16 !important; border-color: #2A2420 !important; }
    .stSelectbox > div > div, .stNumberInput > div > div > input { background: #252018 !important; border-color: #2A2420 !important; color: #F5F0EA !important; }
    label[data-testid="stWidgetLabel"] { color: #A89D91 !important; }
    [data-testid="stMetricValue"] { color: #F5F0EA !important; }
    summary { color: #F5F0EA !important; }
    </style>
    """, unsafe_allow_html=True)


# ─── MODEL LOADING ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    pca_model = joblib.load("pca.pkl")
    model_torch = torch.jit.load("torch_model.pth")
    model_torch.eval()
    model_tensorflow = tf.keras.models.load_model("tensorflow_model.keras")
    return scaler, pca_model, model_torch, model_tensorflow

try:
    scaler, pca_model, model_torch, model_tensorflow = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False

def preprocessing_pipeline(sexe, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
                            totChol, sysBP, diaBP, BMI, heartRate, glucose):
    cols = ['male','age','currentSmoker','cigsPerDay','BPMeds','diabetes',
            'totChol','sysBP','diaBP','BMI','heartRate','glucose']
    data = pd.DataFrame([[sexe, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
                          totChol, sysBP, diaBP, BMI, heartRate, glucose]], columns=cols)
    return pca_model.transform(scaler.transform(data))

def tensorflow_prediction(data_pca):
    return int(round(model_tensorflow.predict(data_pca, verbose=0)[0][0]))

def pytorch_prediction(data_pca):
    with torch.no_grad():
        pred = model_torch(torch.tensor(data_pca, dtype=torch.float32))
    return int(round(pred.item()))


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Theme toggle
    st.markdown('<div class="cp-sidebar-section">', unsafe_allow_html=True)
    col_label, col_btn = st.columns([2, 1])
    with col_label:
        st.markdown(f"""
        <div class="cp-theme-label">
            {'🌙 Mode Sombre' if st.session_state.dark_mode else '☀️ Mode Clair'}
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        if st.button("Changer" if st.session_state.dark_mode else "Changer", key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Brand
    st.markdown('<div class="cp-sidebar-section">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
        <div style="font-size:3rem; animation: heartbeat 1.8s ease-in-out infinite; display:inline-block;">❤️</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; letter-spacing:-0.02em; margin-top:0.5rem;">CardioPredict</div>
        <div style="font-size:0.75rem; opacity:0.5; letter-spacing:0.08em; text-transform:uppercase;">Intelligence Artificielle</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Models status
    st.markdown('<div class="cp-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="cp-sidebar-label">Modèles Actifs</div>', unsafe_allow_html=True)
    if models_loaded:
        st.markdown("""
        <div class="cp-model-pill ok">TensorFlow Neural Network</div>
        <div class="cp-model-pill ok">PyTorch Neural Network</div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Modèles non disponibles")
    st.markdown('</div>', unsafe_allow_html=True)

    # Info
    st.markdown('<div class="cp-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="cp-sidebar-label">Comment ça marche</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; line-height:1.65; opacity:0.7;">
    Deux réseaux de neurones profonds analysent vos paramètres médicaux de façon indépendante,
    puis un consensus est établi pour maximiser la fiabilité du diagnostic.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Privacy
    st.markdown('<div class="cp-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="cp-sidebar-label">🔒 Confidentialité</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; line-height:1.6; opacity:0.65;">
    Aucune donnée n'est conservée. Vos informations restent strictement locales à votre session.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Warning
    st.markdown("""
    <div style="background: rgba(255,179,71,0.1); border:1px solid rgba(255,179,71,0.25);
    border-radius:10px; padding:0.9rem 1rem; font-size:0.78rem; line-height:1.55; opacity:0.85;">
    ⚕️ <strong>Usage informatif uniquement.</strong><br>
    Consultez toujours un professionnel de santé qualifié pour tout diagnostic.
    </div>
    """, unsafe_allow_html=True)


# ─── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cp-hero animate-in animate-in-1">
    <div class="cp-hero-icon">❤️</div>
    <div>
        <div class="cp-hero-title">Cardio<span>Predict</span> AI</div>
        <div class="cp-hero-sub">Évaluation intelligente du risque cardiovasculaire par réseaux de neurones profonds</div>
        <div class="cp-badge">✦ Dual AI · TensorFlow + PyTorch</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Modèles IA introuvables — placez scaler.pkl, pca.pkl, torch_model.pth et tensorflow_model.keras dans le répertoire.")


# ─── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🩺  Évaluation", "📊  Visualisation", "📖  Guide"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    with st.form("patient_form"):

        # ── Section 1: Profil ──
        st.markdown('<div class="cp-form-card animate-in animate-in-1">', unsafe_allow_html=True)
        st.markdown('<div class="cp-section-title">👤 Profil Patient</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sexe = st.selectbox("Sexe", options=[0,1], format_func=lambda x: "♀ Femme" if x==0 else "♂ Homme")
        with c2:
            age = st.number_input("Âge (années)", min_value=1, max_value=120, value=50, step=1)
        with c3:
            BMI = st.number_input("IMC — kg/m²", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        with c4:
            heartRate = st.number_input("Fréquence Cardiaque — bpm", min_value=40, max_value=200, value=70, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 2: Mode de vie ──
        st.markdown('<div class="cp-form-card animate-in animate-in-2">', unsafe_allow_html=True)
        st.markdown('<div class="cp-section-title">🌿 Mode de Vie & Antécédents</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            currentSmoker = st.selectbox("Fumeur actuel", options=[0,1], format_func=lambda x: "🚭 Non" if x==0 else "🚬 Oui")
        with c2:
            cigsPerDay = st.number_input("Cigarettes / jour", min_value=0, max_value=100, value=0, step=1)
        with c3:
            BPMeds = st.selectbox("Médicaments tension", options=[0,1], format_func=lambda x: "💊 Non" if x==0 else "💊 Oui")
        with c4:
            diabetes = st.selectbox("Diabète", options=[0,1], format_func=lambda x: "🩸 Non" if x==0 else "🩸 Oui")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 3: Biologie ──
        st.markdown('<div class="cp-form-card animate-in animate-in-3">', unsafe_allow_html=True)
        st.markdown('<div class="cp-section-title">🧬 Paramètres Biologiques</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            totChol = st.number_input("Cholestérol total — mg/dL", min_value=100, max_value=600, value=200, step=1)
        with c2:
            sysBP = st.number_input("Pression Systolique — mmHg", min_value=80, max_value=250, value=120, step=1)
        with c3:
            diaBP = st.number_input("Pression Diastolique — mmHg", min_value=40, max_value=150, value=80, step=1)
        with c4:
            glucose = st.number_input("Glycémie — mg/dL", min_value=50, max_value=400, value=100, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  LANCER L'ANALYSE CARDIAQUE", use_container_width=True)

    # ── RESULTS ──
    if submitted:
        if models_loaded:
            with st.spinner("🧠  Analyse par les deux réseaux de neurones en cours…"):
                try:
                    if currentSmoker == 0:
                        cigsPerDay = 0
                    data_pca = preprocessing_pipeline(
                        sexe, age, currentSmoker, cigsPerDay, BPMeds,
                        diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
                    )
                    tf_pred    = tensorflow_prediction(data_pca)
                    torch_pred = pytorch_prediction(data_pca)
                    consensus  = tf_pred == torch_pred
                    high_risk  = tf_pred == 1 or torch_pred == 1

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="cp-section-title">🎯 Résultats de l\'Analyse</div>', unsafe_allow_html=True)

                    # Model cards
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        res_tf = "Risque Élevé" if tf_pred == 1 else "Risque Faible"
                        icon_tf = "⚠️" if tf_pred == 1 else "✅"
                        st.markdown(f"""
                        <div class="cp-model-card">
                            <div class="icon">🤖</div>
                            <div class="label">TensorFlow</div>
                            <div class="value">{icon_tf} {res_tf}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        res_torch = "Risque Élevé" if torch_pred == 1 else "Risque Faible"
                        icon_torch = "⚠️" if torch_pred == 1 else "✅"
                        st.markdown(f"""
                        <div class="cp-model-card">
                            <div class="icon">🔥</div>
                            <div class="label">PyTorch</div>
                            <div class="value">{icon_torch} {res_torch}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="cp-model-card">
                            <div class="icon">{"🎯" if consensus else "⚡"}</div>
                            <div class="label">Consensus IA</div>
                            <div class="value">{"✓ Accord" if consensus else "△ Divergence"}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Main result banner
                    if high_risk:
                        st.markdown("""
                        <div class="cp-result-high">
                            <div class="cp-result-title">⚠️ Profil à Risque Cardiovasculaire Élevé</div>
                            <div class="cp-result-sub">Au moins un modèle IA a détecté des indicateurs préoccupants</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="cp-reco-box">
                            <div class="cp-reco-title">Recommandations Prioritaires</div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">🏥</span><span>Consultez un cardiologue dans les meilleurs délais</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">💊</span><span>Respectez scrupuleusement vos traitements prescrits</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">🏃</span><span>Adoptez une activité physique régulière et adaptée</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">🥗</span><span>Suivez un régime alimentaire cardiovasculaire équilibré</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">🚭</span><span>Cessez le tabac immédiatement si vous fumez</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="cp-result-low">
                            <div class="cp-result-title">✅ Profil à Risque Cardiovasculaire Faible</div>
                            <div class="cp-result-sub">Les deux modèles IA confirment un profil cardiaque favorable</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="cp-reco-box">
                            <div class="cp-reco-title">Conseils pour Maintenir votre Santé</div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">🏃</span><span>Continuez une activité physique régulière (150 min/semaine)</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">🥗</span><span>Maintenez une alimentation saine, riche en fibres et oméga-3</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">📊</span><span>Effectuez des bilans médicaux annuels de prévention</span></div>
                            <div class="cp-reco-item"><span class="cp-reco-emoji">😌</span><span>Gérez votre stress par la méditation ou le yoga</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction : {e}")
        else:
            st.error("⚠️ Les modèles ne sont pas chargés. Vérifiez les fichiers .pkl et .keras.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="cp-section-title">📊 Visualisation de vos Paramètres</div>', unsafe_allow_html=True)

    if 'age' in locals() and submitted:
        dark = st.session_state.dark_mode
        bg_color   = "#0F0D0B" if dark else "#F7F4EF"
        paper_bg   = "#1E1A16" if dark else "#FFFFFF"
        font_color = "#F5F0EA" if dark else "#1A1410"
        grid_color = "#2A2420" if dark else "#E2DDD7"
        accent_col = "#E8564A" if dark else "#C0392B"

        categories = ['Âge', 'IMC', 'Cholestérol', 'Pression\nSystolique', 'Glycémie', 'Fréq.\nCardiaque']
        values = [
            min(age / 100, 1),
            min(BMI / 40, 1),
            min(totChol / 300, 1),
            min(sysBP / 200, 1),
            min(glucose / 200, 1),
            min(heartRate / 150, 1)
        ]

        # Reference (healthy) values
        ref_values = [0.45, 0.62, 0.67, 0.6, 0.5, 0.47]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=ref_values + [ref_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Référence Saine',
            line=dict(color='rgba(100,200,130,0.6)', width=2, dash='dot'),
            fillcolor='rgba(100,200,130,0.08)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Votre Profil',
            line=dict(color=accent_col, width=2.5),
            fillcolor=f'rgba({232 if dark else 192},{86 if dark else 57},{74 if dark else 43},0.2)'
        ))

        fig.update_layout(
            polar=dict(
                bgcolor=paper_bg,
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=10, color=font_color), gridcolor=grid_color, linecolor=grid_color),
                angularaxis=dict(tickfont=dict(size=11, color=font_color, family="Syne"), gridcolor=grid_color, linecolor=grid_color)
            ),
            showlegend=True,
            legend=dict(font=dict(color=font_color, size=11), bgcolor='rgba(0,0,0,0)'),
            title=dict(text="Radar Cardiovasculaire — Votre profil vs Référence", font=dict(color=font_color, size=14, family="Syne"), x=0.5),
            height=480,
            paper_bgcolor=paper_bg,
            plot_bgcolor=paper_bg,
            margin=dict(l=60, r=60, t=60, b=40),
            font=dict(color=font_color)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Metrics row ──
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Âge", f"{age} ans")
        with c2:
            imc_label = "✅ Normal" if 18.5<=BMI<=24.9 else ("⚠️ Surpoids" if BMI<30 else "⚠️ Obèse")
            st.metric("IMC", f"{BMI:.1f}", delta=imc_label)
        with c3:
            chol_label = "✅ OK" if totChol < 200 else "⚠️ Élevé"
            st.metric("Cholestérol", f"{totChol} mg/dL", delta=chol_label)
        with c4:
            bp_label = "✅ Normal" if sysBP < 130 else "⚠️ Élevé"
            st.metric("Sys. BP", f"{sysBP} mmHg", delta=bp_label)
        with c5:
            gluc_label = "✅ Normal" if glucose < 100 else "⚠️ Élevé"
            st.metric("Glycémie", f"{glucose} mg/dL", delta=gluc_label)

        # ── Bar chart ──
        st.markdown("<br>", unsafe_allow_html=True)
        bar_labels = ['Âge\n(norm.)', 'IMC\n(norm.)', 'Cholestérol\n(norm.)', 'Pression\n(norm.)', 'Glycémie\n(norm.)', 'Fréq.Card.\n(norm.)']
        bar_colors = [accent_col if v > r else '#3DD68C' for v, r in zip(values, ref_values)]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=bar_labels, y=ref_values,
            name='Référence',
            marker=dict(color='rgba(150,150,150,0.25)', line=dict(color='rgba(150,150,150,0.4)', width=1)),
            width=0.35
        ))
        fig2.add_trace(go.Bar(
            x=bar_labels, y=values,
            name='Votre Profil',
            marker=dict(color=bar_colors, opacity=0.85),
            width=0.35
        ))
        fig2.update_layout(
            barmode='group',
            paper_bgcolor=paper_bg, plot_bgcolor=paper_bg,
            font=dict(color=font_color, family="DM Sans"),
            title=dict(text="Comparaison par Paramètre", font=dict(color=font_color, size=13, family="Syne"), x=0.5),
            yaxis=dict(range=[0,1.1], gridcolor=grid_color, linecolor=grid_color),
            xaxis=dict(gridcolor=grid_color, linecolor=grid_color),
            legend=dict(font=dict(color=font_color), bgcolor='rgba(0,0,0,0)'),
            height=360,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        # Placeholder state
        dark = st.session_state.dark_mode
        st.markdown(f"""
        <div style="
            text-align:center;
            padding: 5rem 2rem;
            background: {'#1E1A16' if dark else '#FFFFFF'};
            border: 2px dashed {'#2A2420' if dark else '#E2DDD7'};
            border-radius: 22px;
            color: {'#6E6358' if dark else '#9E9286'};
        ">
            <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; margin-bottom:0.5rem;">
                Aucune donnée à visualiser
            </div>
            <div style="font-size:0.9rem; opacity:0.7;">
                Remplissez le formulaire dans l'onglet <strong>Évaluation</strong> et lancez l'analyse pour voir votre profil cardiovasculaire.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — GUIDE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="cp-section-title">📖 Guide des Paramètres</div>', unsafe_allow_html=True)

    params = [
        ("👤", "Sexe", "Hommes et femmes présentent des profils de risque cardiovasculaire différents selon l'âge et les hormones."),
        ("🎂", "Âge", "Facteur de risque majeur — le risque augmente significativement après 45 ans chez l'homme et 55 ans chez la femme."),
        ("⚖️", "IMC (Indice de Masse Corporelle)", "Calculé par poids (kg) / taille² (m²). Normal : 18.5–24.9. Au-delà, le risque cardiaque s'accroît."),
        ("💓", "Fréquence Cardiaque", "Rythme cardiaque au repos. Une valeur saine se situe entre 60 et 100 bpm."),
        ("🚬", "Tabagisme", "Le tabac double le risque de maladie coronarienne. Même quelques cigarettes/jour sont nocives."),
        ("💊", "Médicaments Antihypertenseurs", "Indique un traitement de l'hypertension artérielle — facteur de risque important."),
        ("🩸", "Diabète", "L'hyperglycémie chronique endommage les vaisseaux sanguins et multiplie le risque cardiovasculaire."),
        ("🧪", "Cholestérol Total", "Idéalement < 200 mg/dL. Au-delà de 240 mg/dL, le risque d'athérosclérose est élevé."),
        ("📈", "Pression Systolique", "La pression lors de la contraction cardiaque. Valeur normale : < 120 mmHg."),
        ("📉", "Pression Diastolique", "La pression au repos entre deux battements. Valeur normale : < 80 mmHg."),
        ("🍬", "Glycémie", "Taux de glucose dans le sang à jeun. Normal : 70–99 mg/dL. Pré-diabète : 100–125 mg/dL."),
    ]

    for icon, term, desc in params:
        st.markdown(f"""
        <div class="cp-guide-card">
            <div class="cp-guide-icon">{icon}</div>
            <div>
                <div class="cp-guide-term">{term}</div>
                <div class="cp-guide-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cp-section-title">🤖 Architecture IA</div>', unsafe_allow_html=True)

    dark = st.session_state.dark_mode
    st.markdown(f"""
    <div style="background:{'#1E1A16' if dark else '#FFFFFF'}; border:1px solid {'#2A2420' if dark else '#E2DDD7'};
         border-radius:16px; padding:2rem; font-size:0.88rem; line-height:1.8; color:{'#A89D91' if dark else '#6B6459'};">
        <strong style="color:{'#E8564A' if dark else '#C0392B'}; font-family:'Syne',sans-serif;">Étape 1 — Prétraitement</strong><br>
        Les 12 paramètres cliniques sont standardisés via un <em>StandardScaler</em>, puis compressés par 
        <em>Analyse en Composantes Principales (PCA)</em> pour réduire le bruit et améliorer les performances.<br><br>
        <strong style="color:{'#E8564A' if dark else '#C0392B'}; font-family:'Syne',sans-serif;">Étape 2 — Double Inférence</strong><br>
        Le vecteur PCA est soumis simultanément à un réseau <em>TensorFlow (Keras)</em> et un réseau 
        <em>PyTorch (TorchScript)</em>, entraînés indépendamment sur les données Framingham Heart Study.<br><br>
        <strong style="color:{'#E8564A' if dark else '#C0392B'}; font-family:'Syne',sans-serif;">Étape 3 — Consensus</strong><br>
        Un accord entre les deux modèles renforce la fiabilité du résultat. En cas de divergence, 
        une prudence accrue est recommandée et une consultation médicale est conseillée.
    </div>
    """, unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="cp-footer">
    <strong>CardioPredict AI</strong> &nbsp;·&nbsp; © 2026 &nbsp;·&nbsp; 
    Propulsé par TensorFlow & PyTorch &nbsp;·&nbsp;
    Développé avec ❤️ pour la santé cardiovasculaire
</div>
""", unsafe_allow_html=True)
