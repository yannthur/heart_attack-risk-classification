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

# ── Load base CSS ──────────────────────────────────────────────────────────────
def load_css(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ── Theme state ────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Dark override — all custom classes + Streamlit internals
DARK = (
    "<style>"
    # App shell
    "html,body,.stApp,.block-container{"
        "background-color:#0E0C0A!important;"
        "color:#F2EDE6!important;}"
    # Sidebar
    "section[data-testid='stSidebar']{"
        "background:#161210!important;"
        "border-right-color:#252018!important;}"
    "section[data-testid='stSidebar'] *{color:#F2EDE6!important;}"
    # Tabs
    ".stTabs [data-baseweb='tab-list']{"
        "background:#1C1814!important;"
        "border-color:#252018!important;}"
    ".stTabs [data-baseweb='tab']{color:#8A7E72!important;}"
    ".stTabs [aria-selected='true']{"
        "background:#E05A4E!important;color:#fff!important;}"
    # Inputs
    ".stSelectbox>div>div{"
        "background:#221E19!important;"
        "border-color:#2E2822!important;"
        "color:#F2EDE6!important;}"
    ".stNumberInput>div>div>input{"
        "background:#221E19!important;"
        "border-color:#2E2822!important;"
        "color:#F2EDE6!important;}"
    "label[data-testid='stWidgetLabel']{color:#8A7E72!important;}"
    # Metrics
    "[data-testid='stMetric']{"
        "background:#1C1814!important;"
        "border-color:#252018!important;}"
    "[data-testid='stMetricValue']{color:#F2EDE6!important;}"
    "[data-testid='stMetricLabel']{color:#5E5448!important;}"
    # Expanders
    "details{background:#1C1814!important;border-color:#252018!important;}"
    "summary{color:#F2EDE6!important;}"
    "hr{border-color:#252018!important;}"
    # Hero
    ".cp-hero{background:#1C1814!important;border-color:#252018!important;}"
    ".cp-htitle{color:#F2EDE6!important;}"
    ".cp-hsub{color:#8A7E72!important;}"
    # Section blocks
    ".cp-block{background:#1C1814!important;border-color:#252018!important;}"
    ".cp-sec{color:#E05A4E!important;border-bottom-color:#E05A4E!important;}"
    # Model cards
    ".cp-mcard{background:#1C1814!important;border-color:#252018!important;}"
    ".cp-mcard-lbl{color:#5E5448!important;}"
    ".cp-mcard-val{color:#F2EDE6!important;}"
    # Reco box
    ".cp-reco{background:#1C1814!important;border-color:#252018!important;}"
    ".cp-reco-hd{color:#5E5448!important;}"
    ".cp-reco-row{color:#8A7E72!important;border-bottom-color:#252018!important;}"
    # Guide
    ".cp-gcard{background:#1C1814!important;border-color:#252018!important;}"
    ".cp-gterm{color:#F2EDE6!important;}"
    ".cp-gdesc{color:#8A7E72!important;}"
    # Sidebar components
    ".cp-sb-lbl{color:#5E5448!important;}"
    ".cp-sb-sec{border-bottom-color:#252018!important;}"
    # Footer
    ".cp-foot{color:#5E5448!important;}"
    "</style>"
)

if st.session_state.dark_mode:
    st.markdown(DARK, unsafe_allow_html=True)


def is_dark():
    return st.session_state.dark_mode


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    pca    = joblib.load("pca.pkl")
    m_pt   = torch.jit.load("torch_model.pth")
    m_pt.eval()
    m_tf   = tf.keras.models.load_model("tensorflow_model.keras")
    return scaler, pca, m_pt, m_tf

try:
    scaler, pca_model, model_torch, model_tensorflow = load_models()
    models_loaded = True
except Exception:
    models_loaded = False


def preprocess(sexe, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
               totChol, sysBP, diaBP, BMI, heartRate, glucose):
    cols = ["male","age","currentSmoker","cigsPerDay","BPMeds","diabetes",
            "totChol","sysBP","diaBP","BMI","heartRate","glucose"]
    df = pd.DataFrame(
        [[sexe,age,currentSmoker,cigsPerDay,BPMeds,diabetes,
          totChol,sysBP,diaBP,BMI,heartRate,glucose]], columns=cols)
    return pca_model.transform(scaler.transform(df))


def pred_tf(pca_data):
    return int(round(model_tensorflow.predict(pca_data, verbose=0)[0][0]))

def pred_pt(pca_data):
    with torch.no_grad():
        out = model_torch(torch.tensor(pca_data, dtype=torch.float32))
    return int(round(out.item()))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Theme toggle
    st.markdown('<div class="cp-sb-sec">', unsafe_allow_html=True)
    ca, cb = st.columns([3, 1])
    with ca:
        lbl = "🌙 Mode Sombre" if is_dark() else "☀️ Mode Clair"
        st.markdown('<p class="cp-sb-lbl">' + lbl + "</p>", unsafe_allow_html=True)
    with cb:
        if st.button("↺", key="toggle_theme", help="Changer le thème"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Brand
    st.markdown('<div class="cp-sb-sec">', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0;">'
        '<div style="font-size:2.8rem;animation:hb 1.8s ease-in-out infinite;display:inline-block;line-height:1;">❤️</div>'
        '<div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;letter-spacing:-0.02em;margin-top:0.5rem;">CardioPredict</div>'
        '<div style="font-size:0.7rem;opacity:0.45;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem;">Intelligence Artificielle</div>'
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Models
    st.markdown('<div class="cp-sb-sec">', unsafe_allow_html=True)
    st.markdown('<p class="cp-sb-lbl">Modèles Actifs</p>', unsafe_allow_html=True)
    if models_loaded:
        st.markdown(
            '<div class="cp-pill-ok">TensorFlow Neural Network</div>'
            '<div class="cp-pill-ok">PyTorch Neural Network</div>',
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Modèles introuvables")
    st.markdown("</div>", unsafe_allow_html=True)

    # About
    st.markdown('<div class="cp-sb-sec">', unsafe_allow_html=True)
    st.markdown('<p class="cp-sb-lbl">Comment ça marche</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;line-height:1.65;opacity:0.65;margin:0;">'
        "Deux réseaux de neurones analysent vos données indépendamment. "
        "Un consensus est ensuite établi pour maximiser la fiabilité."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Privacy
    st.markdown('<div class="cp-sb-sec">', unsafe_allow_html=True)
    st.markdown('<p class="cp-sb-lbl">🔒 Confidentialité</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;line-height:1.6;opacity:0.55;margin:0;">'
        "Aucune donnée n'est stockée. Tout reste local à votre session."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div style="background:rgba(255,179,71,0.1);border:1px solid rgba(255,179,71,0.3);"'
        'border-radius:10px;padding:0.8rem 0.9rem;font-size:0.76rem;line-height:1.55;">'
        "⚕️ <strong>Informatif uniquement.</strong> Consultez un médecin pour tout diagnostic."
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="cp-hero anim-1">'
    '<div class="cp-hico">❤️</div>'
    "<div>"
    '<h1 class="cp-htitle">Cardio<span>Predict</span> AI</h1>'
    '<p class="cp-hsub">Évaluation intelligente du risque cardiovasculaire par réseaux de neurones profonds</p>'
    '<span class="cp-badge">✦ Dual AI &nbsp;·&nbsp; TensorFlow + PyTorch</span>'
    "</div></div>",
    unsafe_allow_html=True
)

if not models_loaded:
    st.error("⚠️ Modèles introuvables — vérifiez scaler.pkl, pca.pkl, torch_model.pth, tensorflow_model.keras")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🩺  Évaluation", "📊  Visualisation", "📖  Guide"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FORMULAIRE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    with st.form("cardio_form"):

        # ── Section 1: Profil ──────────────────────────────────────────────
        st.markdown('<div class="cp-block anim-1">', unsafe_allow_html=True)
        st.markdown('<span class="cp-sec">👤 Profil Patient</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sexe = st.selectbox(
                "Sexe", [0, 1],
                format_func=lambda x: "♀ Femme" if x == 0 else "♂ Homme"
            )
        with c2:
            age = st.number_input("Âge (années)", min_value=1, max_value=120, value=50, step=1)
        with c3:
            BMI = st.number_input("IMC — kg/m²", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        with c4:
            heartRate = st.number_input("Fréquence Cardiaque — bpm", min_value=40, max_value=200, value=70, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Section 2: Mode de vie ─────────────────────────────────────────
        st.markdown('<div class="cp-block anim-2">', unsafe_allow_html=True)
        st.markdown('<span class="cp-sec">🌿 Mode de Vie &amp; Antécédents</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            currentSmoker = st.selectbox(
                "Fumeur actuel", [0, 1],
                format_func=lambda x: "🚭 Non" if x == 0 else "🚬 Oui"
            )
        with c2:
            cigsPerDay = st.number_input("Cigarettes / jour", min_value=0, max_value=100, value=0, step=1)
        with c3:
            BPMeds = st.selectbox(
                "Médicaments tension", [0, 1],
                format_func=lambda x: "Non" if x == 0 else "Oui"
            )
        with c4:
            diabetes = st.selectbox(
                "Diabète", [0, 1],
                format_func=lambda x: "Non" if x == 0 else "Oui"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Section 3: Biologie ────────────────────────────────────────────
        st.markdown('<div class="cp-block anim-3">', unsafe_allow_html=True)
        st.markdown('<span class="cp-sec">🧬 Paramètres Biologiques</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            totChol = st.number_input("Cholestérol total — mg/dL", min_value=100, max_value=600, value=200, step=1)
        with c2:
            sysBP = st.number_input("Pression Systolique — mmHg", min_value=80, max_value=250, value=120, step=1)
        with c3:
            diaBP = st.number_input("Pression Diastolique — mmHg", min_value=40, max_value=150, value=80, step=1)
        with c4:
            glucose = st.number_input("Glycémie — mg/dL", min_value=50, max_value=400, value=100, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "⚡  LANCER L'ANALYSE CARDIAQUE",
            use_container_width=True
        )

    # ── Résultats ───────────────────────────────────────────────────────────
    if submitted:
        if not models_loaded:
            st.error("⚠️ Modèles non chargés.")
        else:
            with st.spinner("🧠 Analyse IA en cours…"):
                try:
                    if currentSmoker == 0:
                        cigsPerDay = 0
                    pca_data = preprocess(
                        sexe, age, currentSmoker, cigsPerDay, BPMeds,
                        diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
                    )
                    r_tf   = pred_tf(pca_data)
                    r_pt   = pred_pt(pca_data)
                    agree  = r_tf == r_pt
                    danger = r_tf == 1 or r_pt == 1

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<span class="cp-sec">🎯 Résultats de l\'Analyse</span>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        lbl = "Risque Élevé" if r_tf == 1 else "Risque Faible"
                        ico = "⚠️" if r_tf == 1 else "✅"
                        st.markdown(
                            '<div class="cp-mcard">'
                            '<div class="cp-mcard-ico">🤖</div>'
                            '<div class="cp-mcard-lbl">TensorFlow</div>'
                            '<div class="cp-mcard-val">' + ico + " " + lbl + "</div>"
                            "</div>", unsafe_allow_html=True
                        )
                    with col2:
                        lbl = "Risque Élevé" if r_pt == 1 else "Risque Faible"
                        ico = "⚠️" if r_pt == 1 else "✅"
                        st.markdown(
                            '<div class="cp-mcard">'
                            '<div class="cp-mcard-ico">🔥</div>'
                            '<div class="cp-mcard-lbl">PyTorch</div>'
                            '<div class="cp-mcard-val">' + ico + " " + lbl + "</div>"
                            "</div>", unsafe_allow_html=True
                        )
                    with col3:
                        ci = "🎯" if agree else "⚡"
                        cl = "✓ Accord" if agree else "△ Divergence"
                        st.markdown(
                            '<div class="cp-mcard">'
                            '<div class="cp-mcard-ico">' + ci + "</div>"
                            '<div class="cp-mcard-lbl">Consensus IA</div>'
                            '<div class="cp-mcard-val">' + cl + "</div>"
                            "</div>", unsafe_allow_html=True
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    if danger:
                        st.markdown(
                            '<div class="cp-high">'
                            '<div class="cp-rtitle">⚠️ Risque Cardiovasculaire Élevé Détecté</div>'
                            '<div class="cp-rsub">Au moins un modèle a identifié des indicateurs de risque</div>'
                            "</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<div class="cp-reco">'
                            '<div class="cp-reco-hd">Recommandations Prioritaires</div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">🏥</span><span>Consultez un cardiologue dans les meilleurs délais</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">💊</span><span>Respectez scrupuleusement vos traitements prescrits</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">🏃</span><span>Adoptez une activité physique régulière et adaptée</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">🥗</span><span>Suivez un régime alimentaire cardiovasculaire équilibré</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">🚭</span><span>Cessez le tabac immédiatement si vous fumez</span></div>'
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="cp-low">'
                            '<div class="cp-rtitle">✅ Risque Cardiovasculaire Faible</div>'
                            '<div class="cp-rsub">Les deux modèles confirment un profil cardiaque favorable</div>'
                            "</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<div class="cp-reco">'
                            '<div class="cp-reco-hd">Conseils Préventifs</div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">🏃</span><span>Continuez 150 min d\'activité physique par semaine</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">🥗</span><span>Maintenez une alimentation riche en fibres et oméga-3</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">📊</span><span>Effectuez des bilans médicaux annuels de prévention</span></div>'
                            '<div class="cp-reco-row"><span class="cp-reco-ico">😌</span><span>Gérez votre stress avec méditation ou relaxation</span></div>'
                            "</div>",
                            unsafe_allow_html=True
                        )

                except Exception as exc:
                    st.error("❌ Erreur de prédiction : " + str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<span class="cp-sec">📊 Votre Profil Cardiovasculaire</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if submitted and models_loaded:
        dark  = is_dark()
        bg_p  = "#1C1814" if dark else "#FFFFFF"
        fcol  = "#F2EDE6" if dark else "#1A1410"
        gcol  = "#252018" if dark else "#E5E0D8"
        acol  = "#E05A4E" if dark else "#C0392B"
        fill  = "rgba(224,90,78,0.18)" if dark else "rgba(192,57,43,0.12)"

        cats  = ["Âge", "IMC", "Cholestérol", "Pression\nSys.", "Glycémie", "Fréq.\nCard."]
        vals  = [
            min(age / 100, 1),
            min(BMI / 40, 1),
            min(totChol / 300, 1),
            min(sysBP / 200, 1),
            min(glucose / 200, 1),
            min(heartRate / 150, 1),
        ]
        refs  = [0.45, 0.62, 0.67, 0.60, 0.50, 0.47]

        # Radar
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=refs + [refs[0]], theta=cats + [cats[0]],
            fill="toself", name="Référence Saine",
            line=dict(color="rgba(100,200,130,0.6)", width=2, dash="dot"),
            fillcolor="rgba(100,200,130,0.07)"
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name="Votre Profil",
            line=dict(color=acol, width=2.5),
            fillcolor=fill
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=bg_p,
                radialaxis=dict(visible=True, range=[0,1],
                                tickfont=dict(size=9, color=fcol),
                                gridcolor=gcol, linecolor=gcol),
                angularaxis=dict(tickfont=dict(size=10, color=fcol, family="Syne"),
                                 gridcolor=gcol, linecolor=gcol)
            ),
            showlegend=True,
            legend=dict(font=dict(color=fcol, size=10), bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Radar — Votre profil vs Référence saine",
                       font=dict(color=fcol, size=13, family="Syne"), x=0.5),
            height=460,
            paper_bgcolor=bg_p, plot_bgcolor=bg_p,
            margin=dict(l=55, r=55, t=55, b=35),
            font=dict(color=fcol)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Âge", str(age) + " ans")
        with m2:
            bmi_d = "✅ Normal" if 18.5 <= BMI <= 24.9 else ("⚠️ Surpoids" if BMI < 30 else "⚠️ Obèse")
            st.metric("IMC", str(round(BMI, 1)), delta=bmi_d)
        with m3:
            st.metric("Cholestérol", str(totChol) + " mg/dL",
                      delta="✅ OK" if totChol < 200 else "⚠️ Élevé")
        with m4:
            st.metric("Sys. BP", str(sysBP) + " mmHg",
                      delta="✅ Normal" if sysBP < 130 else "⚠️ Élevé")
        with m5:
            st.metric("Glycémie", str(glucose) + " mg/dL",
                      delta="✅ Normal" if glucose < 100 else "⚠️ Élevé")

        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart
        bcolors = [acol if v > r else "#3DD68C" for v, r in zip(vals, refs)]
        blabels = ["Âge", "IMC", "Chol.", "Pression", "Glycémie", "Fréq.C."]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=blabels, y=refs, name="Référence",
            marker=dict(color="rgba(140,140,140,0.22)", line=dict(color="rgba(140,140,140,0.35)", width=1)),
            width=0.33
        ))
        fig2.add_trace(go.Bar(
            x=blabels, y=vals, name="Votre Profil",
            marker=dict(color=bcolors, opacity=0.88), width=0.33
        ))
        fig2.update_layout(
            barmode="group",
            paper_bgcolor=bg_p, plot_bgcolor=bg_p,
            font=dict(color=fcol, family="DM Sans"),
            title=dict(text="Comparaison par paramètre (normalisé)",
                       font=dict(color=fcol, size=12, family="Syne"), x=0.5),
            yaxis=dict(range=[0,1.15], gridcolor=gcol, linecolor=gcol),
            xaxis=dict(gridcolor=gcol, linecolor=gcol),
            legend=dict(font=dict(color=fcol), bgcolor="rgba(0,0,0,0)"),
            height=340, margin=dict(l=35, r=35, t=45, b=35)
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        dark = is_dark()
        bg   = "#1C1814" if dark else "#FFFFFF"
        bdr  = "#252018" if dark else "#E5E0D8"
        col  = "#5E5448" if dark else "#9E9285"
        st.markdown(
            '<div style="text-align:center;padding:5rem 2rem;'
            "background:" + bg + ";border:2px dashed " + bdr + ";"
            "border-radius:22px;color:" + col + ";\">"
            '<div style="font-size:3.5rem;margin-bottom:1rem;opacity:0.5;">📊</div>'
            '<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;">'
            "Aucune donnée à afficher</div>"
            '<div style="font-size:0.87rem;opacity:0.65;">'
            "Soumettez le formulaire dans l'onglet <strong>Evaluation</strong> pour voir votre profil.</div>"
            "</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GUIDE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<span class="cp-sec">📖 Guide des Paramètres</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    PARAMS = [
        ("👤", "Sexe", "Hommes et femmes présentent des profils de risque différents selon l'âge et les hormones."),
        ("🎂", "Âge", "Facteur majeur. Le risque augmente après 45 ans chez l'homme, 55 ans chez la femme."),
        ("⚖️", "IMC", "Poids (kg) divisé par taille au carré (m). Normal entre 18.5 et 24.9. Au-delà, risque accru."),
        ("💓", "Fréquence Cardiaque", "Rythme au repos. Valeur saine entre 60 et 100 bpm."),
        ("🚬", "Tabagisme", "Double le risque coronarien. Même quelques cigarettes par jour sont nocives."),
        ("💊", "Antihypertenseurs", "Traitement actif de l'hypertension — facteur de risque important."),
        ("🩸", "Diabète", "L'hyperglycémie chronique endommage les vaisseaux et multiplie le risque."),
        ("🧪", "Cholestérol Total", "Idéalement sous 200 mg/dL. Au-delà de 240 mg/dL, risque d'athérosclérose élevé."),
        ("📈", "Pression Systolique", "Pression lors de la contraction. Normale sous 120 mmHg."),
        ("📉", "Pression Diastolique", "Pression au repos entre battements. Normale sous 80 mmHg."),
        ("🍬", "Glycémie", "Glucose à jeun. Normal 70-99 mg/dL. Pré-diabète 100-125 mg/dL."),
    ]

    for ico, term, desc in PARAMS:
        st.markdown(
            '<div class="cp-gcard">'
            '<div class="cp-gico">' + ico + "</div>"
            "<div>"
            '<div class="cp-gterm">' + term + "</div>"
            '<div class="cp-gdesc">' + desc + "</div>"
            "</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="cp-sec">🤖 Architecture des Modèles</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    dark    = is_dark()
    ai_bg   = "#1C1814" if dark else "#FFFFFF"
    ai_bdr  = "#252018" if dark else "#E5E0D8"
    ai_txt  = "#8A7E72" if dark else "#6B6357"
    ai_acc  = "#E05A4E" if dark else "#C0392B"

    st.markdown(
        '<div style="background:' + ai_bg + ";border:1px solid " + ai_bdr + ";"
        "border-radius:16px;padding:1.8rem;font-size:0.87rem;line-height:1.8;color:" + ai_txt + ";\">"
        '<strong style="font-family:Syne,sans-serif;color:' + ai_acc + ';">① Prétraitement</strong><br>'
        "Les 12 paramètres sont standardisés (<em>StandardScaler</em>) puis réduits par "
        "<em>PCA</em> pour éliminer le bruit et améliorer les performances.<br><br>"
        '<strong style="font-family:Syne,sans-serif;color:' + ai_acc + ';">② Double Inférence</strong><br>'
        "Le vecteur PCA est soumis à un réseau <em>TensorFlow/Keras</em> et un réseau "
        "<em>PyTorch/TorchScript</em>, entraînés indépendamment sur Framingham Heart Study.<br><br>"
        '<strong style="font-family:Syne,sans-serif;color:' + ai_acc + ';">③ Consensus</strong><br>'
        "Un accord entre les deux modèles renforce la fiabilité. En cas de divergence, "
        "une consultation médicale est fortement conseillée."
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    '<div class="cp-foot">'
    "<strong>CardioPredict AI</strong> &nbsp;·&nbsp; © 2026 &nbsp;·&nbsp; "
    "TensorFlow &amp; PyTorch &nbsp;·&nbsp; ❤️ pour la santé cardiovasculaire"
    "</div>",
    unsafe_allow_html=True
)
