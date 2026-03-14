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

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── load CSS ──────────────────────────────────────────────────────────────────
try:
    with open("style.css", "r", encoding="utf-8") as _f:
        st.markdown("<style>" + _f.read() + "</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── theme ─────────────────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = True

_DK = (
    "<style>"
    "html,body,.stApp,.block-container"
        "{background:#0E0C0A!important;color:#F2EDE6!important}"
    "section[data-testid='stSidebar']"
        "{background:#161210!important;border-right-color:#252018!important}"
    "section[data-testid='stSidebar'] *{color:#F2EDE6!important}"
    ".stTabs [data-baseweb='tab-list']"
        "{background:#1C1814!important;border-color:#252018!important}"
    ".stTabs [data-baseweb='tab']{color:#8A7E72!important}"
    ".stTabs [aria-selected='true']{background:#E05A4E!important;color:#fff!important}"
    ".stSelectbox>div>div"
        "{background:#1E1A15!important;border-color:#2A2420!important;color:#F2EDE6!important}"
    ".stNumberInput>div>div>input"
        "{background:#1E1A15!important;border-color:#2A2420!important;color:#F2EDE6!important}"
    "label[data-testid='stWidgetLabel']{color:#7A6E62!important}"
    "[data-testid='stMetric']"
        "{background:#1C1814!important;border-color:#252018!important}"
    "[data-testid='stMetricValue']{color:#F2EDE6!important}"
    "[data-testid='stMetricLabel']{color:#5A4E42!important}"
    "details{background:#1C1814!important;border-color:#252018!important}"
    "summary{color:#F2EDE6!important}"
    "hr{border-color:#252018!important}"
    ".cp-hero{background:#1C1814!important;border-color:#252018!important}"
    ".cp-htitle{color:#F2EDE6!important}"
    ".cp-hsub{color:#8A7E72!important}"
    ".cp-secrow{background:transparent!important}"
    ".cp-secline{background:#252018!important}"
    ".cp-mcard{background:#1C1814!important;border-color:#252018!important}"
    ".cp-mlbl{color:#5A4E42!important}"
    ".cp-mval{color:#F2EDE6!important}"
    ".cp-reco{background:#1C1814!important;border-color:#252018!important}"
    ".cp-reco-hd{color:#5A4E42!important}"
    ".cp-rrow{color:#8A7E72!important;border-bottom-color:#252018!important}"
    ".cp-gc{background:#1C1814!important;border-color:#252018!important}"
    ".cp-gt{color:#F2EDE6!important}"
    ".cp-gd{color:#8A7E72!important}"
    ".cp-foot{color:#5A4E42!important}"
    "</style>"
)

if st.session_state.dark:
    st.markdown(_DK, unsafe_allow_html=True)

def dk():
    return st.session_state.dark

# ── models ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    sc  = joblib.load("scaler.pkl")
    pc  = joblib.load("pca.pkl")
    mpt = torch.jit.load("torch_model.pth")
    mpt.eval()
    mtf = tf.keras.models.load_model("tensorflow_model.keras")
    return sc, pc, mpt, mtf

try:
    scaler, pca_model, model_pt, model_tf = load_models()
    ok = True
except Exception:
    ok = False

def run_preprocess(sx, ag, cs, cp, bpm, db, tc, sbp, dbp, bmi, hr, gl):
    cols = ["male","age","currentSmoker","cigsPerDay","BPMeds","diabetes",
            "totChol","sysBP","diaBP","BMI","heartRate","glucose"]
    df = pd.DataFrame([[sx,ag,cs,cp,bpm,db,tc,sbp,dbp,bmi,hr,gl]], columns=cols)
    return pca_model.transform(scaler.transform(df))

def run_tf(d):
    return int(round(model_tf.predict(d, verbose=0)[0][0]))

def run_pt(d):
    with torch.no_grad():
        return int(round(model_pt(torch.tensor(d, dtype=torch.float32)).item()))

# ── helper: section header (self-contained, no open divs) ────────────────────
def sec(icon, label):
    st.markdown(
        '<div class="cp-secrow">'
        '<span class="cp-sectag">' + icon + " " + label + "</span>"
        '<div class="cp-secline"></div>'
        "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    ca, cb = st.columns([3,1])
    with ca:
        lbl = "🌙 Mode Sombre" if dk() else "☀️ Mode Clair"
        st.markdown('<p class="cp-sblbl">' + lbl + "</p>", unsafe_allow_html=True)
    with cb:
        if st.button("↺", key="tg", help="Changer le thème"):
            st.session_state.dark = not st.session_state.dark
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:.4rem 0 .8rem;">'
        '<div style="font-size:2.6rem;animation:hb 1.8s ease-in-out infinite;display:inline-block;">❤️</div>'
        '<div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;margin-top:.4rem;">CardioPredict</div>'
        '<div style="font-size:.68rem;opacity:.4;letter-spacing:.1em;text-transform:uppercase;">Intelligence Artificielle</div>'
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<p class="cp-sblbl">Modèles Actifs</p>', unsafe_allow_html=True)
    if ok:
        st.markdown(
            '<div class="cp-pill">TensorFlow Neural Network</div>'
            '<div class="cp-pill">PyTorch Neural Network</div>',
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Modèles introuvables")

    st.markdown("---")
    st.markdown('<p class="cp-sblbl">Comment ça marche</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.81rem;line-height:1.65;opacity:.6;margin:0;">'
        "Deux réseaux de neurones analysent vos données indépendamment. "
        "Un consensus est établi pour maximiser la fiabilité du résultat."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown('<p class="cp-sblbl">🔒 Confidentialité</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.79rem;line-height:1.6;opacity:.5;margin:0;">'
        "Aucune donnée n'est stockée. Tout reste local à votre session."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.warning("⚕️ **Informatif uniquement.** Consultez un médecin pour tout diagnostic médical.")


# ══════════════════════════════════════════════════════════════════════════════
# HERO  (self-contained HTML — no widgets inside)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="cp-hero">'
    '<div class="cp-hico">❤️</div>'
    "<div>"
    '<h1 class="cp-htitle">Cardio<span>Predict</span> AI</h1>'
    '<p class="cp-hsub">Évaluation intelligente du risque cardiovasculaire par réseaux de neurones profonds</p>'
    '<span class="cp-badge">✦ Dual AI &nbsp;·&nbsp; TensorFlow + PyTorch</span>'
    "</div></div>",
    unsafe_allow_html=True
)

if not ok:
    st.error("⚠️ Modèles introuvables — vérifiez scaler.pkl · pca.pkl · torch_model.pth · tensorflow_model.keras")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🩺  Évaluation", "📊  Visualisation", "📖  Guide"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FORM
# Key rule: NO open <div> before a Streamlit widget. Every st.markdown() is
# a *complete*, self-closing HTML snippet.
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    with st.form("cf"):

        # section header — self-contained
        sec("👤", "Profil Patient")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sexe = st.selectbox("Sexe", [0,1], format_func=lambda x:"♀ Femme" if x==0 else "♂ Homme")
        with c2:
            age = st.number_input("Âge (années)", min_value=1, max_value=120, value=50, step=1)
        with c3:
            BMI = st.number_input("IMC — kg/m²", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        with c4:
            heartRate = st.number_input("Fréquence Cardiaque — bpm", min_value=40, max_value=200, value=70, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("🌿", "Mode de Vie &amp; Antécédents")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            currentSmoker = st.selectbox("Fumeur actuel", [0,1], format_func=lambda x:"🚭 Non" if x==0 else "🚬 Oui")
        with c2:
            cigsPerDay = st.number_input("Cigarettes / jour", min_value=0, max_value=100, value=0, step=1)
        with c3:
            BPMeds = st.selectbox("Médicaments tension", [0,1], format_func=lambda x:"Non" if x==0 else "Oui")
        with c4:
            diabetes = st.selectbox("Diabète", [0,1], format_func=lambda x:"Non" if x==0 else "Oui")

        st.markdown("<br>", unsafe_allow_html=True)
        sec("🧬", "Paramètres Biologiques")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            totChol = st.number_input("Cholestérol total — mg/dL", min_value=100, max_value=600, value=200, step=1)
        with c2:
            sysBP = st.number_input("Pression Systolique — mmHg", min_value=80, max_value=250, value=120, step=1)
        with c3:
            diaBP = st.number_input("Pression Diastolique — mmHg", min_value=40, max_value=150, value=80, step=1)
        with c4:
            glucose = st.number_input("Glycémie — mg/dL", min_value=50, max_value=400, value=100, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  LANCER L'ANALYSE CARDIAQUE", use_container_width=True)

    # ── results ───────────────────────────────────────────────────────────────
    if submitted:
        if not ok:
            st.error("⚠️ Modèles non chargés.")
        else:
            with st.spinner("🧠 Analyse IA en cours…"):
                try:
                    if currentSmoker == 0:
                        cigsPerDay = 0
                    pcd   = run_preprocess(sexe, age, currentSmoker, cigsPerDay,
                                           BPMeds, diabetes, totChol, sysBP, diaBP,
                                           BMI, heartRate, glucose)
                    r_tf  = run_tf(pcd)
                    r_pt  = run_pt(pcd)
                    agree = r_tf == r_pt
                    hi    = r_tf == 1 or r_pt == 1

                    st.markdown("<br>", unsafe_allow_html=True)
                    sec("🎯", "Résultats de l'Analyse")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # three model cards — all self-contained
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        l = "Risque Élevé" if r_tf==1 else "Risque Faible"
                        i = "⚠️" if r_tf==1 else "✅"
                        st.markdown(
                            '<div class="cp-mcard"><div class="cp-mico">🤖</div>'
                            '<div class="cp-mlbl">TensorFlow</div>'
                            '<div class="cp-mval">' + i + " " + l + "</div></div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        l = "Risque Élevé" if r_pt==1 else "Risque Faible"
                        i = "⚠️" if r_pt==1 else "✅"
                        st.markdown(
                            '<div class="cp-mcard"><div class="cp-mico">🔥</div>'
                            '<div class="cp-mlbl">PyTorch</div>'
                            '<div class="cp-mval">' + i + " " + l + "</div></div>",
                            unsafe_allow_html=True
                        )
                    with col3:
                        ci = "🎯" if agree else "⚡"
                        cl = "✓ Accord" if agree else "△ Divergence"
                        st.markdown(
                            '<div class="cp-mcard"><div class="cp-mico">' + ci + "</div>"
                            '<div class="cp-mlbl">Consensus IA</div>'
                            '<div class="cp-mval">' + cl + "</div></div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    if hi:
                        st.markdown(
                            '<div class="cp-high">'
                            '<div class="cp-rtitle">⚠️ Risque Cardiovasculaire Élevé Détecté</div>'
                            '<div class="cp-rsub">Au moins un modèle a identifié des indicateurs de risque</div>'
                            "</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<div class="cp-reco"><div class="cp-reco-hd">Recommandations Prioritaires</div>'
                            '<div class="cp-rrow"><span class="cp-rico">🏥</span><span>Consultez un cardiologue dans les meilleurs délais</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">💊</span><span>Respectez scrupuleusement vos traitements prescrits</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">🏃</span><span>Adoptez une activité physique régulière et adaptée</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">🥗</span><span>Suivez un régime alimentaire cardiovasculaire équilibré</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">🚭</span><span>Cessez le tabac immédiatement si vous fumez</span></div>'
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
                            '<div class="cp-reco"><div class="cp-reco-hd">Conseils Préventifs</div>'
                            '<div class="cp-rrow"><span class="cp-rico">🏃</span><span>Continuez 150 min d\'activité physique par semaine</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">🥗</span><span>Maintenez une alimentation riche en fibres et oméga-3</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">📊</span><span>Effectuez des bilans médicaux annuels de prévention</span></div>'
                            '<div class="cp-rrow"><span class="cp-rico">😌</span><span>Gérez votre stress avec méditation ou relaxation</span></div>'
                            "</div>",
                            unsafe_allow_html=True
                        )
                except Exception as exc:
                    st.error("❌ Erreur : " + str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    sec("📊", "Votre Profil Cardiovasculaire")
    st.markdown("<br>", unsafe_allow_html=True)

    if submitted and ok:
        dark  = dk()
        bgp   = "#1C1814" if dark else "#FFFFFF"
        fc    = "#F2EDE6" if dark else "#1A1410"
        gc    = "#252018" if dark else "#E5E0D8"
        ac    = "#E05A4E" if dark else "#C0392B"
        fill  = "rgba(224,90,78,.18)" if dark else "rgba(192,57,43,.12)"

        cats = ["Âge","IMC","Cholestérol","Pression\nSys.","Glycémie","Fréq.\nCard."]
        vals = [
            min(age/100,1), min(BMI/40,1), min(totChol/300,1),
            min(sysBP/200,1), min(glucose/200,1), min(heartRate/150,1)
        ]
        refs = [.45,.62,.67,.60,.50,.47]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=refs+[refs[0]], theta=cats+[cats[0]], fill="toself",
            name="Référence Saine",
            line=dict(color="rgba(100,200,130,.6)", width=2, dash="dot"),
            fillcolor="rgba(100,200,130,.07)"
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
            name="Votre Profil",
            line=dict(color=ac, width=2.5), fillcolor=fill
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=bgp,
                radialaxis=dict(visible=True, range=[0,1],
                    tickfont=dict(size=9, color=fc), gridcolor=gc, linecolor=gc),
                angularaxis=dict(tickfont=dict(size=10, color=fc, family="Syne"),
                    gridcolor=gc, linecolor=gc)
            ),
            showlegend=True,
            legend=dict(font=dict(color=fc, size=10), bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Radar — Votre profil vs Référence saine",
                font=dict(color=fc, size=13, family="Syne"), x=.5),
            height=460, paper_bgcolor=bgp, plot_bgcolor=bgp,
            margin=dict(l=55,r=55,t=55,b=35), font=dict(color=fc)
        )
        st.plotly_chart(fig, use_container_width=True)

        m1,m2,m3,m4,m5 = st.columns(5)
        with m1: st.metric("Âge", str(age)+" ans")
        with m2:
            bd = "✅ Normal" if 18.5<=BMI<=24.9 else ("⚠️ Surpoids" if BMI<30 else "⚠️ Obèse")
            st.metric("IMC", str(round(BMI,1)), delta=bd)
        with m3: st.metric("Cholestérol", str(totChol)+" mg/dL", delta="✅ OK" if totChol<200 else "⚠️ Élevé")
        with m4: st.metric("Sys. BP", str(sysBP)+" mmHg", delta="✅ Normal" if sysBP<130 else "⚠️ Élevé")
        with m5: st.metric("Glycémie", str(glucose)+" mg/dL", delta="✅ Normal" if glucose<100 else "⚠️ Élevé")

        st.markdown("<br>", unsafe_allow_html=True)

        bc = [ac if v>r else "#3DD68C" for v,r in zip(vals,refs)]
        bl = ["Âge","IMC","Chol.","Pression","Glycémie","Fréq.C."]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=bl, y=refs, name="Référence",
            marker=dict(color="rgba(140,140,140,.22)",line=dict(color="rgba(140,140,140,.35)",width=1)),
            width=.33))
        fig2.add_trace(go.Bar(x=bl, y=vals, name="Votre Profil",
            marker=dict(color=bc, opacity=.88), width=.33))
        fig2.update_layout(
            barmode="group", paper_bgcolor=bgp, plot_bgcolor=bgp,
            font=dict(color=fc, family="DM Sans"),
            title=dict(text="Comparaison par paramètre (normalisé)",
                font=dict(color=fc, size=12, family="Syne"), x=.5),
            yaxis=dict(range=[0,1.15],gridcolor=gc,linecolor=gc),
            xaxis=dict(gridcolor=gc,linecolor=gc),
            legend=dict(font=dict(color=fc),bgcolor="rgba(0,0,0,0)"),
            height=340, margin=dict(l=35,r=35,t=45,b=35)
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        dark = dk()
        bg  = "#1C1814" if dark else "#FFFFFF"
        bdr = "#252018" if dark else "#E5E0D8"
        col = "#5A4E42" if dark else "#9E9285"
        st.markdown(
            '<div style="text-align:center;padding:5rem 2rem;'
            "background:"+bg+";border:2px dashed "+bdr+";"
            "border-radius:22px;color:"+col+";\">"
            '<div style="font-size:3.2rem;margin-bottom:1rem;opacity:.45;">📊</div>'
            '<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:.4rem;">Aucune donnée</div>'
            '<div style="font-size:.86rem;opacity:.6;">Soumettez le formulaire dans l\'onglet <strong>Evaluation</strong></div>'
            "</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GUIDE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    sec("📖", "Guide des Paramètres")
    st.markdown("<br>", unsafe_allow_html=True)

    PARAMS = [
        ("👤","Sexe","Hommes et femmes présentent des profils de risque différents selon l'âge et les hormones."),
        ("🎂","Âge","Facteur majeur. Risque accru après 45 ans chez l'homme, 55 ans chez la femme."),
        ("⚖️","IMC","Poids (kg) / taille² (m). Normal 18.5–24.9. Au-delà, risque cardiovasculaire augmenté."),
        ("💓","Fréquence Cardiaque","Rythme au repos. Valeur saine entre 60 et 100 bpm."),
        ("🚬","Tabagisme","Double le risque coronarien. Même quelques cigarettes par jour sont nocives."),
        ("💊","Antihypertenseurs","Traitement actif de l'hypertension — facteur de risque important."),
        ("🩸","Diabète","L'hyperglycémie chronique endommage les vaisseaux et multiplie le risque."),
        ("🧪","Cholestérol Total","Idéalement sous 200 mg/dL. Au-delà de 240, risque d'athérosclérose élevé."),
        ("📈","Pression Systolique","Pression lors de la contraction. Normale sous 120 mmHg."),
        ("📉","Pression Diastolique","Pression au repos entre battements. Normale sous 80 mmHg."),
        ("🍬","Glycémie","Glucose à jeun. Normal 70–99 mg/dL. Pré-diabète 100–125 mg/dL."),
    ]
    for ic, tm, dc in PARAMS:
        st.markdown(
            '<div class="cp-gc"><div class="cp-gi">'+ic+"</div><div>"
            '<div class="cp-gt">'+tm+"</div>"
            '<div class="cp-gd">'+dc+"</div>"
            "</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    sec("🤖", "Architecture des Modèles")
    st.markdown("<br>", unsafe_allow_html=True)

    dark   = dk()
    aibg   = "#1C1814" if dark else "#FFFFFF"
    aibdr  = "#252018" if dark else "#E5E0D8"
    aitxt  = "#8A7E72" if dark else "#6B6357"
    aiacc  = "#E05A4E" if dark else "#C0392B"

    st.markdown(
        '<div style="background:'+aibg+";border:1px solid "+aibdr+";"
        "border-radius:16px;padding:1.8rem;font-size:.87rem;line-height:1.8;color:"+aitxt+";\">"
        '<strong style="font-family:Syne,sans-serif;color:'+aiacc+'">① Prétraitement</strong><br>'
        "Les 12 paramètres sont standardisés (<em>StandardScaler</em>) puis réduits "
        "par <em>PCA</em> pour éliminer le bruit.<br><br>"
        '<strong style="font-family:Syne,sans-serif;color:'+aiacc+'">② Double Inférence</strong><br>'
        "Le vecteur PCA est soumis à un réseau <em>TensorFlow/Keras</em> et un réseau "
        "<em>PyTorch/TorchScript</em>, entraînés sur Framingham Heart Study.<br><br>"
        '<strong style="font-family:Syne,sans-serif;color:'+aiacc+'">③ Consensus</strong><br>'
        "Un accord entre les deux modèles renforce la fiabilité. En cas de divergence, "
        "une consultation médicale est conseillée."
        "</div>",
        unsafe_allow_html=True
    )


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    '<div class="cp-foot"><strong>CardioPredict AI</strong>'
    " &nbsp;·&nbsp; © 2026 &nbsp;·&nbsp; TensorFlow &amp; PyTorch"
    " &nbsp;·&nbsp; ❤️ pour la santé cardiovasculaire</div>",
    unsafe_allow_html=True
)
