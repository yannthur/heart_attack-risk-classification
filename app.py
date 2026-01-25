import os
# Suppress TensorFlow logs (Must be before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import torch
import plotly.graph_objects as go
import numpy as np
import warnings

# Suppress Sklearn version warnings if re-training isn't an option
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration de la page
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    h1 {
        color: #1e3a8a;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
        animation: pulse 2s infinite;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Style ajusté pour le bouton de formulaire */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Chargement des modèles
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    pca_model = joblib.load("pca.pkl")
    model_torch = torch.jit.load("torch_model.pth")
    model_torch.eval()
    model_tensorflow = tf.keras.models.load_model("models_loadedtensorflow_model.keras")
    return scaler, pca_model, model_torch, model_tensorflow

try:
    scaler, pca_model, model_torch, model_tensorflow = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Erreur lors du chargement des modèles: {e}")

def preprocessing_pipeline(sexe, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    cols = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    data = pd.DataFrame([[sexe, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]], columns=cols)
    data_scale = scaler.transform(data)
    data_pca = pca_model.transform(data_scale)
    return data_pca

def tensorflow_prediction(data_pca, model=None):
    if model is None:
        model = model_tensorflow
    prediction = model.predict(data_pca, verbose=0)
    return int(round(prediction[0][0]))

def pytorch_prediction(data_pca, model=None):
    if model is None:
        model = model_torch
    inputs = torch.tensor(data_pca, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(inputs)
    return int(round(prediction.item()))

# En-tête de l'application
st.markdown("<h1>❤️ CardioPredict AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Évaluation intelligente du risque cardiovasculaire basée sur l'IA</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/200/heart-health.png", width=150)
    st.markdown("### 📊 À propos")
    st.markdown("""
    Cette application utilise des réseaux de neurones avancés (TensorFlow & PyTorch) 
    pour prédire le risque cardiovasculaire basé sur vos paramètres médicaux.
    """)
    st.markdown("### 🔒 Confidentialité")
    st.info("Vos données ne sont pas stockées et restent totalement confidentielles.")
    st.markdown("### 🎯 Modèles IA")
    st.success("✅ TensorFlow Neural Network")
    st.success("✅ PyTorch Neural Network")
    st.markdown("### ⚕️ Avertissement")
    st.warning("Cet outil est à but informatif uniquement. Consultez toujours un professionnel de santé.")

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["🩺 Évaluation", "📈 Statistiques", "ℹ️ Guide"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Informations du Patient")
    
    # --- MODIFICATION ICI : AJOUT DU FORMULAIRE (st.form) ---
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Informations Générales")
            sexe = st.selectbox("👤 Sexe", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
            age = st.number_input("🎂 Âge", min_value=1, max_value=120, value=50, step=1)
            BMI = st.number_input("⚖️ IMC (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            heartRate = st.number_input("💓 Fréquence Cardiaque (bpm)", min_value=40, max_value=200, value=70, step=1)
        
        with col2:
            st.markdown("#### Habitudes de Vie")
            currentSmoker = st.selectbox("🚬 Fumeur Actuel", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            cigsPerDay = st.number_input("📊 Cigarettes/Jour", min_value=0, max_value=100, value=0, step=1) # Disabled non supporté bien dans form, on laisse activé
            
            st.markdown("#### Conditions Médicales")
            BPMeds = st.selectbox("💊 Médicaments Tension", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            diabetes = st.selectbox("🩸 Diabète", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        
        with col3:
            st.markdown("#### Mesures Biologiques")
            totChol = st.number_input("🧪 Cholestérol Total (mg/dL)", min_value=100, max_value=600, value=200, step=1)
            sysBP = st.number_input("📈 Pression Systolique (mmHg)", min_value=80, max_value=250, value=120, step=1)
            diaBP = st.number_input("📉 Pression Diastolique (mmHg)", min_value=40, max_value=150, value=80, step=1)
            glucose = st.number_input("🍬 Glucose (mg/dL)", min_value=50, max_value=400, value=100, step=1)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Le bouton doit être un form_submit_button
        submitted = st.form_submit_button("🔍 ANALYSER LE RISQUE CARDIAQUE", type="primary", use_container_width=True)

    # Logique de prédiction hors du bloc form, déclenchée par "submitted"
    if submitted:
        if models_loaded:
            with st.spinner("🧠 Analyse en cours par les réseaux de neurones..."):
                try:
                    # Correction logique: si non fumeur, cigs = 0 forcés
                    if currentSmoker == 0:
                        cigsPerDay = 0
                        
                    # Prétraitement
                    data_pca = preprocessing_pipeline(
                        sexe, age, currentSmoker, cigsPerDay, BPMeds, 
                        diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
                    )
                    
                    # Prédictions
                    tf_pred = tensorflow_prediction(data_pca, model_tensorflow)
                    torch_pred = pytorch_prediction(data_pca, model_torch)
                    
                    # Résultats
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("## 🎯 Résultats de l'Analyse")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("### 🤖 TensorFlow")
                        st.markdown(f"**{'Risque Élevé ⚠️' if tf_pred == 1 else 'Risque Faible ✅'}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("### 🔥 PyTorch")
                        st.markdown(f"**{'Risque Élevé ⚠️' if torch_pred == 1 else 'Risque Faible ✅'}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        consensus = tf_pred == torch_pred
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("### 🎯 Consensus")
                        st.markdown(f"**{'Accord ✓' if consensus else 'Divergence ⚠️'}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Résultat final
                    st.markdown("<br>", unsafe_allow_html=True)
                    if tf_pred == 1 or torch_pred == 1:
                        st.markdown("<div class='risk-high'>⚠️ PROFIL À RISQUE CARDIOVASCULAIRE ÉLEVÉ</div>", unsafe_allow_html=True)
                        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
                        st.markdown("""
                        **Recommandations:**
                        - 🏥 Consultez un cardiologue dans les plus brefs délais
                        - 💊 Respectez scrupuleusement vos traitements
                        - 🏃 Adoptez une activité physique régulière
                        - 🥗 Suivez un régime alimentaire équilibré
                        - 🚭 Arrêtez le tabac si vous fumez
                        """)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='risk-low'>✅ PROFIL À RISQUE CARDIOVASCULAIRE FAIBLE</div>", unsafe_allow_html=True)
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown("""
                        **Recommandations pour maintenir votre santé:**
                        - 🏃 Continuez une activité physique régulière
                        - 🥗 Maintenez une alimentation saine
                        - 📊 Effectuez des contrôles médicaux réguliers
                        - 😌 Gérez votre stress efficacement
                        """)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")
        else:
            st.error("⚠️ Les modèles ne sont pas chargés. Veuillez vérifier les fichiers.")

with tab2:
    st.markdown("### 📊 Visualisation de vos Paramètres")
    
    if 'age' in locals():
        # Création d'un graphique radar
        categories = ['Âge\n(normalisé)', 'IMC', 'Cholestérol\n(normalisé)', 
                     'Pression\nSystolique', 'Glucose\n(normalisé)', 'Fréquence\nCardiaque']
        
        # Normalisation approximative pour le graphique
        values = [
            min(age / 100, 1),
            min(BMI / 40, 1),
            min(totChol / 300, 1),
            min(sysBP / 200, 1),
            min(glucose / 200, 1),
            min(heartRate / 150, 1)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Vos valeurs',
            line_color='rgb(102, 126, 234)',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Profil de Santé Cardiovasculaire",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Fixé pour éviter le warning use_container_width
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques supplémentaires
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Âge", f"{age} ans")
        with col2:
            imc_status = "Normal" if 18.5 <= BMI <= 24.9 else "Attention"
            st.metric("IMC", f"{BMI:.1f}", delta=imc_status)
        with col3:
            st.metric("Cholestérol", f"{totChol} mg/dL")
        with col4:
            st.metric("Fréquence C.", f"{heartRate} bpm")

with tab3:
    st.markdown("### 📖 Guide d'Utilisation")
    
    with st.expander("🔍 Comprendre les Paramètres"):
        st.markdown("""
        - **Sexe**: Impact sur le risque cardiovasculaire
        - **Âge**: Facteur de risque majeur
        - **IMC**: Indice de Masse Corporelle (poids/taille²)
        - **Fumeur**: Facteur de risque majeur modifiable
        - **Cigarettes/Jour**: Quantification du tabagisme
        - **Médicaments Tension**: Traitement antihypertenseur
        - **Diabète**: Facteur de risque métabolique
        - **Cholestérol Total**: Lipides sanguins
        - **Pressions Systolique/Diastolique**: Tension artérielle
        - **Glucose**: Glycémie à jeun
        - **Fréquence Cardiaque**: Rythme cardiaque au repos
        """)
    
    with st.expander("🤖 Comment fonctionne l'IA?"):
        st.markdown("""
        Notre système utilise deux réseaux de neurones profonds:
        
        1. **Prétraitement des données**:
           - Standardisation avec scaler
           - Réduction dimensionnelle (PCA)
        
        2. **Modèles de prédiction**:
           - Réseau TensorFlow
           - Réseau PyTorch
        
        3. **Consensus**: Les deux modèles analysent vos données indépendamment pour une prédiction fiable.
        """)
    
    with st.expander("⚕️ Interprétation des Résultats"):
        st.markdown("""
        - **Risque Faible**: Profil cardiovasculaire favorable - Maintenez vos bonnes habitudes
        - **Risque Élevé**: Profil nécessitant une attention médicale - Consultez un professionnel
        - **Consensus**: Accord entre les deux modèles IA - Résultat fiable
        - **Divergence**: Résultats différents - Prudence recommandée, consultation conseillée
        """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b;'>CardioPredict AI © 2026 | Développé avec ❤️ pour votre santé | " 
    "Powered by TensorFlow & PyTorch</p>",
    unsafe_allow_html=True
)
