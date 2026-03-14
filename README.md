# ❤️ CardioPredict AI

## Version de test ![Streamlit cloud](https://heart-attack-risk-classification.streamlit.app/)

> **Système de prédiction du risque cardiovasculaire par double réseau de neurones profonds**
> — TensorFlow · PyTorch · Streamlit

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

---

## 📋 Vue d'ensemble

**CardioPredict AI** est une application web de prédiction du risque de maladie cardiovasculaire à 10 ans, construite sur la cohorte **Framingham Heart Study**. Elle déploie deux réseaux de neurones profonds entraînés indépendamment (TensorFlow/Keras et PyTorch/TorchScript) et établit un **consensus inter-modèles** pour maximiser la fiabilité clinique du résultat.

Le projet couvre l'intégralité du cycle de la donnée de santé : nettoyage, imputation, normalisation, réduction dimensionnelle, modélisation, évaluation et mise en production via une interface clinique interactive.

---

## 🩺 Contexte Médical & Données

### Dataset — Framingham Heart Study

| Attribut | Détail |
|---|---|
| **Source** | [Framingham Heart Study](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) — étude longitudinale de référence en cardiologie |
| **Taille** | ~4 240 patients · 15 variables cliniques |
| **Cible** | `TenYearCHD` — risque d'événement coronarien à 10 ans (binaire) |
| **Type** | Données médicales réelles, déidentifiées |

### Variables Cliniques Utilisées

| Variable | Type | Description Médicale |
|---|---|---|
| `age` | Numérique | Âge du patient (années) |
| `male` | Binaire | Sexe biologique |
| `currentSmoker` | Binaire | Statut tabagique actif |
| `cigsPerDay` | Numérique | Quantification du tabagisme journalier |
| `BPMeds` | Binaire | Traitement antihypertenseur en cours |
| `diabetes` | Binaire | Diabète diagnostiqué |
| `totChol` | Numérique | Cholestérol total (mg/dL) |
| `sysBP` | Numérique | Pression artérielle systolique (mmHg) |
| `diaBP` | Numérique | Pression artérielle diastolique (mmHg) |
| `BMI` | Numérique | Indice de Masse Corporelle (kg/m²) |
| `heartRate` | Numérique | Fréquence cardiaque au repos (bpm) |
| `glucose` | Numérique | Glycémie à jeun (mg/dL) |

---

## 🔬 Pipeline de Traitement des Données de Santé

La robustesse d'un modèle médical repose avant tout sur la qualité du prétraitement. Ce projet implémente un pipeline complet, reproductible et sérialisé.

```
Données Brutes (CSV)
        │
        ▼
┌───────────────────────────────┐
│  1. Analyse Exploratoire      │  Distribution, corrélations, outliers
│     (EDA)                     │  Visualisation des déséquilibres de classes
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  2. Nettoyage & Imputation    │  Suppression des doublons
│                               │  Imputation par médiane (variables numériques)
│                               │  Cohérence métier (ex: cigsPerDay=0 si non-fumeur)
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  3. Gestion du Déséquilibre   │  Classe minoritaire (~15% de positifs)
│                               │  Stratégie : class_weight / oversampling
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4. Normalisation             │  StandardScaler → μ=0, σ=1
│     (StandardScaler)          │  Sérialisé : scaler.pkl
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  5. Réduction Dimensionnelle  │  PCA — conservation de la variance optimale
│     (PCA)                     │  Sérialisé : pca.pkl
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  6. Split Stratifié           │  Train / Validation / Test
│                               │  Stratification sur la classe cible
└───────────────┬───────────────┘
                │
                ▼
         Modélisation
```

> **Note sur la reproductibilité** : le `StandardScaler` et le `PCA` sont entraînés **uniquement sur le set d'entraînement** et appliqués au set de test — éliminant tout risque de data leakage.

---

## 🧠 Architecture des Modèles

### Stratégie Double Modèle

L'utilisation de deux frameworks indépendants permet de :
- Réduire le risque de faux négatifs (non-détection d'un risque réel)
- Identifier les cas ambigus via la divergence inter-modèles
- Renforcer la confiance clinique par consensus

### Modèle TensorFlow / Keras

```
Input (n_components PCA)
    │
    Dense(128, activation='relu') + BatchNormalization + Dropout(0.3)
    │
    Dense(64, activation='relu')  + BatchNormalization + Dropout(0.2)
    │
    Dense(32, activation='relu')
    │
    Dense(1, activation='sigmoid')
    │
Output — Probabilité P(CHD=1)
```

**Paramètres clés :**
- Optimiseur : `Adam` (lr=1e-3, weight decay)
- Loss : `BinaryCrossentropy`
- Callbacks : `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Export : `tensorflow_model.keras`

### Modèle PyTorch / TorchScript

Architecture identique, implémentée nativement en PyTorch pour la comparaison inter-framework et l'export TorchScript (`torch.jit.script`).

- Optimiseur : `AdamW`
- Scheduler : `CosineAnnealingLR`
- Export : `torch_model.pth` (TorchScript — inference sans dépendance à la définition de classe)

### Métriques d'Évaluation

| Métrique | TF Model | PT Model |
|---|---|---|
| **AUC-ROC** | — | — |
| **F1-Score** | — | — |
| **Recall (Sensibilité)** | — | — |
| **Precision** | — | — |
| **Accuracy** | — | — |

> *Remplir avec les résultats de votre entraînement*

**Priorité au Recall** : dans un contexte clinique, un faux négatif (risque non détecté) est plus coûteux qu'un faux positif. Le seuil de classification est ajusté en conséquence.

---

## 🖥️ Application Streamlit

L'interface clinique expose le pipeline complet en temps réel.

### Fonctionnalités

- **Formulaire structuré** en 3 sections médicales (Profil · Mode de vie · Biologie)
- **Inférence dual-modèle** simultanée avec affichage du consensus
- **Radar cardiovasculaire** comparant le profil patient à une référence saine normalisée
- **Visualisation comparative** par paramètre (normalisé)
- **Mode sombre / clair** — bascule dynamique sans rechargement
- **Guide clinique** interactif expliquant chaque variable

---

## ⚙️ Installation & Lancement

### Prérequis

- Python 3.10+
- pip

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/cardiopredict-ai.git
cd cardiopredict-ai

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Lancement de l'application

```bash
streamlit run app.py
```

L'application est accessible sur `http://localhost:8501`


---

## 📦 Dépendances

```txt
streamlit>=1.30.0
tensorflow>=2.13.0
torch>=2.1.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
joblib>=1.3.0
```

---

## 🔒 Éthique & Confidentialité

- **Aucune donnée patient n'est collectée ou stockée** — toutes les inférences sont réalisées localement en session
- Le dataset Framingham est utilisé **à des fins académiques et pédagogiques uniquement**
- L'application est **explicitement présentée comme un outil informatif**, non diagnostique
- Les recommandations générées invitent systématiquement à consulter un professionnel de santé

---

## ⚕️ Avertissement Médical

> Ce projet est réalisé à des fins **éducatives et de démonstration technique**. Les prédictions produites **ne constituent pas un diagnostic médical** et ne sauraient remplacer l'avis d'un professionnel de santé qualifié. Toute décision médicale doit être prise en consultation avec un médecin.

---

## 👤 Auteur

**[NZOGNI OMONG Yann Arthur]**

Passionné par l'intersection entre **intelligence artificielle et santé numérique**, ce projet illustre ma capacité à :

- Maîtriser le cycle complet de la donnée de santé — de l'EDA au déploiement
- Appliquer les bonnes pratiques de preprocessing sur des données médicales sensibles (imputation, normalisation, évitement du data leakage)
- Concevoir et entraîner des modèles de deep learning avec TensorFlow et PyTorch sur des problèmes de classification clinique déséquilibrée
- Déployer des modèles en production via des interfaces utilisateur accessibles

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/votre-profil)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/votre-username)

---

---

*Développé avec ❤️ pour la santé cardiovasculaire*
