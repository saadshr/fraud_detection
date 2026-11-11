# ============================================================
# dashboard/app.py
# But : dashboard interactif pour visualiser les prédictions
# ============================================================

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Smart Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# ── HEADER ────────────────────────────────────────────────
st.title("🔍 Smart Fraud Detection System")
st.markdown("**Système de détection de fraude par carte bancaire — ML Dashboard**")
st.divider()

# ── CHARGER ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/fraud_model.pkl"
X_TEST_PATH = ROOT / "data/processed/X_test.csv"
Y_TEST_PATH = ROOT / "data/processed/y_test.csv"

if not all(path.is_file() for path in (MODEL_PATH, X_TEST_PATH, Y_TEST_PATH)):
    st.error("Le modèle ou les données de test sont manquants. Suivez les étapes d’entraînement dans le README.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze(axis='columns')
    return X_test, y_test

model = load_model()
X_test, y_test = load_data()
n_features = model.n_features_in_
feature_names = list(getattr(model, "feature_names_in_", X_test.columns))
if len(feature_names) != n_features or set(feature_names) != set(X_test.columns):
    st.error("Les colonnes de test ne correspondent pas au modèle. Régénérez les données et le modèle ensemble.")
    st.stop()

X_test = X_test.loc[:, feature_names]
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ── MÉTRIQUES ─────────────────────────────────────────────
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🎯 Precision", f"{precision_score(y_test, y_pred):.1%}")
col2.metric("📡 Recall",    f"{recall_score(y_test, y_pred):.1%}")
col3.metric("⚖️ F1-Score",  f"{f1_score(y_test, y_pred):.1%}")
col4.metric("📈 ROC-AUC",  f"{roc_auc_score(y_test, y_pred_proba):.4f}")
col5.metric("🔢 Tests",    f"{len(y_test):,}")

st.divider()

# ── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Analyse", "🔮 Prédiction", "ℹ️ À propos"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des prédictions")
        fig, ax = plt.subplots()
        ax.hist(y_pred_proba[y_test==0], bins=50, alpha=0.7,
               color='#2563EB', label='Normal')
        ax.hist(y_pred_proba[y_test==1], bins=50, alpha=0.7,
               color='#EF4444', label='Fraude')
        ax.set_xlabel('Probabilité de fraude')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal','Fraude'],
                    yticklabels=['Normal','Fraude'])
        ax.set_ylabel('Réel')
        ax.set_xlabel('Prédit')
        st.pyplot(fig)
        plt.close()

with tab2:
    st.subheader("🔮 Analyser une transaction")
    st.info(f"Entrez {n_features} valeurs numériques déjà prétraitées, séparées par des virgules.")
    st.caption("Le formulaire contient un exemple prérempli à partir du jeu de test.")
    example = ", ".join(map(str, X_test.iloc[0]))
    with st.form("transaction"):
        user_input = st.text_area(
            f"Variables ({n_features} valeurs)",
            value=example,
            help="Ordre des variables : " + ", ".join(feature_names),
        )
        submitted = st.form_submit_button("🔍 Analyser", type="primary")

    if submitted:
        try:
            values = [float(value.strip()) for value in user_input.split(",")]
        except ValueError:
            st.error("Saisissez uniquement des nombres séparés par des virgules, avec un point pour les décimales.")
        else:
            if len(values) != n_features:
                st.error(f"{n_features} valeurs requises, {len(values)} reçues.")
            elif not np.isfinite(values).all():
                st.error("Toutes les valeurs doivent être des nombres finis.")
            else:
                transaction = pd.DataFrame([values], columns=feature_names)
                prediction = model.predict(transaction)[0]
                probability = float(model.predict_proba(transaction)[0][1])
                if prediction == 1:
                    st.error(f"🚨 FRAUDE DÉTECTÉE — Probabilité : {probability:.1%}")
                else:
                    st.success(f"✅ TRANSACTION NORMALE — Probabilité de fraude : {probability:.1%}")
                st.progress(probability)


with tab3:
    st.markdown(f"""
    ## Smart Fraud Detection System
    **Développé par : Saad Sahraoui — ENSAM Rabat**
    
    - **Dataset** : Credit Card Fraud Detection (Kaggle — 284,807 transactions)
    - **Modèle** : {type(model).__name__} avec SMOTE
    - **Features** : {n_features} variables attendues
    - **GitHub** : github.com/saadshr
    """)

