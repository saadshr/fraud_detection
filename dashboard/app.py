# ============================================================
# dashboard/app.py
# But : dashboard interactif pour visualiser les prédictions
# ============================================================

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
@st.cache_resource
def load_model():
    return joblib.load('models/fraud_model.pkl')

@st.cache_data
def load_data():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    return X_test, y_test

model = load_model()
X_test, y_test = load_data()
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
    st.info("Entrez les 25 valeurs de features séparées par des virgules")
    user_input = st.text_area(
        "Features (25 valeurs)",
        value="0.1,-1.2,0.5,-0.3,1.1,-0.8,0.2,-0.1,0.9,-1.5,0.3,-0.6,1.2,-0.4,0.7,-0.2,0.5,-0.9,1.0,-0.3,0.8,-0.5,0.1,0.4,-0.7"
    )
    if st.button("🔍 Analyser", type="primary"):
        try:
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) == 25:
                X_input = np.array(values).reshape(1, -1)
                pred  = model.predict(X_input)[0]
                proba = model.predict_proba(X_input)[0][1]
                if pred == 1:
                    st.error(f"🚨 FRAUDE DÉTECTÉE — Probabilité : {proba:.1%}")
                else:
                    st.success(f"✅ TRANSACTION NORMALE — Probabilité fraude : {proba:.1%}")
                st.progress(float(proba))
            else:
                st.error(f"25 valeurs requises, vous avez entré {len(values)}")
        except:
            st.error("Format invalide. Utilisez des nombres séparés par des virgules.")

with tab3:
    st.markdown("""
    ## Smart Fraud Detection System
    **Développé par : Saad Sahraoui — ENSAM Rabat**
    
    - **Dataset** : Credit Card Fraud Detection (Kaggle — 284,807 transactions)
    - **Modèle** : Random Forest / XGBoost avec SMOTE
    - **Features** : 25 features sélectionnées
    - **GitHub** : github.com/saadshr
    """)

