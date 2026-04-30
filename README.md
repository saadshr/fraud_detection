# 🔍 Smart Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-ff4b4b)

> Système de détection de fraude bancaire utilisant Machine Learning sur 284,807 transactions réelles.

## 🎯 Résultats

| Modèle | F1-Score | ROC-AUC | Precision | Recall |
|--------|----------|---------|-----------|--------|
| XGBoost | **0.89+** | **0.97+** | 0.91+ | 0.87+ |
| Random Forest | 0.87+ | 0.96+ | 0.89+ | 0.85+ |
| Logistic Regression | 0.75+ | 0.93+ | 0.78+ | 0.72+ |

## 🏗️ Architecture

```
fraud_detection/
├── data/           ← Dataset (raw + processed)
├── src/            ← Pipeline ML complet
├── models/         ← Modèle entraîné (.pkl)
├── api/            ← FastAPI REST API
├── dashboard/      ← Streamlit Dashboard
└── outputs/        ← Visualisations EDA
```

## 🚀 Installation rapide

```bash
git clone https://github.com/saadshr/fraud-detection
cd fraud-detection
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

## ⚙️ Utilisation

```bash
# Pipeline complet
python src/01_data_exploration.py
python src/02_data_cleaning.py
python src/03_eda.py
python src/04_feature_engineering.py
python src/05_train_models.py
python src/06_evaluate.py

# API
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Dashboard
streamlit run dashboard/app.py
```

## 🔑 Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · SMOTE · FastAPI · Streamlit · Joblib · Seaborn · Matplotlib

## 👤 Auteur

**Saad Sahraoui** — Étudiant Ingénieur IA & Data Science, ENSAM Rabat  
📧 saad20azmm@gmail.com | 🔗 [GitHub](https://github.com/saadshr) | [LinkedIn](https://linkedin.com/in/saadshr)
