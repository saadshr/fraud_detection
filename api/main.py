# ============================================================
# api/main.py
# But : exposer le modèle ML via une API REST professionnelle
# Accès Swagger : http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ── APP ───────────────────────────────────────────────────
app = FastAPI(
    title="Smart Fraud Detection API",
    description="""
    ## API de Détection de Fraude par Carte Bancaire
    
    Utilise un modèle Random Forest / XGBoost entraîné sur 284,807 transactions réelles.
    
    ### Endpoints :
    - **POST /predict** — Analyser une transaction
    - **GET /health** — Statut de l'API
    - **GET /model/info** — Informations sur le modèle
    """,
    version="1.0.0"
)

# ── CHARGEMENT DU MODÈLE ─────────────────────────────────
try:
    model = joblib.load('models/fraud_model.pkl')
    n_features = model.n_features_in_
    print(f"✅ Modèle chargé — {n_features} features attendues")
except:
    model = None
    n_features = 30
    print("⚠️  Modèle non trouvé")

# ── SCHÉMAS ──────────────────────────────────────────────
class Transaction(BaseModel):
    features: List[float] = Field(
        ...,
        description="Valeurs numériques représentant la transaction",
        example=[0.1, -1.2, 0.5, -0.3, 1.1, -0.8, 0.2, -0.1,
                 0.9, -1.5, 0.3, -0.6, 1.2, -0.4, 0.7, -0.2,
                 0.5, -0.9, 1.0, -0.3, 0.8, -0.5, 0.1, 0.4,
                 -0.7, 0.3, -0.1, 0.6, -0.4, 0.2]
    )

class PredictionResult(BaseModel):
    is_fraud:          bool
    fraud_probability: float
    risk_level:        str
    prediction_label:  str
    n_features_used:   int
    timestamp:         str
    model_version:     str

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Smart Fraud Detection API v1.0",
        "docs": "http://localhost:8000/docs",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "n_features_expected": n_features,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
def model_info():
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "model_type":       type(model).__name__,
        "dataset":          "Credit Card Fraud Detection (Kaggle)",
        "training_samples": "~456,000 (après SMOTE)",
        "features_expected": n_features,
        "version":          "1.0.0"
    }

@app.post("/predict", response_model=PredictionResult)
def predict(transaction: Transaction):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if len(transaction.features) != n_features:
        raise HTTPException(
            status_code=422,
            detail=f"{n_features} features requises, reçu {len(transaction.features)}"
        )
    try:
        X = np.array(transaction.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        proba      = model.predict_proba(X)[0][1]

        risk = ("CRITIQUE" if proba > 0.8
           else "ÉLEVÉ"    if proba > 0.5
           else "MODÉRÉ"   if proba > 0.3
           else "FAIBLE")

        return PredictionResult(
            is_fraud=          bool(prediction),
            fraud_probability= round(float(proba), 4),
            risk_level=        risk,
            prediction_label=  "🚨 FRAUDE DÉTECTÉE" if prediction else "✅ TRANSACTION NORMALE",
            n_features_used=   n_features,
            timestamp=         datetime.now().isoformat(),
            model_version=     "1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))