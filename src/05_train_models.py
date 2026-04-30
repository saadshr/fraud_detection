# ============================================================
# 05_train_models.py
# But : entraîner et comparer plusieurs modèles ML
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                               confusion_matrix, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os

os.makedirs('models', exist_ok=True)

# ── CHARGEMENT ────────────────────────────────────────────
X_train = pd.read_csv('data/processed/X_train.csv')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ── DÉFINIR LES MODÈLES ───────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1    # utilise tous les CPUs
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

results = {}

# ── ENTRAÎNER ET ÉVALUER CHAQUE MODÈLE ───────────────────
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🤖 Entraînement : {name}")

    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start

    y_pred     = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision  = precision_score(y_test, y_pred)
    recall     = recall_score(y_test, y_pred)
    f1         = f1_score(y_test, y_pred)
    roc_auc    = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'model':     model,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'roc_auc':   roc_auc,
        'time':      duration
    }

    print(f"  ⏱️  Temps    : {duration:.1f}s")
    print(f"  🎯 Precision : {precision:.4f}")
    print(f"  📡 Recall    : {recall:.4f}")
    print(f"  ⚖️  F1-Score  : {f1:.4f}")
    print(f"  📈 ROC-AUC   : {roc_auc:.4f}")

# ── SÉLECTION DU MEILLEUR MODÈLE ──────────────────────────
best_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_name]['model']

print(f"\n{'='*50}")
print(f"🏆 MEILLEUR MODÈLE : {best_name}")
print(f"   F1-Score : {results[best_name]['f1']:.4f}")
print(f"   ROC-AUC  : {results[best_name]['roc_auc']:.4f}")

# ── SAUVEGARDER LE MEILLEUR MODÈLE ────────────────────────
joblib.dump(best_model, 'models/fraud_model.pkl')
print(f"\n✅ Modèle sauvegardé : models/fraud_model.pkl")

# ── TABLEAU COMPARATIF ────────────────────────────────────
comparison = pd.DataFrame({
    name: {'Precision': r['precision'], 'Recall': r['recall'],
           'F1': r['f1'], 'ROC-AUC': r['roc_auc']}
    for name, r in results.items()
}).round(4)
print("\n📊 Tableau comparatif :")
print(comparison)
