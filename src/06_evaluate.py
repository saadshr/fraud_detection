# ============================================================
# 06_evaluate.py
# But : évaluation complète avec toutes les métriques pro
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                               roc_curve, roc_auc_score, precision_recall_curve)
import joblib

model   = joblib.load('models/fraud_model.pkl')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ── 1. RAPPORT COMPLET ────────────────────────────────────
print("="*60)
print("RAPPORT D'ÉVALUATION COMPLET")
print("="*60)
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Fraude']))
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_pred_proba):.4f}")

# ── 2. CONFUSION MATRIX ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Fraude'],
            yticklabels=['Normal', 'Fraude'])
axes[0].set_title('Confusion Matrix', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Réel')
axes[0].set_xlabel('Prédit')

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 Confusion Matrix :")
print(f"  Vrais Positifs  (TP) : {tp}  ← fraudes détectées ✅")
print(f"  Faux Positifs   (FP) : {fp}  ← normales classées fraude")
print(f"  Vrais Négatifs  (TN) : {tn}  ← normales détectées ✅")
print(f"  Faux Négatifs   (FN) : {fn}  ← fraudes manquées ❌")

# ── 3. ROC CURVE ──────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

axes[1].plot(fpr, tpr, color='#2563EB', lw=2,
             label=f'ROC Curve (AUC = {auc:.4f})')
axes[1].plot([0,1], [0,1], color='gray', linestyle='--', label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#2563EB')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold', fontsize=13)
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/05_evaluation.png', dpi=150)
print("\n✅ outputs/05_evaluation.png")
plt.close()

print("\n✅ Évaluation complète terminée !")