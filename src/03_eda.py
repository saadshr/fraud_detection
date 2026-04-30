# ============================================================
# 03_eda.py
# But : visualiser les données pour comprendre les patterns
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)
sns.set_style('darkgrid')
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv('data/raw/creditcard.csv')

# ── 1. DISTRIBUTION DES CLASSES ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts = df['Class'].value_counts()
colors = ['#2563EB', '#EF4444']
axes[0].bar(['Normal', 'Fraude'], counts.values, color=colors, edgecolor='white')
axes[0].set_title('Distribution des Classes', fontweight='bold')
axes[0].set_ylabel('Nombre de transactions')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=['Normal (99.83%)', 'Fraude (0.17%)'],
           colors=colors, autopct='%1.2f%%', startangle=90)
axes[1].set_title('Proportion des Classes', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_class_distribution.png')
print("✅ outputs/01_class_distribution.png")
plt.close()

# ── 2. DISTRIBUTION DES MONTANTS ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df[df['Class']==0]['Amount'].hist(bins=50, ax=axes[0], color='#2563EB', alpha=0.7)
axes[0].set_title('Montants — Transactions Normales', fontweight='bold')
axes[0].set_xlabel('Montant (€)')

df[df['Class']==1]['Amount'].hist(bins=50, ax=axes[1], color='#EF4444', alpha=0.7)
axes[1].set_title('Montants — Transactions Frauduleuses', fontweight='bold')
axes[1].set_xlabel('Montant (€)')

plt.tight_layout()
plt.savefig('outputs/02_amount_distribution.png')
print("✅ outputs/02_amount_distribution.png")
plt.close()

# ── 3. HEATMAP DE CORRÉLATION ─────────────────────────────
# Montre quelles features sont liées à la fraude
print("⏳ Génération de la heatmap (peut prendre 30 secondes)...")
fig, ax = plt.subplots(figsize=(20, 16))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title('Matrice de Corrélation — Credit Card Fraud Dataset',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png')
print("✅ outputs/03_correlation_heatmap.png")
plt.close()

# ── 4. TOP FEATURES CORRÉLÉES AVEC LA FRAUDE ──────────────
corr_with_fraud = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
top_features = corr_with_fraud.head(15)
colors = ['#EF4444' if x > 0.3 else '#2563EB' for x in top_features.values]
ax.barh(top_features.index, top_features.values, color=colors)
ax.set_title('Top 15 Features Corrélées avec la Fraude', fontweight='bold')
ax.set_xlabel('Corrélation absolue avec Class')
plt.tight_layout()
plt.savefig('outputs/04_top_features.png')
print("✅ outputs/04_top_features.png")
plt.close()

print("\n✅ EDA terminée ! Graphiques dans outputs/")
print(f"\n📊 Top 5 features liées à la fraude :")
print(corr_with_fraud.head(5))
