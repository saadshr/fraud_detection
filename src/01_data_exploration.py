# ============================================================
# 01_data_exploration.py
# But : charger et explorer le dataset pour comprendre
#       ce avec quoi on travaille
# ============================================================

import pandas as pd
import numpy as np
import os

# ── 1. CHARGER LE DATASET ──────────────────────────────────
print("Chargement du dataset...")
df = pd.read_csv('data/raw/creditcard.csv')
print(f"✅ Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

# ── 2. APERÇU GÉNÉRAL ─────────────────────────────────────
print("\n=== APERÇU DES 5 PREMIÈRES LIGNES ===")
print(df.head())

print("\n=== INFOS GÉNÉRALES ===")
print(df.info())

print("\n=== STATISTIQUES DESCRIPTIVES ===")
print(df.describe())

# ── 3. ANALYSE DU DÉSÉQUILIBRE ────────────────────────────
# C'est le problème CENTRAL de ce projet
# 99.83% des transactions sont normales
# 0.17% sont des fraudes → le modèle doit quand même les trouver
print("\n=== DISTRIBUTION DES CLASSES ===")
class_counts = df['Class'].value_counts()
class_pct = df['Class'].value_counts(normalize=True) * 100

print(f"Transactions normales (0) : {class_counts[0]:,} ({class_pct[0]:.2f}%)")
print(f"Transactions fraude (1)   : {class_counts[1]:,} ({class_pct[1]:.2f}%)")
print(f"\n⚠️  Ratio déséquilibre : 1 fraude pour {int(class_counts[0]/class_counts[1])} transactions normales")

# ── 4. VALEURS MANQUANTES ─────────────────────────────────
missing = df.isnull().sum()
print(f"\n=== VALEURS MANQUANTES ===")
if missing.sum() == 0:
    print("✅ Aucune valeur manquante")
else:
    print(missing[missing > 0])

# ── 5. ANALYSE DES COLONNES ───────────────────────────────
print("\n=== COLONNES ===")
print("Colonnes V1-V28 : transformées par PCA (anonymisées)")
print("Colonne Time    : secondes depuis la première transaction")
print("Colonne Amount  : montant de la transaction (€)")
print("Colonne Class   : 0=Normal, 1=Fraude (TARGET)")

# ── 6. ANALYSE DU MONTANT ─────────────────────────────────
print("\n=== ANALYSE DU MONTANT ===")
print(f"Montant moyen normal : {df[df['Class']==0]['Amount'].mean():.2f}€")
print(f"Montant moyen fraude : {df[df['Class']==1]['Amount'].mean():.2f}€")
print(f"Montant max fraude   : {df[df['Class']==1]['Amount'].max():.2f}€")

print("\n✅ Exploration terminée !")
