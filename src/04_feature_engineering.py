# ============================================================
# 04_feature_engineering.py
# But : créer de nouvelles features pour améliorer le modèle
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

print("Chargement des données traitées...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_test  = pd.read_csv('data/processed/X_test.csv')
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

# ── NOUVELLES FEATURES ────────────────────────────────────
# On crée des interactions entre les features les plus importantes

def add_features(df):
    # Interaction entre les top features
    df['V14_V17_interaction'] = df['V14'] * df['V17']
    df['V14_V12_ratio']       = df['V14'] / (df['V12'] + 1e-8)
    df['V10_V14_sum']         = df['V10'] + df['V14']
    df['amount_v14_product']  = df['Amount_scaled'] * df['V14']
    return df

X_train = add_features(X_train)
X_test  = add_features(X_test)

# ── SÉLECTION DES MEILLEURES FEATURES ────────────────────
# SelectKBest sélectionne les K meilleures features selon un test statistique
selector = SelectKBest(f_classif, k=25)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected  = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"✅ {len(selected_features)} features sélectionnées :")
print(list(selected_features))

# Convertir en DataFrame avec les bons noms
X_train_final = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_final  = pd.DataFrame(X_test_selected,  columns=selected_features)

# Sauvegarder
X_train_final.to_csv('data/processed/X_train_features.csv', index=False)
X_test_final.to_csv('data/processed/X_test_features.csv',  index=False)

print(f"\n✅ Features sauvegardées !")
print(f"Shape X_train : {X_train_final.shape}")
print(f"Shape X_test  : {X_test_final.shape}")
