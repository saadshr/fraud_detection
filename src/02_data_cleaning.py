# ============================================================
# 02_data_cleaning.py
# But : nettoyer, normaliser, équilibrer le dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Chargement
    print("📂 Chargement des données...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Fichier introuvable : {raw_path}")
    df = pd.read_csv(raw_path)

    # 2. Suppression des doublons
    initial_size = len(df)
    df = df.drop_duplicates()
    print(f"✅ Doublons supprimés : {initial_size - len(df)}")

    # 3. Normalisation de 'Amount' et 'Time' (colonnes séparées)
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    df['Amount_scaled'] = amount_scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = time_scaler.fit_transform(df[['Time']])

    # Supprimer les colonnes originales
    df = df.drop(['Amount', 'Time'], axis=1)
    print("✅ Amount et Time normalisés (colonnes 'Amount_scaled' et 'Time_scaled')")

    # 4. Séparation features / target
    X = df.drop('Class', axis=1)
    y = df['Class']

    print(f"\n📊 Avant SMOTE :")
    print(f"  Normal : {(y == 0).sum():,}")
    print(f"  Fraude : {(y == 1).sum():,}")

    # 5. Train/test split (faire le split AVANT SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. SMOTE — rééquilibrage (uniquement sur le train)
    print("\n⚙️  Application de SMOTE sur le train...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n📊 Après SMOTE (train uniquement) :")
    print(f"  Normal : {(y_train_resampled == 0).sum():,}")
    print(f"  Fraude : {(y_train_resampled == 1).sum():,}")
    print(f"  Total  : {len(y_train_resampled):,}")

    # 7. Standardisation finale (fitter sur X_train_resampled puis transformer train+test)
    scaler_final = StandardScaler()
    scaler_final.fit(X_train_resampled)
    X_train_scaled = scaler_final.transform(X_train_resampled)
    X_test_scaled = scaler_final.transform(X_test)

    # Sauvegardes
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    pd.Series(y_train_resampled).to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False, header=['Class'])
    pd.Series(y_test).to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False, header=['Class'])

    # Sauvegarder le scaler utilisé en production
    joblib.dump(scaler_final, os.path.join(models_dir, 'scaler.pkl'))

    print("\n✅ Données sauvegardées dans data/processed/")
    print(f"✅ Scaler sauvegardé dans {os.path.join(models_dir, 'scaler.pkl')}")


if __name__ == '__main__':
    main()
