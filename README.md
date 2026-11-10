# 🛡️ Smart Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-306998?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-984400?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-993B35?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-00776D?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-B51E40?style=for-the-badge&logo=streamlit&logoColor=white)

[Installation](#installation-sur-windows) · [Entraînement](#entraîner-et-évaluer-le-modèle) · [Démo](#lancer-la-démo) · [Captures](#captures-décran)

---

Projet de détection de fraude à la carte bancaire. Il compare une régression logistique, une forêt aléatoire et XGBoost, puis expose le modèle retenu via FastAPI et un tableau de bord Streamlit.

Le jeu de données d'origine contient 284 807 transactions anonymisées, dont 492 fraudes. [Credit Card Fraud Detection sur Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/)

## Installation sur Windows

Depuis le dossier qui contient les dépôts, cloner le projet si nécessaire, puis ouvrir PowerShell à la racine du dépôt. Utiliser Python 3.11 ou 3.12.

```powershell
git clone https://github.com/saadshr/fraud_detection.git
Set-Location fraud_detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data/raw
```

Télécharger et extraire `creditcard.csv` depuis Kaggle, puis placer le fichier dans `data/raw/creditcard.csv`. Git ignore les données, les modèles et les graphiques générés.

## Entraîner et évaluer le modèle

```powershell
.\.venv\Scripts\python.exe src/02_data_cleaning.py
New-Item -ItemType Directory -Force outputs
.\.venv\Scripts\python.exe src/05_train_models.py
.\.venv\Scripts\python.exe src/06_evaluate.py
```

Le script de préparation crée les fichiers d'entraînement et de test sous `data/processed/`. Il applique SMOTE au jeu d'entraînement. Le script d'entraînement affiche la précision, le rappel, le score F1 et la ROC-AUC, puis écrit dans `models/fraud_model.pkl` le modèle qui obtient le meilleur F1 sur le jeu de test. Le script d'évaluation produit `outputs/05_evaluation.png`.

Pour explorer les données, utiliser `src/01_data_exploration.py`, `src/03_eda.py` ou le [notebook d'exploration](notebooks/01_exploration.ipynb). Le script `src/04_feature_engineering.py` génère des CSV supplémentaires. L'entraînement actuel lit `X_train.csv` et `X_test.csv`.

## Lancer la démo

Exécuter ces commandes dans deux terminaux, depuis la racine du projet après l'entraînement.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

- Swagger à l'adresse [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) et tableau de bord à l'adresse [localhost:8501](http://localhost:8501).
- `GET /health` indique si le modèle est chargé et le nombre de variables attendues. `GET /model/info` affiche les informations du modèle.
- `POST /predict` reçoit un objet JSON contenant une liste numérique `features`. Fournir une ligne déjà prétraitée selon l'ordre des colonnes de `data/processed/X_test.csv`. L'API transmet directement ces valeurs au modèle.
- Le tableau de bord lit le modèle et les CSV de test générés localement. Le formulaire de saisie manuelle attend actuellement 25 valeurs, alors que le modèle entraîné avec le pipeline ci-dessus en attend 30. Utiliser Swagger pour tester une prédiction avec ce modèle.

## Captures d’écran

[![Tableau de bord Streamlit avec les résultats du modèle](screenshots/dashboard.png)](screenshots/dashboard.png)

[Voir la capture de l’API dans Swagger](screenshots/api.png)

## Auteur

Saad Sahraoui · [GitHub](https://github.com/saadshr) · [LinkedIn](https://linkedin.com/in/saadshr)
