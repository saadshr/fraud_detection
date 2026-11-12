from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/fraud_model.pkl"
X_TEST_PATH = ROOT / "data/processed/X_test.csv"
Y_TEST_PATH = ROOT / "data/processed/y_test.csv"
MODEL_NAMES = {
    "RandomForestClassifier": "Forêt aléatoire",
    "LogisticRegression": "Régression logistique",
    "XGBClassifier": "XGBoost",
}

st.set_page_config(page_title="Fraud Detection | Démo interactive", page_icon="🛡️", layout="wide")
styles = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
st.markdown(
    """
    <section class="fraud-hero">
      <p class="fraud-kicker">SMART FRAUD DETECTION</p>
      <h1>Détecter la fraude à la carte bancaire</h1>
      <p class="fraud-intro">Explorez les résultats du modèle sur des transactions anonymisées,
      puis testez une prédiction avec un exemple déjà prêt.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_revision):
    return joblib.load(MODEL_PATH)


@st.cache_data
def evaluate(model_revision, data_revision):
    model = load_model(model_revision)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze(axis="columns")
    feature_names = list(getattr(model, "feature_names_in_", X_test.columns))
    if len(feature_names) != model.n_features_in_ or set(feature_names) != set(X_test.columns):
        raise ValueError("Les colonnes de test ne correspondent pas aux variables du modèle.")
    if len(X_test) != len(y_test) or set(y_test) != {0, 1}:
        raise ValueError("Le jeu de test doit associer chaque transaction à une classe et contenir les deux classes.")

    X_test = X_test.loc[:, feature_names]
    predicted = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    return X_test, y_test, predicted, scores


def number(value):
    return f"{value:,}".replace(",", " ")


def percent(value, digits=1):
    return f"{value:.{digits}%}".replace(".", ",").replace("%", " %")


if not all(path.is_file() for path in (MODEL_PATH, X_TEST_PATH, Y_TEST_PATH)):
    st.error("Le modèle ou les données de test sont manquants.")
    st.markdown("Depuis la racine du dépôt, suivez les étapes **Entraîner et évaluer le modèle** du README.")
    st.stop()

model_revision = MODEL_PATH.stat().st_mtime_ns
data_revision = (X_TEST_PATH.stat().st_mtime_ns, Y_TEST_PATH.stat().st_mtime_ns)
try:
    X_test, y_test, predicted, scores = evaluate(model_revision, data_revision)
except ValueError as error:
    st.error(f"{error} Régénérez les données et le modèle avec le pipeline du projet.")
    st.stop()

model = load_model(model_revision)
model_type = type(model).__name__
model_name = MODEL_NAMES.get(model_type, model_type)
n_features = model.n_features_in_
tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
fraud_count = int(tp + fn)

overview, simulation, results = st.tabs(["Vue d’ensemble", "Simulateur", "Résultats"])

with overview:
    st.header("Comment le projet fonctionne")
    st.markdown(
        "Une transaction est représentée par des variables numériques. Le modèle calcule un score "
        "de fraude et décide si la transaction doit être signalée."
    )
    st.info("Pour essayer la démo, ouvrez **Simulateur**, choisissez un exemple et cliquez sur **Analyser**.")
    source, training, decision = st.columns(3)
    with source:
        st.subheader("Données")
        st.markdown(
            "Le jeu original contient **284 807 transactions** anonymisées, dont **492 fraudes**. "
            "Les variables V1 à V28 résument des caractéristiques anonymisées."
        )
        st.markdown("[Source du jeu de données · Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/)")
    with training:
        st.subheader("Entraînement")
        st.markdown(
            "Le pipeline retire les doublons et standardise les variables. "
            "SMOTE ajoute des exemples synthétiques de fraude au jeu d’entraînement. "
            "Trois modèles sont comparés avec le score F1."
        )
    with decision:
        st.subheader("Prédiction")
        st.markdown(
            f"Le modèle chargé est **{model_name}**. Il attend **{n_features} variables prétraitées** "
            "dans un ordre précis et renvoie une classe et un score de fraude."
        )

    st.divider()
    st.subheader("Pourquoi les fausses alertes comptent")
    st.markdown(
        f"Sur **{number(len(y_test))} transactions de test**, **{number(fraud_count)}** sont des fraudes "
        f"(**{percent(fraud_count / len(y_test), 2)}**). Même quand très peu de transactions normales "
        "sont signalées, les fausses alertes peuvent représenter une grande part des alertes."
    )
    st.caption("Projet de Saad Sahraoui · ENSAM Rabat · [Code et instructions](https://github.com/saadshr/fraud_detection)")

with simulation:
    st.header("Simuler une prédiction")
    st.markdown("Choisissez un exemple du jeu de test ou saisissez une transaction déjà prétraitée.")
    choice = st.radio(
        "Transaction à analyser",
        ("Exemple normal", "Exemple de fraude", "Saisie manuelle"),
        horizontal=True,
    )
    actual_class = {"Exemple normal": 0, "Exemple de fraude": 1}.get(choice)
    if actual_class is None:
        st.caption(
            f"Collez {n_features} nombres dans l’ordre du modèle. "
            "Utilisez une virgule entre les valeurs et un point pour les décimales."
        )
    else:
        st.caption("L’exemple est prérempli à partir du jeu de test. Sa classe réelle sera affichée avec le résultat.")

    with st.form("transaction"):
        if actual_class is None:
            raw_input = st.text_area(
                f"Variables prétraitées ({n_features} valeurs)",
                placeholder="0.12, -1.43, 0.08, …",
                height=140,
            )
        submitted = st.form_submit_button("Analyser", type="primary")

    with st.expander("Quelles valeurs puis-je utiliser ?"):
        st.markdown(
            "En mode manuel, copiez une ligne de `data/processed/X_test.csv` sans son en-tête. "
            "Les variables sont déjà dans le format utilisé par le modèle. "
            "Ne saisissez ni numéro de carte, ni données personnelles."
        )
        st.code(", ".join(X_test.columns), language=None)

    if submitted:
        if actual_class is None:
            transaction = None
            try:
                values = [float(value.strip()) for value in raw_input.split(",")]
            except ValueError:
                st.error("Saisissez uniquement des nombres séparés par des virgules, avec un point pour les décimales.")
            else:
                if len(values) != n_features:
                    st.error(f"{n_features} valeurs requises, {len(values)} reçues.")
                elif not np.isfinite(values).all():
                    st.error("Toutes les valeurs doivent être des nombres finis.")
                else:
                    transaction = pd.DataFrame([values], columns=X_test.columns)
        else:
            transaction = X_test.loc[y_test == actual_class].iloc[[0]]

        if transaction is not None:
            prediction = int(model.predict(transaction)[0])
            score = float(model.predict_proba(transaction)[0][1])
            if prediction == 1:
                st.warning("Prédiction du modèle · Transaction signalée")
            else:
                st.success("Prédiction du modèle · Transaction normale")
            score_column, actual_column = st.columns(2)
            score_column.metric("Score de fraude", percent(score))
            if actual_class is not None:
                actual_column.metric("Classe réelle", "Fraude" if actual_class == 1 else "Normale")
            st.progress(score)
            st.caption("Le score reflète l’estimation du modèle. La prédiction ne constitue pas une confirmation de fraude.")

with results:
    st.header("Résultats sur le jeu de test")
    st.markdown(f"**{model_name}** · {number(len(y_test))} transactions évaluées · {number(fraud_count)} fraudes réelles")
    auc = roc_auc_score(y_test, scores)
    metrics = st.columns(4)
    metrics[0].metric("Précision", percent(precision_score(y_test, predicted, zero_division=0), 2),
                      help="Part des alertes qui correspondent à une vraie fraude.")
    metrics[1].metric("Rappel", percent(recall_score(y_test, predicted), 2),
                      help="Part des fraudes réelles que le modèle a détectées.")
    metrics[2].metric("Score F1", f"{f1_score(y_test, predicted):.4f}".replace(".", ","),
                      help="Combine la précision et le rappel.")
    metrics[3].metric("ROC-AUC", f"{auc:.4f}".replace(".", ","),
                      help="Mesure la capacité du modèle à classer les fraudes au-dessus des transactions normales.")

    st.markdown(
        f"Le modèle a détecté **{number(tp)} des {number(fraud_count)} fraudes**. "
        f"Il en a manqué **{number(fn)}** et a déclenché **{number(fp)} fausses alertes**."
    )

    matrix_column, roc_column = st.columns(2)
    with matrix_column:
        st.subheader("Matrice de confusion")
        st.markdown(
            f"""
<table class="fraud-matrix" aria-label="Matrice de confusion">
  <thead><tr><th scope="col">Classe réelle</th><th scope="col">Prédite normale</th><th scope="col">Prédite fraude</th></tr></thead>
  <tbody>
    <tr><th scope="row">Normale</th><td class="correct"><strong>{number(tn)}</strong>Classées normales</td>
        <td class="incorrect"><strong>{number(fp)}</strong>Fausses alertes</td></tr>
    <tr><th scope="row">Fraude</th><td class="incorrect"><strong>{number(fn)}</strong>Fraudes manquées</td>
        <td class="correct"><strong>{number(tp)}</strong>Fraudes détectées</td></tr>
  </tbody>
</table>
            """,
            unsafe_allow_html=True,
        )
    with roc_column:
        st.subheader("Courbe ROC")
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
        ax.plot(false_positive_rate, true_positive_rate, color="#087d78", linewidth=2.8, label="Modèle")
        ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", label="Hasard")
        ax.fill_between(false_positive_rate, true_positive_rate, alpha=0.09, color="#087d78")
        ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Taux de fausses alertes", ylabel="Part des fraudes détectées")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Chaque point correspond à un seuil d’alerte. Les métriques du haut utilisent la décision du modèle.")

    with st.expander("Comment interpréter ces scores ?"):
        st.markdown(
            "La **précision** répond à « sur toutes les alertes, combien étaient justifiées ? ». "
            "Le **rappel** répond à « sur toutes les fraudes, combien avons-nous repérées ? »."
        )
        st.markdown(
            "La ROC-AUC résume la qualité du classement pour l’ensemble des seuils. "
            "Avec une fraude rare, une ROC-AUC élevée peut coexister avec beaucoup de fausses alertes "
            "par rapport au nombre de fraudes détectées."
        )
        st.caption(
            "Le même jeu de test sert à sélectionner le meilleur modèle et à calculer ces scores. "
            "Une évaluation indépendante demanderait un jeu réservé à la mesure finale."
        )
