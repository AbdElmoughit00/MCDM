import pandas as pd
import streamlit as st

st.title("Aide à la décision multicritère")
st.caption("Chaque méthode du cours appliquée pas à pas, sur les exercices du cours et sur des cas réalistes")

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(
        "Un problème multi-attributs se pose comme une **matrice de décision** : *m* alternatives "
        "évaluées selon *n* critères, chaque critère ayant un sens (bénéfice à maximiser, coût à minimiser) "
        "et un poids. La résolution se fait en deux étapes."
    )
    st.markdown(
        "1. **Pondérer les critères** : par jugement d'experts (AHP, BWM) ou à partir des données (Entropie, CRITIC)\n"
        "2. **Classer les alternatives** : par agrégation compensatoire (WSM, WPM, WASPAS, TOPSIS, VIKOR) "
        "ou par surclassement, sans compensation (PROMETHEE, ELECTRE III)"
    )
    st.markdown(
        "Chaque page suit le même déroulé : le principe et ses formules, les données (modifiables, ou votre "
        "propre CSV), le calcul étape par étape, puis le résultat encadré."
    )
with right:
    with st.container(border=True):
        st.caption("Processus de prise de décision (cours p.13)")
        st.markdown(
            "1. Identifier le problème\n"
            "2. Élaborer les alternatives\n"
            "3. Fixer les critères et collecter les données\n"
            "4. Choisir les méthodes de pondération et de classement"
        )

st.subheader("Quelle méthode choisir")
st.dataframe(pd.DataFrame([
    ("Nature des données", "Valeurs exactes", "Méthodes crisp : AHP, TOPSIS, VIKOR…"),
    ("Nature des données", "Jugements imprécis ou linguistiques", "Méthodes floues : Fuzzy TOPSIS, TOPSIS intuitionniste"),
    ("Nombre de décideurs", "Un seul expert", "MCDM simple"),
    ("Nombre de décideurs", "Plusieurs experts", "MCDM de groupe, avec agrégation des avis (moyenne floue, IFWA)"),
    ("Compensation entre critères", "Acceptée", "Méthodes additives : WSM, TOPSIS"),
    ("Compensation entre critères", "Refusée (sécurité, environnement)", "Surclassement : ELECTRE, PROMETHEE"),
    ("Effort demandé au décideur", "Élevé", "AHP : n(n − 1)/2 comparaisons"),
    ("Effort demandé au décideur", "Réduit", "BWM : 2n − 3 comparaisons"),
    ("Effort demandé au décideur", "Aucun", "CRITIC, Entropie"),
], columns=["Critère de choix", "Situation", "Méthodes adaptées"]), hide_index=True, width="stretch")

st.markdown(
    "Aucune méthode n'est meilleure dans l'absolu. Bonne pratique du cours : appliquer plusieurs méthodes, "
    "comparer les classements et tester leur stabilité quand les poids varient (page **Robustesse et comparaison**)."
)
st.caption("Support : cours « Aide à la décision », 2026")
