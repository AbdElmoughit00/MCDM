import numpy as np
import pandas as pd
import streamlit as st

import datasets as ds
import ui
from mcdm.weighting import entropy, minmax

ui.header("Entropie de Shannon",
          "Pondération objective : un critère dont les valeurs sont dispersées discrimine mieux et pèse plus")

with st.expander("Principe et formules"):
    st.latex(r"p_{ij}=\frac{d_{ij}}{\sum_i d_{ij}}\qquad E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij}\qquad "
             r"w_j=\frac{1-E_j}{n-\sum_k E_k}")
    st.markdown("$E_j$ proche de 1 : valeurs uniformes, peu d'information, poids faible. "
                "$1 - E_j$ est le degré de diversification du critère.")

X, types, _ = ui.problem_editor("entropy", {"Actions, Vuković et al. 2020 (cours)": ds.FINANCE,
                                            "Machines de découpage (cours)": ds.CRITIC_MACHINES,
                                            "Projets d'investissement": ds.INVESTMENTS}, show_weights=False)
crit = list(X.columns)
raw = X.to_numpy(float)

needs_norm = (raw < 0).any()
mode = st.radio("Normalisation avant le calcul", ["Min-max orientée (bénéfice / coût)", "Aucune (valeurs brutes)"],
                index=0, horizontal=True, disabled=needs_norm)
if needs_norm:
    st.caption("La matrice contient des valeurs négatives (rendements, ROE…) : le logarithme exige p_ij ≥ 0, "
               "la normalisation min-max est donc imposée")
D = minmax(raw, types) if needs_norm or mode.startswith("Min") else raw

try:
    res = entropy(D)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Étape 1 : matrice normalisée d_ij"):
    ui.table(pd.DataFrame(D, index=X.index, columns=crit))
with st.expander("Étape 2 : proportions p_ij"):
    ui.table(pd.DataFrame(res["P"], index=X.index, columns=crit))
with st.expander("Étape 3 : entropie, diversification et poids", expanded=True):
    ui.table(pd.DataFrame({"Entropie E_j": res["E"], "Diversification 1 − E_j": res["diversification"],
                           "Poids w_j": res["weights"]}, index=crit))

ui.ranking_chart(crit, res["weights"], "Poids par entropie")
top = crit[int(np.argmax(res["weights"]))]
ui.result(f"**{top}** est le critère le plus discriminant (w = {res['weights'].max():.3f}). "
          "L'entropie ne dit rien de son importance stratégique : à confronter à l'avis d'un expert (page Pipeline hybride).")
