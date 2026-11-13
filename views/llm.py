import numpy as np
import pandas as pd
import streamlit as st

import datasets as ds
import ui
from mcdm.ranking import topsis

ui.header("Scores extraits de textes par un LLM",
          "Un LLM fine-tuné lit un document non structuré et produit une ligne de la matrice de décision")
st.markdown("$f_{LLM}(T, C) \\to s = (s_1, \\dots, s_n) \\in [0, 10]^n$, puis $\\text{Décision} = \\text{MCDM}(s)$ "
            "(cours p.152-154). Le LLM fait l'extraction sémantique ; la méthode MCDM garde la décision traçable.")
st.info("Démonstration hors ligne : les scores et justifications ci-dessous sont pré-rédigés à titre d'illustration "
        "(le premier CV reprend l'exemple du cours). Aucun modèle n'est appelé.")

crit = ds.CV_CRITERIA
for cv in ds.CVS:
    with st.container(border=True):
        st.markdown(f"**{cv['nom']}**")
        st.caption(cv["texte"])
        st.dataframe(pd.DataFrame({"Score /10": cv["scores"], "Justification générée": cv["justifications"]},
                                  index=crit), width="stretch")

st.subheader("Matrice de décision obtenue et classement TOPSIS")
st.caption("L'adéquation salariale est déjà un score inversé (critère coût ramené en bénéfice) : 10 = salaire bas")
X = pd.DataFrame([cv["scores"] for cv in ds.CVS], index=[cv["nom"] for cv in ds.CVS], columns=crit, dtype=float)
X = st.data_editor(X, key="llm-x", width="stretch",
                   column_config={c: st.column_config.NumberColumn(min_value=0, max_value=10, step=1) for c in crit})
cols = st.columns(3)
w = np.array([cols[k].slider(f"Poids : {c}", 0.0, 1.0, v, 0.05) for k, (c, v) in enumerate(zip(crit, [0.5, 0.3, 0.2]))])
if w.sum() == 0 or (X.to_numpy() <= 0).all(axis=0).any():
    st.error("Il faut au moins un poids non nul et des scores non tous nuls par critère")
    st.stop()
rc = topsis(X.to_numpy(float), w / w.sum(), [1, 1, 1])["RC"]
ui.ranking_chart(X.index, rc, "TOPSIS sur les scores extraits")
ui.result(f"**{X.index[int(np.argmax(rc))]}** arrive en tête. Limites : la qualité dépend des annotations d'experts "
          "ayant servi au fine-tuning, et une rubrique de notation rigoureuse par critère est indispensable.")
