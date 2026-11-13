import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.ranking import waspas

ui.header("WSM, WPM et WASPAS",
          "Méthodes de pondération simple : somme pondérée, produit pondéré et leur combinaison")

with st.expander("Principe et formules"):
    st.latex(r"r_{ij}=\frac{x_{ij}}{\max_i x_{ij}}\ (\text{bénéfice})\qquad r_{ij}=\frac{\min_i x_{ij}}{x_{ij}}\ (\text{coût})")
    st.latex(r"Q_i^{(1)}=\sum_j w_j r_{ij}\ (\text{WSM})\qquad Q_i^{(2)}=\prod_j r_{ij}^{w_j}\ (\text{WPM})\qquad "
             r"Q_i=\lambda Q_i^{(1)}+(1-\lambda)Q_i^{(2)}\ (\text{WASPAS})")
    st.markdown("WASPAS (Zavadskas et al., 2012) ; λ = 0,5 donne la moyenne des deux.")

X, types, w = ui.problem_editor("waspas", {"Choix d'une machine (cours)": ds.WASPAS_MACHINES,
                                           "Projets d'investissement": ds.INVESTMENTS})
lam = st.slider("λ (poids de la somme pondérée)", 0.0, 1.0, 0.5, 0.05)
try:
    res = waspas(X.to_numpy(float), w, types, lam)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

alts = list(X.index)
with st.expander("Étape 1 : matrice normalisée"):
    ui.table(pd.DataFrame(res["normalized"], index=alts, columns=X.columns))
with st.expander("Étape 2 à 4 : WSM, WPM et critère généralisé Q", expanded=True):
    t = pd.DataFrame({"Q(1) WSM": res["WSM"], "Q(2) WPM": res["WPM"], f"Q WASPAS (λ = {lam:.2f})": res["WASPAS"]},
                     index=alts)
    t["Rang WASPAS"] = ui.rank_series(alts, res["WASPAS"])
    ui.table(t)

ui.ranking_chart(alts, res["WASPAS"], f"Classement WASPAS, λ = {lam:.2f}")

# Effet de λ : le classement est-il le même de WPM (λ = 0) à WSM (λ = 1) ?
lams = np.linspace(0, 1, 21)
curves = np.array([waspas(X.to_numpy(float), w, types, l)["WASPAS"] for l in lams])
best_i = int(np.argmax(res["WASPAS"]))
fig = go.Figure()
for i, a in enumerate(alts):
    fig.add_scatter(x=lams, y=curves[:, i], name=a, mode="lines",
                    line=dict(color=ui.ACCENT if i == best_i else "#85858b", width=2.5 if i == best_i else 1.5))
fig.add_vline(x=lam, line=dict(color="#5a5a5f", dash="dot"))
fig.update_layout(title="Score Q selon λ", height=320, xaxis_title="λ", yaxis_title="Q")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

winners = {alts[int(np.argmax(c))] for c in curves}
ui.result(f"**{alts[best_i]}** est la meilleure alternative (Q = {res['WASPAS'][best_i]:.4f}). "
          + ("Elle reste première pour tout λ entre 0 et 1." if len(winners) == 1
             else f"La première place change selon λ : {', '.join(sorted(winners))}."))
