import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.ranking import topsis

ui.header("TOPSIS", "La meilleure alternative est la plus proche de la solution idéale et la plus éloignée de "
          "l'anti-idéale (Hwang et Yoon, 1981)")

with st.expander("Principe et formules"):
    st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}\qquad v_{ij}=w_j r_{ij}\qquad "
             r"I^+=\{\max_i v_{ij}\ (J),\ \min_i v_{ij}\ (J')\}\quad I^-=\{\min_i v_{ij}\ (J),\ \max_i v_{ij}\ (J')\}")
    st.latex(r"S_i^+=\sqrt{\sum_j (v_j^*-v_{ij})^2}\qquad S_i^-=\sqrt{\sum_j (v'_j-v_{ij})^2}\qquad "
             r"RC_i^*=\frac{S_i^-}{S_i^++S_i^-}\in[0,1]")


X, types, w = ui.problem_editor("topsis", {"Choix d'une voiture (cours)": ds.TOPSIS_CARS,
                                           "Projets d'investissement": ds.INVESTMENTS})
try:
    res = topsis(X.to_numpy(float), w, types)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

alts, crit = list(X.index), list(X.columns)
with st.expander("Étape 1 : normalisation vectorielle"):
    ui.table(pd.DataFrame(res["normalized"], index=alts, columns=crit))
with st.expander("Étape 2 et 3 : matrice pondérée et solutions idéales"):
    t = pd.DataFrame(res["weighted"], index=alts, columns=crit)
    t.loc["I+ (idéale)"] = res["ideal"]
    t.loc["I− (anti-idéale)"] = res["anti"]
    ui.table(t)
with st.expander("Étape 4 et 5 : séparations et coefficient de proximité", expanded=True):
    t = pd.DataFrame({"S+": res["S+"], "S−": res["S-"], "RC*": res["RC"]}, index=alts)
    t["Rang"] = ui.rank_series(alts, res["RC"])
    ui.table(t)

left, right = st.columns([3, 2], gap="large")
with left:
    ui.ranking_chart(alts, res["RC"], "Coefficient de proximité RC*")
with right:
    best_i = int(np.argmax(res["RC"]))
    fig = go.Figure(go.Scatter(x=res["S+"], y=res["S-"], mode="markers+text", text=alts, textposition="top center",
                               marker=dict(size=11, color=["#85858b"] * len(alts)), textfont=dict(color=ui.MUTED)))
    fig.add_scatter(x=[res["S+"][best_i]], y=[res["S-"][best_i]], mode="markers",
                    marker=dict(size=13, color=ui.ACCENT), hoverinfo="skip")
    fig.update_layout(title="Distance à l'idéal (S+) et à l'anti-idéal (S−)", height=80 + 36 * len(alts),
                      xaxis=dict(title="S+, à minimiser", range=ui.padded(res["S+"])),
                      yaxis=dict(title="S−, à maximiser", range=ui.padded(res["S-"], 0.3)), showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

ui.result(f"**{alts[best_i]}** est la plus proche de la solution idéale (RC* = {res['RC'][best_i]:.4f}).")
