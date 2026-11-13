import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.weighting import critic

ui.header("CRITIC : CRiteria Importance Through Intercriteria Correlation",
          "Pondération objective : un critère pèse plus s'il est dispersé et peu corrélé aux autres")

with st.expander("Principe et formules"):
    st.latex(r"r_{ij}=\frac{x_{ij}-\min x_j}{\max x_j-\min x_j}\ (\text{bénéfice})\qquad "
             r"r_{ij}=\frac{\max x_j-x_{ij}}{\max x_j-\min x_j}\ (\text{coût})")
    st.latex(r"\rho_{jk}=\text{corr}(r_j, r_k)\qquad \sigma_j=\sqrt{\tfrac{1}{m-1}\textstyle\sum_i(r_{ij}-\bar r_j)^2}"
             r"\qquad C_j=\sigma_j\sum_k(1-|\rho_{jk}|)\qquad w_j=\frac{C_j}{\sum_k C_k}")

X, types, _ = ui.problem_editor("critic", {"Machines de découpage (cours)": ds.CRITIC_MACHINES,
                                           "Actions, Vuković et al. 2020 (cours)": ds.FINANCE,
                                           "Projets d'investissement": ds.INVESTMENTS}, show_weights=False)
crit = list(X.columns)
try:
    res = critic(X.to_numpy(float), types)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Étape 1 : matrice normalisée r_ij"):
    ui.table(pd.DataFrame(res["normalized"], index=X.index, columns=crit))
with st.expander("Étape 2 : corrélations entre critères", expanded=True):
    fig = go.Figure(go.Heatmap(z=res["corr"], x=crit, y=crit, zmin=-1, zmax=1,
                               colorscale=[[0, "#e5484d"], [0.5, "#19191b"], [1, ui.ACCENT]],
                               text=np.round(res["corr"], 2), texttemplate="%{text}",
                               textfont=dict(family=ui.MONO)))
    fig.update_layout(height=120 + 40 * len(crit), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with st.expander("Étape 3 et 4 : écart type, indice C et poids", expanded=True):
    ui.table(pd.DataFrame({"σ_j": res["sigma"], "Σ (1 − |ρ_jk|)": res["independence"],
                           "C_j": res["C"], "Poids w_j": res["weights"]}, index=crit))

ui.ranking_chart(crit, res["weights"], "Poids CRITIC")
top = crit[int(np.argmax(res["weights"]))]
ui.result(f"**{top}** reçoit le poids le plus élevé ({res['weights'].max():.3f}) : il combine dispersion et "
          "indépendance vis-à-vis des autres critères.")
