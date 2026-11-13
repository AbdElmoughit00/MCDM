import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.ranking import vikor

ui.header("VIKOR", "Classement de compromis : maximiser l'utilité de groupe (S) et minimiser le regret individuel (R)")

with st.expander("Principe et formules"):
    st.latex(r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\qquad R_i=\max_j\Big[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\Big]")
    st.latex(r"Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}")
    st.markdown("Plus Q est petit, meilleure est l'alternative. A′ (plus petit Q) est la solution de compromis si :\n"
                "- **C1, avantage acceptable** : $Q(A'') - Q(A') \\ge 1/(m-1)$\n"
                "- **C2, stabilité** : A′ est aussi la meilleure selon S et selon R\n\n"
                "Sinon : {A′, A″} si seule C2 échoue ; {A′, …, A_k} avec $Q(A_k) - Q(A') < 1/(m-1)$ si C1 échoue.")

X, types, w = ui.problem_editor("vikor", {"Choix d'une machine (cours)": ds.VIKOR_MACHINES,
                                          "Six fournisseurs": ds.SUPPLIERS6,
                                          "Projets d'investissement": ds.INVESTMENTS})
v = st.slider("v : poids de la stratégie d'utilité de groupe", 0.0, 1.0, 0.5, 0.05)
try:
    res = vikor(X.to_numpy(float), w, types, v)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

alts, crit = list(X.index), list(X.columns)
with st.expander("Étape 1 : meilleures et pires valeurs"):
    ui.table(pd.DataFrame([res["best"], res["worst"]], index=["x* (meilleure)", "x− (pire)"], columns=crit))
with st.expander(r"Étape 2 : termes w\_j (x\*\_j − x\_ij) / (x\*\_j − x−\_j)"):
    ui.table(pd.DataFrame(res["terms"], index=alts, columns=crit))
with st.expander("Étape 3 et 4 : S, R, Q et les trois classements", expanded=True):
    t = pd.DataFrame({"S": res["S"], "R": res["R"], "Q": res["Q"]}, index=alts)
    for k in ("S", "R", "Q"):
        t[f"Rang {k}"] = ui.rank_series(alts, res[k], higher_is_better=False)
    ui.table(t)

ui.ranking_chart(alts, res["Q"], f"Indice Q (à minimiser), v = {v:.2f}", higher_is_better=False)

fig = go.Figure(go.Scatter(x=res["S"], y=res["R"], mode="markers+text", text=alts, textposition="top center",
                           marker=dict(size=11, color=[ui.ACCENT if i in res["compromise"] else "#85858b"
                                                       for i in range(len(alts))]),
                           textfont=dict(color=ui.MUTED)))
fig.update_layout(title="Plan utilité de groupe S / regret individuel R (en bas à gauche = meilleur)", height=340,
                  xaxis=dict(title="S", range=ui.padded(res["S"])), yaxis=dict(title="R", range=ui.padded(res["R"], 0.25)))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

order = np.argsort(res["Q"], kind="stable")
gap = res["Q"][order[1]] - res["Q"][order[0]]
mark = lambda ok: ":green[vérifiée]" if ok else ":red[non vérifiée]"
st.markdown(f"- **C1, avantage acceptable** : {mark(res['C1'])}, Q(A″) − Q(A′) = {gap:.3f} pour un seuil "
            f"DQ = {res['DQ']:.3f}\n- **C2, stabilité** (A′ meilleure en S et en R) : {mark(res['C2'])}")

names = [alts[i] for i in res["compromise"]]
if len(names) == 1:
    ui.result(f"**{names[0]}** est la solution de compromis : les deux conditions sont vérifiées.")
else:
    ui.result(f"Pas de gagnant unique : l'ensemble de compromis est **{', '.join(names)}**. "
              "C'est la base d'une négociation entre décideurs.")
