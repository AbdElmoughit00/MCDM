import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.outranking import PREFERENCE_FUNCTIONS, preference, promethee

ui.header("PROMETHEE I et II", "Surclassement par flux de préférence (Brans et Vincke, 1985) : un mauvais score "
          "sur un critère n'est compensé que dans la limite de sa fonction de préférence")
st.caption("Les diapositives du cours ne détaillent pas PROMETHEE : formulation de référence de la littérature")

with st.expander("Principe et formules"):
    st.markdown("Pour chaque paire (a, b) et chaque critère j, l'écart orienté $d_j(a,b)$ passe par une fonction de "
                "préférence $P_j \\in [0,1]$ réglée par les seuils q (indifférence), p (préférence stricte) "
                "ou s (gaussienne).")
    st.latex(r"\pi(a,b)=\sum_j w_j P_j(a,b)\qquad \phi^+(a)=\frac{1}{m-1}\sum_b \pi(a,b)\qquad "
             r"\phi^-(a)=\frac{1}{m-1}\sum_b \pi(b,a)\qquad \phi(a)=\phi^+(a)-\phi^-(a)")
    st.markdown("- **PROMETHEE I** (rangement partiel) : a P b si a est au moins aussi bon sur φ+ et φ−, et "
                "meilleur sur l'un des deux ; si les deux flux se contredisent, a et b sont incomparables (R)\n"
                "- **PROMETHEE II** (rangement complet) : tri selon le flux net φ")

presets = {"Électrification d'un village": ds.ENERGY, "Six fournisseurs": ds.SUPPLIERS6,
           "Projets d'investissement": ds.INVESTMENTS}
X, types, w = ui.problem_editor("promethee", presets)
alts, crit = list(X.index), list(X.columns)
M = X.to_numpy(float)

st.subheader("Fonctions de préférence")
if st.session_state.get("promethee-choice") == "Électrification d'un village" and list(crit) == list(ds.ENERGY["matrix"].columns):
    cfg = pd.DataFrame(ds.ENERGY_PROMETHEE, index=crit)
else:
    span = M.max(0) - M.min(0)
    cfg = pd.DataFrame({"kinds": ["Linéaire"] * len(crit), "q": span * 0.1, "p": span * 0.5, "s": span * 0.3}, index=crit)
cfg.columns = ["Fonction", "q", "p", "s"]
cfg = st.data_editor(cfg, key=f"prom-cfg-{st.session_state.get('promethee-choice')}", width="stretch", column_config={
    "Fonction": st.column_config.SelectboxColumn(options=PREFERENCE_FUNCTIONS, required=True),
    "q": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
    "p": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
    "s": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})
if (cfg["q"] > cfg["p"]).any():
    st.error("Il faut q ≤ p pour chaque critère")
    st.stop()

j = crit.index(st.selectbox("Visualiser la fonction du critère", crit))
dmax = max(float(M[:, j].max() - M[:, j].min()), float(cfg["p"].iloc[j]) * 1.5, 1e-9)
d = np.linspace(-dmax * 0.1, dmax, 300)
fig = go.Figure(go.Scatter(x=d, y=preference(d, cfg["Fonction"].iloc[j], cfg["q"].iloc[j], cfg["p"].iloc[j],
                                             max(cfg["s"].iloc[j], 1e-9)), line=dict(color=ui.ACCENT, width=2.5)))
fig.update_layout(title=f"P(d) : {cfg['Fonction'].iloc[j]}, {crit[j]}", height=260, xaxis_title="écart orienté d",
                  yaxis=dict(range=[-0.05, 1.1]))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

res = promethee(M, w, types, list(cfg["Fonction"]), cfg["q"].to_numpy(float), cfg["p"].to_numpy(float),
                np.maximum(cfg["s"].to_numpy(float), 1e-9))

with st.expander("Étape 1 : préférences par critère P_j(a, b)"):
    for k, tab in enumerate(st.tabs(crit)):
        with tab:
            ui.table(pd.DataFrame(res["P"][k], index=alts, columns=alts), digits=3)
with st.expander("Étape 2 : indice de préférence agrégé π(a, b)"):
    ui.table(pd.DataFrame(res["pi"], index=alts, columns=alts), digits=3)
with st.expander("Étape 3 : flux φ+, φ− et φ", expanded=True):
    t = pd.DataFrame({"φ+ (sortant)": res["phi+"], "φ− (entrant)": res["phi-"], "φ net": res["phi"]}, index=alts)
    t["Rang PROMETHEE II"] = ui.rank_series(alts, res["phi"])
    ui.table(t)

left, right = st.columns(2, gap="large")
with left:
    ui.ranking_chart(alts, res["phi"], "PROMETHEE II : flux net φ")
with right:
    best_i = int(np.argmax(res["phi"]))
    fig = go.Figure(go.Scatter(x=res["phi-"], y=res["phi+"], mode="markers+text", text=alts, textposition="top center",
                               marker=dict(size=11, color=[ui.ACCENT if i == best_i else "#85858b" for i in range(len(alts))]),
                               textfont=dict(color=ui.MUTED)))
    fig.update_layout(title="PROMETHEE I : φ+ (haut) et φ− (gauche) sont meilleurs", height=80 + 36 * len(alts),
                      xaxis=dict(title="φ−", range=ui.padded(res["phi-"])[::-1]),
                      yaxis=dict(title="φ+", range=ui.padded(res["phi+"], 0.25)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.markdown("**PROMETHEE I, relation entre lignes et colonnes** : P = la ligne est préférée, P⁻¹ = la colonne "
            "est préférée, I = indifférence, R = incomparables")
st.dataframe(pd.DataFrame(res["relation"], index=alts, columns=alts), width="stretch")

incomp = sorted({tuple(sorted((alts[a], alts[b]))) for a in range(len(alts)) for b in range(len(alts))
                 if res["relation"][a, b] == "R"})
ui.result(f"PROMETHEE II classe **{alts[best_i]}** en tête (φ = {res['phi'][best_i]:.3f}). "
          + (f"PROMETHEE I signale {len(incomp)} paire(s) incomparable(s), dont "
             f"{incomp[0][0]} et {incomp[0][1]} : leurs forces et faiblesses ne se compensent pas."
             if incomp else "PROMETHEE I ne relève aucune incomparabilité : le rangement complet est fiable."))
