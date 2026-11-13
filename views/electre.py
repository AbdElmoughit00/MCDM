import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.outranking import class_rank, electre3

ui.header("ELECTRE III", "Surclassement avec seuils (Roy, 1978) : un très mauvais score sur un seul critère peut "
          "bloquer une alternative, ce qu'aucune méthode additive ne fait")
st.caption("Les diapositives du cours ne détaillent pas ELECTRE III : formulation de référence de la littérature, "
           "seuil de discrimination s(λ) = 0,3 − 0,15 λ")

with st.expander("Principe et formules"):
    st.markdown("Pour « a surclasse b » (a au moins aussi bon que b), avec $\\Delta_j$ = avance de b sur a :")
    st.latex(r"c_j(a,b)=\begin{cases}1 & \Delta_j\le q_j\\ \frac{p_j-\Delta_j}{p_j-q_j} & q_j<\Delta_j<p_j\\ 0 & \Delta_j\ge p_j\end{cases}"
             r"\qquad d_j(a,b)=\begin{cases}0 & \Delta_j\le p_j\\ \frac{\Delta_j-p_j}{v_j-p_j} & p_j<\Delta_j<v_j\\ 1 & \Delta_j\ge v_j\end{cases}")
    st.latex(r"C(a,b)=\sum_j w_j c_j(a,b)\qquad S(a,b)=C(a,b)\prod_{j:\,d_j>C}\frac{1-d_j(a,b)}{1-C(a,b)}")
    st.markdown("Les **distillations** descendante (meilleures d'abord) et ascendante (pires d'abord) donnent deux "
                "préordres ; leur intersection est le classement final, qui peut contenir des incomparabilités.")

X, types, w = ui.problem_editor("electre", {"Implantation d'une usine chimique": ds.SITES,
                                            "Électrification d'un village": ds.ENERGY, "Six fournisseurs": ds.SUPPLIERS6})
alts, crit = list(X.index), list(X.columns)
M = X.to_numpy(float)

st.subheader("Seuils")
st.caption("q : indifférence, p : préférence stricte, v : veto (q ≤ p ≤ v), dans l'unité du critère")
if st.session_state.get("electre-choice") == "Implantation d'une usine chimique" and crit == list(ds.SITES["matrix"].columns):
    th = pd.DataFrame(ds.SITES_ELECTRE, index=crit)
else:
    span = M.max(0) - M.min(0)
    th = pd.DataFrame({"q": span * 0.05, "p": span * 0.2, "v": span * 0.8}, index=crit)
th = st.data_editor(th, key=f"el-th-{st.session_state.get('electre-choice')}", width="stretch",
                    column_config={c: st.column_config.NumberColumn(min_value=0.0, format="%.2f") for c in "qpv"})
try:
    res = electre3(M, w, types, th["q"], th["p"], th["v"])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Étape 1 : concordance globale C(a, b)"):
    ui.table(pd.DataFrame(res["C"], index=alts, columns=alts), digits=3)
with st.expander("Étape 2 : discordance par critère d_j(a, b)"):
    for k, tab in enumerate(st.tabs(crit)):
        with tab:
            ui.table(pd.DataFrame(res["discordance"][k], index=alts, columns=alts), digits=3)
with st.expander("Étape 3 : crédibilité S(a, b)", expanded=True):
    fig = go.Figure(go.Heatmap(z=res["S"], x=alts, y=alts, zmin=0, zmax=1,
                               colorscale=[[0, "#19191b"], [1, ui.ACCENT]], text=np.round(res["S"], 2),
                               texttemplate="%{text}", textfont=dict(family=ui.MONO)))
    fig.update_layout(height=120 + 44 * len(alts), yaxis=dict(autorange="reversed", title="a"),
                      xaxis_title="b : degré de crédibilité de « a surclasse b »")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    vetoes = [(alts[a], alts[b], crit[j]) for j in range(len(crit)) for a in range(len(alts)) for b in range(len(alts))
              if res["discordance"][j, a, b] >= 1]
    if vetoes:
        st.caption("Vetos actifs (d_j = 1, crédibilité forcée à 0) : "
                   + " ; ".join(f"« {a} surclasse {b} » bloqué par {c}" for a, b, c in vetoes[:6])
                   + (" …" if len(vetoes) > 6 else ""))

m = len(alts)
r_desc, r_asc = class_rank(res["descending"], m), class_rank(res["ascending"], m)
with st.expander("Étape 4 : distillations", expanded=True):
    c1, c2 = st.columns(2)
    c1.markdown("**Descendante**\n\n" + "\n".join(f"{k}. {', '.join(alts[a] for a in cls)}"
                                                  for k, cls in enumerate(res["descending"], 1)))
    c2.markdown("**Ascendante**\n\n" + "\n".join(f"{k}. {', '.join(alts[a] for a in cls)}"
                                                 for k, cls in enumerate(res["ascending"], 1)))

# Intersection des deux préordres
rel = pd.DataFrame("", index=alts, columns=alts)
for a in range(m):
    for b in range(m):
        if a == b:
            continue
        da, db = r_desc[a] - r_desc[b], r_asc[a] - r_asc[b]
        rel.iloc[a, b] = ("I" if da == 0 and db == 0 else "P" if da <= 0 and db <= 0
                          else "P⁻¹" if da >= 0 and db >= 0 else "R")
t = pd.DataFrame({"Rang descendant": r_desc, "Rang ascendant": r_asc,
                  "Rang moyen": (r_desc + r_asc) / 2}, index=alts).sort_values("Rang moyen")
st.subheader("Classement final")
ui.table(t, digits=1)
st.markdown("**Relation finale** : P = la ligne surclasse la colonne, I = indifférence, R = incomparables")
st.dataframe(rel, width="stretch")

top = [alts[a] for a in range(m) if all(rel.iloc[a, b] in ("P", "I") for b in range(m) if b != a)]
incomp = (rel == "R").to_numpy().sum() // 2
ui.result((f"**{', '.join(top)}** surclasse(nt) toutes les autres alternatives dans les deux distillations. "
           if top else f"Aucune alternative ne domine les deux distillations ; la meilleure en rang moyen est "
                       f"**{t.index[0]}**. ")
          + (f"{incomp} paire(s) restent incomparables." if incomp else "Aucune incomparabilité."))
