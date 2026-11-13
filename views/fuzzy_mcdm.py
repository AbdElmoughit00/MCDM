import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.fuzzy import RATING_SCALE, WEIGHT_SCALE, fuzzy_single, fuzzy_topsis

ui.header("MCDM floue et Fuzzy TOPSIS", "Évaluations linguistiques converties en nombres flous triangulaires "
          "(a pessimiste, b le plus probable, c optimiste)")


def tfn(v):
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


tab1, tab2 = st.tabs(["Une seule alternative", "Fuzzy TOPSIS (Chen, 2000)"])

with tab1:
    f = ds.FUZZY_SINGLE
    st.caption(f["source"])
    st.markdown("1. Normaliser : $(a, b, c) / c_{max}$  \n2. Agréger : $\\tilde S = \\sum_j w_j \\tilde x_j$  \n"
                "3. Défuzzifier : $S = (a + b + c) / 3$ (centre de gravité)")
    df = pd.DataFrame(f["tfn"], index=f["criteria"], columns=["a", "b", "c"])
    df["Poids"] = f["weights"]
    df["c max"] = f["c_max"]
    df = st.data_editor(df, key="fz-single", width="stretch")
    if (np.diff(df[["a", "b", "c"]].to_numpy(), axis=1) < 0).any() or (df["c max"] <= 0).any():
        st.error("Il faut a ≤ b ≤ c et c max > 0")
        st.stop()
    w = df["Poids"].to_numpy(float)
    T, agg, S = fuzzy_single(df[["a", "b", "c"]].to_numpy(float), w / w.sum(), df["c max"].to_numpy(float))
    t = pd.DataFrame(T, index=f["criteria"], columns=["a / c max", "b / c max", "c / c max"])
    ui.table(t, digits=3)
    fig = go.Figure()
    for name, v, color in [(c, T[i], "#85858b") for i, c in enumerate(f["criteria"])] + [("S̃ agrégé", agg, ui.ACCENT)]:
        fig.add_scatter(x=list(v), y=[0, 1, 0], name=name, line=dict(color=color, width=2.5 if color == ui.ACCENT else 1.5))
    fig.add_vline(x=S, line=dict(color=ui.ACCENT, dash="dot"))
    fig.update_layout(height=280, yaxis=dict(range=[-0.05, 1.1], title="μ"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    ui.result(f"S̃ = **{tfn(agg)}**, score défuzzifié S = **{S:.2f}**. Largeur c − a = {agg[2] - agg[0]:.2f} : "
              "l'incertitude, à surveiller en plus du score.")

with tab2:
    h = ds.FUZZY_TOPSIS_HIRING
    st.caption(h["source"])
    with st.expander("Formules et échelle linguistique (tableau 1 du cours)"):
        st.latex(r"\tilde x_{ij}=\tfrac1K(\tilde x_{ij1}\oplus\dots\oplus\tilde x_{ijK})\qquad "
                 r"\tilde r_{ij}=\Big(\tfrac{a_{ij}}{c_j^*},\tfrac{b_{ij}}{c_j^*},\tfrac{c_{ij}}{c_j^*}\Big)\ \text{(bénéfice)}"
                 r"\qquad \tilde v_{ij}=\tilde r_{ij}\otimes\tilde w_j")
        st.latex(r"d(\tilde x,\tilde y)=\sqrt{\tfrac13\sum_k(x_k-y_k)^2}\qquad D_i^*=\sum_j d(\tilde v_{ij},(1,1,1))\qquad "
                 r"D_i^-=\sum_j d(\tilde v_{ij},(0,0,0))\qquad CC_i=\frac{D_i^-}{D_i^*+D_i^-}")
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame({"Nombre flou": [str(v) for v in RATING_SCALE.values()]}, index=RATING_SCALE),
                     width="stretch")
        c2.dataframe(pd.DataFrame({"Nombre flou": [str(v) for v in WEIGHT_SCALE.values()]}, index=WEIGHT_SCALE),
                     width="stretch")

    crit, alts, dms = h["criteria"], h["alternatives"], h["decision_makers"]
    st.markdown("**Tableau 2 : évaluations des candidats par les décideurs**")
    idx = pd.MultiIndex.from_product([alts, dms], names=["Candidat", "Décideur"])
    ev = pd.DataFrame([row for alt in h["ratings"] for row in alt], index=idx, columns=crit).reset_index()
    sel = {c: st.column_config.SelectboxColumn(options=list(RATING_SCALE), required=True) for c in crit}
    ev = st.data_editor(ev, key="fz-ev", width="stretch", hide_index=True, column_config=sel,
                        disabled=["Candidat", "Décideur"])
    st.markdown("**Tableau 3 : importance des critères selon chaque décideur**")
    wt = pd.DataFrame(h["weights"], index=dms, columns=crit)
    wt = st.data_editor(wt, key="fz-w", width="stretch",
                        column_config={c: st.column_config.SelectboxColumn(options=list(WEIGHT_SCALE), required=True)
                                       for c in crit})

    R = np.array([[[RATING_SCALE[ev.loc[(ev["Candidat"] == a) & (ev["Décideur"] == d), c].item()] for d in dms]
                   for c in crit] for a in alts], dtype=float)  # (m, n, K, 3)
    W = np.array([[WEIGHT_SCALE[wt.loc[d, c]] for d in dms] for c in crit], dtype=float)  # (n, K, 3)
    res = fuzzy_topsis(R, W, h["types"])

    with st.expander("Étape 1 : matrice de décision collective et poids collectifs"):
        t = pd.DataFrame([[tfn(res["aggregated"][i, j]) for j in range(len(crit))] for i in range(len(alts))],
                         index=alts, columns=crit)
        t.loc["Poids w̃_j"] = [tfn(v) for v in res["weights"]]
        st.dataframe(t, width="stretch")
    with st.expander("Étape 2 : matrice normalisée"):
        st.dataframe(pd.DataFrame([[tfn(res["normalized"][i, j]) for j in range(len(crit))] for i in range(len(alts))],
                                  index=alts, columns=crit), width="stretch")
    with st.expander("Étape 3 : matrice normalisée pondérée"):
        st.dataframe(pd.DataFrame([[tfn(res["weighted"][i, j]) for j in range(len(crit))] for i in range(len(alts))],
                                  index=alts, columns=crit), width="stretch")
    with st.expander("Étape 4 : distances et coefficient de proximité", expanded=True):
        t = pd.DataFrame({"D*": res["D+"], "D−": res["D-"], "CC": res["CC"]}, index=alts)
        t["Rang"] = ui.rank_series(alts, res["CC"])
        ui.table(t)
    ui.ranking_chart(alts, res["CC"], "Coefficient de proximité CC")
    best = alts[int(np.argmax(res["CC"]))]
    ui.result(f"Le candidat **{best}** est recommandé (CC = {res['CC'].max():.4f}) ; ordre : "
              + " > ".join(alts[i] for i in np.argsort(-res["CC"])) + ".")
