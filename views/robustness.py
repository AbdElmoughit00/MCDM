import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm import robustness as rb

ui.header("Robustesse et comparaison des méthodes",
          "Les classements de plusieurs méthodes concordent-ils, et résistent-ils à une variation des poids")

with st.expander("Corrélation de Spearman"):
    st.latex(r"\rho=1-\frac{6\sum_i d_i^2}{n(n^2-1)}\qquad d_i=\text{rang}_1(A_i)-\text{rang}_2(A_i)")
    st.markdown("ρ = 1 : classements identiques ; ρ = 0 : aucun lien ; ρ = −1 : classements inversés.")


def spearman_heatmap(names, M, title):
    fig = go.Figure(go.Heatmap(z=M, x=names, y=names, zmin=-1, zmax=1,
                               colorscale=[[0, "#e5484d"], [0.5, "#19191b"], [1, ui.ACCENT]],
                               text=np.round(M, 2), texttemplate="%{text}", textfont=dict(family=ui.MONO)))
    fig.update_layout(title=title, height=120 + 48 * len(names), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


tab_course, tab_compare, tab_sens = st.tabs(["Exemple du cours", "Comparer les méthodes", "Sensibilité à un poids"])

with tab_course:
    st.caption("Rangs de A1 à A6 selon trois méthodes (1 = la meilleure)")
    r = pd.DataFrame(ds.COURSE_RANKS, index=[f"A{i}" for i in range(1, 7)])
    st.dataframe(r, width="stretch")
    names, M = rb.spearman_matrix(ds.COURSE_RANKS)
    spearman_heatmap(names, M, "Matrice de Spearman")
    ui.result(f"TOPSIS et PROMETHEE II concordent (ρ = {M[0, 2]:.2f}) ; VIKOR s'écarte (ρ = {M[0, 1]:.2f} et "
              f"{M[1, 2]:.2f}). Avec n = 6, seul ρ ≥ 0,886 est significatif à 5 % : le choix de A1 et A2 est robuste, "
              "le rang de A3 à A6 dépend de la méthode.")

with tab_compare:
    X, types, w = ui.problem_editor("rob", {"Six fournisseurs": ds.SUPPLIERS6, "Projets d'investissement": ds.INVESTMENTS})
    M_ = X.to_numpy(float)
    alts = list(X.index)
    try:
        table = {m: rb.ranks(rb.scores(m, M_, w, types)) for m in rb.METHODS}
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.dataframe(pd.DataFrame(table, index=alts), width="stretch")
    names, M = rb.spearman_matrix(table)
    spearman_heatmap(names, M, "Accord entre méthodes (ρ de Spearman)")
    firsts = {alts[int(np.argmin(r))] for r in table.values()}
    off = M[np.triu_indices(len(names), 1)]
    signif = (f" Avec n = 6, seul ρ ≥ 0,886 est significatif à 5 % : {(off >= 0.886).sum()} paire(s) sur {len(off)}."
              if len(alts) == 6 else "")
    ui.result((f"Toutes les méthodes placent **{firsts.pop()}** en tête. " if len(firsts) == 1
               else f"La première place dépend de la méthode : {', '.join(sorted(firsts))}. ")
              + f"ρ minimal = {off.min():.2f}." + signif)

with tab_sens:
    X, types, w = ui.problem_editor("sens", {"Projets d'investissement": ds.INVESTMENTS, "Six fournisseurs": ds.SUPPLIERS6})
    alts, crit = list(X.index), list(X.columns)
    c1, c2 = st.columns(2)
    method = c1.selectbox("Méthode", rb.METHODS, index=2, key="sens-method")
    j = crit.index(c2.selectbox("Critère dont on fait varier le poids", crit, key="sens-crit"))
    grid = np.linspace(0, 1, 41)
    R = np.array([rb.ranks(rb.scores(method, X.to_numpy(float), rb.vary_weight(w, j, g), types)) for g in grid])
    best0 = int(np.argmin(rb.ranks(rb.scores(method, X.to_numpy(float), w, types))))
    fig = go.Figure()
    for i, a in enumerate(alts):
        fig.add_scatter(x=grid, y=R[:, i], name=a, mode="lines", line_shape="hv",
                        line=dict(color=ui.ACCENT if i == best0 else "#85858b", width=2.5 if i == best0 else 1.5))
    fig.add_vline(x=w[j], line=dict(color="#5a5a5f", dash="dot"), annotation_text="poids actuel",
                  annotation_font_color=ui.MUTED)
    fig.update_layout(title=f"Rang de chaque alternative quand w({crit[j]}) varie (les autres au prorata)", height=360,
                      xaxis_title=f"w({crit[j]})", yaxis=dict(title="Rang", autorange="reversed", dtick=1))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    stable = grid[R[:, best0] == 1]
    ui.result(f"**{alts[best0]}** reste première tant que w({crit[j]}) ∈ [{stable.min():.2f} ; {stable.max():.2f}] "
              f"(pas de 0,025), pour un poids actuel de {w[j]:.3f}." if len(stable)
              else f"**{alts[best0]}** perd la première place dès que le poids s'écarte de sa valeur actuelle.")
