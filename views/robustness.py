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


tab_course, tab_compare, tab_sens, tab_mc = st.tabs(
    ["Exemple du cours", "Comparer les méthodes", "Sensibilité à un poids", "Monte-Carlo sur les poids"])

with tab_course:
    st.caption("Cours p.78-79 : rangs de A1 à A6 selon trois méthodes (1 = la meilleure)")
    r = pd.DataFrame(ds.COURSE_RANKS, index=[f"A{i}" for i in range(1, 7)])
    st.dataframe(r, width="stretch")
    names, M = rb.spearman_matrix(ds.COURSE_RANKS)
    spearman_heatmap(names, M, "Matrice de Spearman")
    ui.result(f"TOPSIS et PROMETHEE II concordent (ρ = {M[0, 2]:.2f}) ; VIKOR s'écarte (ρ = {M[0, 1]:.2f} et "
              f"{M[1, 2]:.2f}). Avec n = 6, seul ρ ≥ 0,886 est significatif à 5 % : le choix de A1 et A2 est robuste, "
              "le rang de A3 à A6 dépend de la méthode.")

with tab_compare:
    X, types, w = ui.problem_editor("rob", {"Six fournisseurs": ds.SUPPLIERS6, "Projets d'investissement": ds.INVESTMENTS,
                                            "Électrification d'un village": ds.ENERGY})
    M_ = X.to_numpy(float)
    alts = list(X.index)
    try:
        table = {m: rb.ranks(rb.scores(m, M_, w, types)) for m in rb.METHODS}
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption("PROMETHEE II utilise ici une fonction linéaire (q = 10 %, p = 50 % de l'étendue de chaque critère)")
    st.dataframe(pd.DataFrame(table, index=alts), width="stretch")
    names, M = rb.spearman_matrix(table)
    spearman_heatmap(names, M, "Accord entre méthodes (ρ de Spearman)")
    crit_r = rb.critical_rho(len(alts))
    firsts = {alts[int(np.argmin(r))] for r in table.values()}
    off = M[np.triu_indices(len(names), 1)]
    ui.result((f"Toutes les méthodes placent **{firsts.pop()}** en tête. " if len(firsts) == 1
               else f"La première place dépend de la méthode : {', '.join(sorted(firsts))}. ")
              + f"ρ minimal = {off.min():.2f} ; seuil de significativité à 5 % pour n = {len(alts)} : ρ ≥ {crit_r:.3f} "
                f"({(off >= crit_r).sum()} paire(s) sur {len(off)} au-dessus).")

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

with tab_mc:
    X, types, w = ui.problem_editor("mc", {"Six fournisseurs": ds.SUPPLIERS6, "Projets d'investissement": ds.INVESTMENTS})
    alts = list(X.index)
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Méthode", rb.METHODS, index=2, key="mc-method")
    conc = c2.select_slider("Dispersion des poids", options=[200, 50, 20, 5],
                            format_func={200: "faible", 50: "modérée", 20: "forte", 5: "très forte"}.get, value=50)
    runs = c3.select_slider("Tirages", options=[200, 500, 1000, 2000], value=500)
    freq, samples = rb.monte_carlo(lambda ws: rb.scores(method, X.to_numpy(float), ws, types), w, runs, conc)
    fig = go.Figure()
    shades = [ui.ACCENT, "#c4c4c8", "#a1a1a6", "#85858b", "#5a5a5f", "#3a3a3d"]
    for k in range(len(alts)):
        fig.add_bar(y=alts, x=freq[:, k] * 100, orientation="h", name=f"Rang {k + 1}",
                    marker_color=shades[min(k, len(shades) - 1)])
    fig.update_layout(barmode="stack", title=f"Fréquence des rangs sur {runs} jeux de poids tirés autour des poids actuels",
                      height=100 + 40 * len(alts), xaxis_title="% des tirages", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    i = int(np.argmax(freq[:, 0]))
    ui.result(f"**{alts[i]}** est première dans {freq[i, 0]:.0%} des tirages (Dirichlet centrée sur les poids, "
              f"écart type moyen des poids ≈ {samples.std(axis=0).mean():.3f}).")
