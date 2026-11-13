import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.single import distance_to_ideal, maut, threshold_table

ui.header("Évaluation absolue d'une alternative isolée",
          "Une seule alternative : on ne la classe pas, on la juge par rapport à un seuil ou à une référence")

crit, types, lo, hi = ds.CREDIT_CRITERIA, ds.CREDIT_TYPES, ds.CREDIT_BOUNDS["min"], ds.CREDIT_BOUNDS["max"]
tab_maut, tab_roc, tab_dist = st.tabs(["Score MAUT et décision", "Choisir le seuil τ", "Distance à l'idéal"])

with tab_maut:
    st.latex(r"S(A)=\sum_j w_j\,u_j(x_j)\qquad u_j=\frac{x_j-x_j^{min}}{x_j^{max}-x_j^{min}}\ (\text{bénéfice})"
             r"\quad u_j=\frac{x_j^{max}-x_j}{x_j^{max}-x_j^{min}}\ (\text{coût})\qquad "
             r"\text{Accepter si } S(A)\ge\tau")
    st.caption("Cas d'usage : scoring de crédit. Un dossier, cinq critères, bornes et poids fixés par la banque")
    left, right = st.columns([2, 3], gap="large")
    with left:
        default = [18000, 28.0, 6.0, 1, 15.0]
        x = [st.number_input(f"{c} ({'bénéfice' if t > 0 else 'coût'})", min_value=float(a), max_value=float(b),
                             value=float(d), key=f"maut-{c}") for c, t, a, b, d in zip(crit, types, lo, hi, default)]
        tau = st.slider("Seuil de décision τ", 0.0, 1.0, 0.6, 0.01, key="tau")
    u, S = maut(x, ds.CREDIT_WEIGHTS, lo, hi, types)
    with right:
        ui.table(pd.DataFrame({"Min": lo, "Max": hi, "Valeur": x, "Utilité u_j": u, "Poids w_j": ds.CREDIT_WEIGHTS,
                               "w_j u_j": np.multiply(ds.CREDIT_WEIGHTS, u)}, index=crit), digits=3)
        fig = go.Figure(go.Bar(x=np.multiply(ds.CREDIT_WEIGHTS, u), y=crit, orientation="h", marker_color="#85858b",
                               name="Contribution"))
        fig.add_bar(x=np.subtract(ds.CREDIT_WEIGHTS, np.multiply(ds.CREDIT_WEIGHTS, u)), y=crit, orientation="h",
                    marker_color="#2a2a2d", name="Manque à gagner")
        fig.update_layout(barmode="stack", title="Contribution de chaque critère au score", height=260,
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    ui.result(f"S(A) = **{S:.3f}** pour τ = {tau:.2f} : dossier **{'accepté' if S >= tau else 'rejeté'}**.")

with tab_roc:
    st.markdown("Le seuil peut venir de la politique de l'entreprise, ou des **décisions passées** : on choisit le τ "
                "qui sépare le mieux dossiers acceptés et rejetés.")
    st.latex(r"TPR=\frac{TP}{TP+FN}\quad FPR=\frac{FP}{FP+TN}\quad d=\sqrt{FPR^2+(1-TPR)^2}\ (\min)\quad J=TPR-FPR\ (\max)")
    src = st.radio("Décisions passées", ["6 décisions du cours", "300 dossiers de crédit (synthétiques)"],
                   horizontal=True)
    if src.startswith("6"):
        df = ds.PAST_DECISIONS
        st.dataframe(df.T, width="stretch")
        s, y = df["Score S"].to_numpy(), (df["Décision"] == "Accepter").to_numpy()
        tbl = threshold_table(s, y, thresholds=[0.75, 0.65, 0.60])
        st.caption("Les trois seuils testés dans le cours")
        ui.table(pd.DataFrame(tbl, columns=["τ", "TPR", "FPR", "d", "J"], index=["Cas 1", "Cas 2", "Cas 3"]), digits=2)
        full = threshold_table(s, y)
    else:
        port = ds.credit_portfolio()
        s = np.array([maut(r, ds.CREDIT_WEIGHTS, lo, hi, types)[1] for r in port[crit].to_numpy(float)])
        y = port["Accepté"].to_numpy().astype(bool)
        st.caption(f"{len(port)} dossiers, {y.sum()} acceptés par le comité ; score S calculé par MAUT pour chacun")
        st.dataframe(port.assign(**{"Score S": s.round(3)}).head(8), width="stretch")
        full = threshold_table(s, y)
    i_d, i_j = int(np.argmin(full[:, 3])), int(np.argmax(full[:, 4]))
    curve = np.vstack([[np.inf, 0, 0, 1, 0], full, [-np.inf, 1, 1, 1, 0]])
    fig = go.Figure(go.Scatter(x=curve[:, 2], y=curve[:, 1], mode="lines+markers", line=dict(color="#a1a1a6", shape="hv"),
                               marker=dict(size=5), name="ROC", hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}"))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#5a5a5f", dash="dot"), name="Hasard")
    fig.add_scatter(x=[full[i_j, 2]], y=[full[i_j, 1]], mode="markers", marker=dict(size=13, color=ui.ACCENT),
                    name=f"Youden : τ = {full[i_j, 0]:.3f}")
    fig.add_scatter(x=[0], y=[1], mode="markers", marker=dict(size=9, color=ui.INK, symbol="x"), name="Point idéal")
    fig.update_layout(title="Courbe ROC", height=380, xaxis_title="FPR", yaxis_title="TPR",
                      xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.05]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    ui.result(f"Distance au point idéal : τ = **{full[i_d, 0]:.3f}** (d = {full[i_d, 3]:.3f}). "
              f"Indice de Youden : τ = **{full[i_j, 0]:.3f}** (J = {full[i_j, 4]:.3f}). "
              + ("Les deux critères s'accordent." if i_d == i_j
                 else "Ils divergent : on départage selon le coût d'un faux positif ou d'un faux négatif."))

with tab_dist:
    st.latex(r"D^+=\sqrt{\sum_j (u_j-u_j^+)^2}\qquad D^-=\sqrt{\sum_j (u_j-u_j^-)^2}\qquad C=\frac{D^-}{D^++D^-}")
    st.caption("Logique TOPSIS appliquée à un seul dossier : idéal = utilité 1 partout, anti-idéal = 0. "
               "Les valeurs sont celles saisies dans l'onglet Score MAUT")
    dp, dm, C = distance_to_ideal(u)
    c1, c2, c3 = st.columns(3)
    c1.metric("D+ (distance à l'idéal)", f"{dp:.3f}")
    c2.metric("D− (distance à l'anti-idéal)", f"{dm:.3f}")
    c3.metric("Indice C", f"{C:.3f}")
    ui.result(f"Indice de performance C = **{C:.3f}** : même avec une seule alternative, on obtient une mesure de "
              "sa proximité au point idéal.")
