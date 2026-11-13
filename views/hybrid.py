import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm import robustness as rb
from mcdm.weighting import critic, entropy, minmax

ui.header("Pipeline hybride", "Une méthode de pondération, puis une méthode de classement ; poids hybrides entre "
          "jugement d'expert et données")
st.markdown("Le cours (p.22) recommande de combiner poids subjectifs et objectifs pour équilibrer expertise et "
            "données. Ici, $w = \\beta\\, w_{\\text{expert}} + (1-\\beta)\\, w_{\\text{objectif}}$, puis le "
            "classement suit, comme dans les combinaisons CRITIC-TOPSIS ou Entropie-VIKOR de la littérature.")

X, types, w_expert = ui.problem_editor("hyb", {"Projets d'investissement": ds.INVESTMENTS,
                                               "Six fournisseurs": ds.SUPPLIERS6,
                                               "Électrification d'un village": ds.ENERGY})
st.caption("La colonne Poids ci-dessus joue le rôle des poids d'expert (issus d'AHP ou de BWM par exemple)")
alts, crit = list(X.index), list(X.columns)
M = X.to_numpy(float)

c1, c2, c3 = st.columns(3)
obj = c1.selectbox("Pondération objective", ["CRITIC", "Entropie"])
beta = c2.slider("β : part du jugement d'expert", 0.0, 1.0, 0.5, 0.05)
method = c3.selectbox("Méthode de classement", rb.METHODS, index=2)

try:
    w_obj = critic(M, types)["weights"] if obj == "CRITIC" else entropy(minmax(M, types))["weights"]
except ValueError as exc:
    st.error(str(exc))
    st.stop()
w = beta * w_expert + (1 - beta) * w_obj

fig = go.Figure()
fig.add_bar(x=crit, y=w_expert, name="Expert", marker_color="#5a5a5f")
fig.add_bar(x=crit, y=w_obj, name=obj, marker_color="#a1a1a6")
fig.add_bar(x=crit, y=w, name=f"Hybride (β = {beta:.2f})", marker_color=ui.ACCENT)
fig.update_layout(barmode="group", title="Poids des critères", height=320, legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

s = rb.scores(method, M, w, types)
t = pd.DataFrame({label: rb.ranks(rb.scores(method, M, w_, types)) for label, w_ in
                  [("Rang, poids expert", w_expert), (f"Rang, poids {obj}", w_obj), ("Rang, poids hybrides", w)]},
                 index=alts)
ui.ranking_chart(alts, s, f"{obj} + {method}, score orienté (plus grand = meilleur)")
st.dataframe(t, width="stretch")

best = alts[int(np.argmax(s))]
changed = (t.iloc[:, 0] != t.iloc[:, 2]).sum()
ui.result(f"Avec les poids hybrides, **{best}** arrive en tête selon {method}. "
          f"{changed} alternative(s) changent de rang par rapport aux seuls poids d'expert.")
