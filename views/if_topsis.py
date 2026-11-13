import numpy as np
import pandas as pd
import streamlit as st

import datasets as ds
import ui
from mcdm.fuzzy import if_topsis

ui.header("TOPSIS intuitionniste", "Décision de groupe : chaque décideur exprime un degré de satisfaction μ, "
          "d'insatisfaction ν et laisse une hésitation π = 1 − μ − ν")

c = ds.IF_CLOUD
st.caption(c["source"])
with st.expander("Principe et formules (cours p.143-150)"):
    st.latex(r"\alpha_{ij}=IFWA_\sigma(\alpha^1_{ij},\dots,\alpha^l_{ij})="
             r"\Big(1-\prod_k(1-\mu^k_{ij})^{\sigma_k},\ \prod_k(\nu^k_{ij})^{\sigma_k}\Big)")
    st.latex(r"\alpha_{ij}\otimes w_j=\big(\mu_{ij}\mu_{w_j},\ \nu_{ij}+\nu_{w_j}-\nu_{ij}\nu_{w_j}\big)")
    st.latex(r"S_i^{\pm}=\sqrt{\tfrac{1}{2n}\sum_j\big[(\mu_{ij}-\mu_j^{\pm})^2+(\nu_{ij}-\nu_j^{\pm})^2+(\pi_{ij}-\pi_j^{\pm})^2\big]}"
             r"\qquad C_i=\frac{S_i^-}{S_i^++S_i^-}")
    st.markdown("I+ prend le μ maximal et le ν minimal sur un critère bénéfice (l'inverse pour un coût).")

alts, crit, dms = c["alternatives"], c["criteria"], c["decision_makers"]
types = np.array(c["types"])

st.subheader("Poids des décideurs")
cols = st.columns(len(dms))
sig = np.array([cols[k].number_input(d, 0.0, 1.0, float(c["dm_weights"][k]), 0.05, key=f"sig-{k}")
                for k, d in enumerate(dms)])
if sig.sum() == 0:
    st.error("Au moins un décideur doit avoir un poids non nul")
    st.stop()
sig = sig / sig.sum()

st.subheader("Évaluations (μ ; ν) de chaque décideur")
mats = []
for k, tab in enumerate(st.tabs(dms)):
    with tab:
        df = pd.DataFrame([[f"{m:.2f} ; {n:.2f}" for m, n in row] for row in c["matrices"][k]], index=alts, columns=crit)
        df = st.data_editor(df, key=f"ifm-{k}", width="stretch")
        mat = []
        for a in alts:
            row = []
            for cr in crit:
                try:
                    m_, n_ = (float(v) for v in str(df.loc[a, cr]).replace(",", ".").split(";"))
                except ValueError:
                    st.error(f"{dms[k]}, {a}, {cr} : format attendu « μ ; ν »")
                    st.stop()
                if min(m_, n_) < 0 or m_ + n_ > 1 + 1e-9:
                    st.error(f"{dms[k]}, {a}, {cr} : il faut μ, ν ≥ 0 et μ + ν ≤ 1")
                    st.stop()
                row.append((m_, n_))
            mat.append(row)
        mats.append(mat)

st.caption("Importance des critères selon chaque décideur (μ ; ν)")
st.dataframe(pd.DataFrame([[f"{m:.2f} ; {n:.2f}" for m, n in row] for row in c["criteria_ifn"]], index=dms,
                          columns=[f"{cr} ({'bénéfice' if t > 0 else 'coût'})" for cr, t in zip(crit, types)]),
             width="stretch")

res = if_topsis(mats, sig, c["criteria_ifn"], types)
fmt = lambda a: f"({a[0]:.3f} ; {a[1]:.3f})"
with st.expander("Étape 1 : matrice agrégée par IFWA et poids des critères"):
    t = pd.DataFrame([[fmt(res["aggregated"][i, j]) for j in range(len(crit))] for i in range(len(alts))],
                     index=alts, columns=crit)
    t.loc["Poids w_j"] = [fmt(v) for v in res["criteria_weights"]]
    st.dataframe(t, width="stretch")
with st.expander("Étape 2 : matrice pondérée et solutions idéales"):
    t = pd.DataFrame([[fmt(res["weighted"][i, j]) for j in range(len(crit))] for i in range(len(alts))],
                     index=alts, columns=crit)
    t.loc["I+"] = [fmt(v) for v in res["I+"]]
    t.loc["I−"] = [fmt(v) for v in res["I-"]]
    st.dataframe(t, width="stretch")
with st.expander("Étape 3 : séparations et coefficient de proximité", expanded=True):
    t = pd.DataFrame({"S+": res["S+"], "S−": res["S-"], "C": res["C"]}, index=alts)
    t["Rang"] = ui.rank_series(alts, res["C"])
    ui.table(t)
ui.ranking_chart(alts, res["C"], "Coefficient de proximité C")
ui.result(f"**{alts[int(np.argmax(res['C']))]}** est retenu (C = {res['C'].max():.4f}), en tenant compte du poids "
          "de chaque décideur et de son hésitation.")
