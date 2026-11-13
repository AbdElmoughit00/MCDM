import numpy as np
import pandas as pd
import streamlit as st

import datasets as ds
import ui
from mcdm.weighting import bwm

ui.header("BWM : Best-Worst Method",
          "Pondération subjective : chaque critère n'est comparé qu'au meilleur et au pire (2n − 3 comparaisons)")

with st.expander("Principe et formules"):
    st.markdown("L'expert choisit le critère le plus important **B** et le moins important **W**, puis note de 1 à 9 "
                "le vecteur Best-to-Others $a_{Bj}$ et le vecteur Others-to-Worst $a_{jW}$. On cherche les poids "
                "qui respectent au mieux $w_B / w_j = a_{Bj}$ et $w_j / w_W = a_{jW}$ (forme linéaire du cours) :")
    st.latex(r"\min \xi \quad\text{s.c.}\quad |w_B - a_{Bj} w_j| \le \xi,\quad |w_j - a_{jW} w_W| \le \xi,"
             r"\quad \textstyle\sum_j w_j = 1,\; w_j \ge 0")
    st.markdown("ξ* proche de 0 signifie des jugements cohérents.")

cases = {"Choix d'un fournisseur (cours)": ds.BWM_SUPPLIER, "Plateforme Fintech": ds.BWM_FINTECH}
case = cases[st.selectbox("Jeu de données", list(cases))]
st.caption(case["source"])
crit = case["criteria"]
n = len(crit)

c1, c2 = st.columns(2)
best = crit.index(c1.selectbox("Critère le plus important (Best)", crit, index=case["best"]))
worst = crit.index(c2.selectbox("Critère le moins important (Worst)", crit, index=case["worst"]))
if best == worst:
    st.error("Best et Worst doivent être deux critères différents")
    st.stop()

vec = pd.DataFrame({"Best → autres (a_Bj)": case["BO"], "Autres → Worst (a_jW)": case["OW"]}, index=crit)
scale = st.column_config.NumberColumn(min_value=1, max_value=9, step=1)
vec = st.data_editor(vec, key=f"bwm-{case['source']}", width="stretch",
                     column_config={c: scale for c in vec.columns})
a_bo, a_ow = vec.iloc[:, 0].to_numpy(float, copy=True), vec.iloc[:, 1].to_numpy(float, copy=True)
a_bo[best], a_ow[worst] = 1, 1
if a_bo.argmax() != worst:
    st.warning("Contrôle de bon sens : la note la plus forte du vecteur BO devrait porter sur le critère Worst")

try:
    res = bwm(best, worst, a_bo, a_ow)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Étape 1 : forme matricielle A x ≤ b du programme linéaire"):
    st.caption(f"Variables x = (w1 … w{n}, ξ), objectif min ξ ; chaque ligne est une contrainte ≤ 0, "
               "plus l'égalité Σ w = 1")
    ui.table(pd.DataFrame(res["A_ub"], columns=[f"w{j + 1}" for j in range(n)] + ["ξ"]), digits=2)

with st.expander("Étape 2 : solution et écarts aux jugements", expanded=True):
    w = res["weights"]
    t = pd.DataFrame({"Poids w_j": w,
                      "|w_B − a_Bj w_j|": np.abs(w[best] - a_bo * w),
                      "|w_j − a_jW w_W|": np.abs(w - a_ow * w[worst])}, index=crit)
    ui.table(t)
    st.metric("ξ* (indicateur de cohérence)", f"{res['xi']:.4f}")

ui.ranking_chart(crit, res["weights"], "Poids optimaux des critères")
ui.result(f"**{crit[best]}** pèse {res['weights'][best]:.3f} ; ξ* = {res['xi']:.4f}"
          + (", les jugements sont cohérents." if res["xi"] < 0.1 else ", les jugements manquent de cohérence."))

if case is ds.BWM_SUPPLIER:
    st.info("Le cours (p.42) affiche les poids 0,552 / 0,207 / 0,155 / 0,086 avec ξ* = 0,08. Le ξ* concorde, mais "
            "ces poids violent les contraintes de son propre modèle : leur plus grand écart vaut 0,223, au-dessus de "
            "ξ*. Les poids ci-dessus sont l'optimum exact du programme linéaire des p.40-43.")
