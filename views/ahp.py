import numpy as np
import pandas as pd
import streamlit as st

import datasets as ds
import ui
from mcdm.weighting import ahp, ahp_global

ui.header("AHP : Analytic Hierarchy Process",
          "Pondération subjective par comparaisons par paires sur l'échelle de Saaty (1 à 9), avec test de cohérence")

with st.expander("Principe et formules", expanded=False):
    st.markdown("L'expert compare chaque paire de critères : $\\alpha_{ij}$ = importance de $C_i$ par rapport à "
                "$C_j$, avec $\\alpha_{ji} = 1/\\alpha_{ij}$. Les poids sont le vecteur propre de $AW = \\lambda_{max} W$, "
                "approché ainsi :")
    st.latex(r"s_j=\sum_i \alpha_{ij}\qquad w_i=\frac{1}{n}\sum_j \frac{\alpha_{ij}}{s_j}\qquad "
             r"\lambda_{max}=\sum_j s_j w_j\qquad CI=\frac{\lambda_{max}-n}{n-1}\qquad CR=\frac{CI}{RI}")
    st.markdown("La matrice est jugée cohérente si **CR < 0,1**. On répète ensuite le calcul pour comparer les "
                "alternatives sous chaque critère (priorités locales), puis on agrège : "
                "priorité finale de $A_k$ = $\\sum_j w_j\\, w_{jk}$.")

cases = {"Achat d'une voiture (cours)": ds.AHP_CAR, "Implantation d'un hôpital": ds.AHP_HOSPITAL,
         "Hôpital, jugements incohérents": ds.AHP_HOSPITAL_INCONSISTENT}
case = cases[st.selectbox("Jeu de données", list(cases))]
st.caption(case["source"])
crit, alts = case["criteria"], case["alternatives"]


def pairwise_editor(M, labels, key):
    """Édition du triangle supérieur ; le triangle inférieur est déduit par réciprocité."""
    df = pd.DataFrame(np.asarray(M, dtype=float), index=labels, columns=labels)
    edited = st.data_editor(df, key=key, width="stretch",
                            column_config={c: st.column_config.NumberColumn(c, min_value=1 / 9, max_value=9.0,
                                                                            format="%.3f") for c in labels})
    A = edited.to_numpy(dtype=float, copy=True)
    iu = np.triu_indices(len(labels), 1)
    A[(iu[1], iu[0])] = 1 / A[iu]
    np.fill_diagonal(A, 1)
    return A


def show_ahp(res, labels, title):
    t = pd.DataFrame(res["normalized"], index=labels, columns=labels)
    t["Poids w"] = res["weights"]
    ui.table(t)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("λ max", f"{res['lambda_max']:.4f}")
    c2.metric("CI", f"{res['CI']:.4f}")
    c3.metric("RI (table du cours)", f"{res['RI']:.2f}")
    c4.metric("CR", f"{res['CR']:.4f}")
    if res["CR"] >= 0.1:
        st.warning(f"{title} : CR = {res['CR']:.3f} ≥ 0,1, jugements incohérents, à revoir avec l'expert")


st.subheader("1. Comparaison des critères")
st.caption("Modifiez le triangle supérieur ; le triangle inférieur est recalculé (α_ji = 1/α_ij)")
A = pairwise_editor(case["A"], crit, f"ahp-crit-{case['source']}")
try:
    res = ahp(A)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Étape 1 : sommes des colonnes"):
    ui.table(pd.DataFrame([res["col_sums"]], columns=crit, index=["s_j"]))
with st.expander("Étape 2 et 3 : matrice normalisée, poids et test de cohérence", expanded=True):
    show_ahp(res, crit, "Critères")
ui.ranking_chart(crit, res["weights"], "Poids des critères")

st.subheader("2. Priorités locales des alternatives")
local, crs = [], []
tabs = st.tabs(crit)
for j, tab in enumerate(tabs):
    with tab:
        L = pairwise_editor(case["local"][j], alts, f"ahp-loc-{j}-{case['source']}")
        try:
            r = ahp(L)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        show_ahp(r, alts, crit[j])
        local.append(r["weights"])
        crs.append(r["CR"])

st.subheader("3. Priorités finales")
final = ahp_global(res["weights"], local)
t = pd.DataFrame(np.column_stack(local), index=alts, columns=[f"{c} (w = {w:.3f})" for c, w in zip(crit, res["weights"])])
t["Priorité finale"] = final
t["Rang"] = ui.rank_series(alts, final)
ui.table(t)
ui.ranking_chart(alts, final, "Priorités finales des alternatives")

best = alts[int(np.argmax(final))]
coherent = res["CR"] < 0.1 and all(c < 0.1 for c in crs)
ui.result(f"**{best}** obtient la priorité la plus élevée ({final.max():.3f}). "
          + ("Toutes les matrices sont cohérentes (CR < 0,1)." if coherent
             else "Au moins une matrice est incohérente : le résultat n'est pas exploitable en l'état."))
st.caption("Analyse de sensibilité des poids : page Robustesse et comparaison")
