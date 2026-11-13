import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datasets as ds
import ui
from mcdm.ranking import topsis
from mcdm.single import holt, maut, shap_weight_update

ui.header("MCDM et apprentissage automatique",
          "Trois scénarios du cours : classer par ML, réviser les poids par SHAP, décider sur des critères prévus")

crit, types, lo, hi = ds.CREDIT_CRITERIA, ds.CREDIT_TYPES, ds.CREDIT_BOUNDS["min"], ds.CREDIT_BOUNDS["max"]
port = ds.credit_portfolio()
Xc = port[crit].to_numpy(float)
S = np.array([maut(r, ds.CREDIT_WEIGHTS, lo, hi, types)[1] for r in Xc])

tab1, tab2, tab3 = st.tabs(["Scénario 1 : MCDM vers classification", "Scénario 2 : SHAP pour pondérer",
                            "MCDM dynamique"])

with tab1:
    st.markdown("1. Pondérer les critères (ici poids fixés par la banque)  \n2. Calculer le score MAUT $S_i$ de "
                "chaque dossier  \n3. Étiqueter A, B, C, D selon $S_i$  \n4. Entraîner un classifieur sur les critères  \n"
                "5. Prédire la classe d'un nouveau dossier sans recalculer le MCDM")
    cuts = np.quantile(S, [0.25, 0.5, 0.75])
    st.caption(f"Classes par quartiles du score : D < {cuts[0]:.3f} ≤ C < {cuts[1]:.3f} ≤ B < {cuts[2]:.3f} ≤ A")
    labels = np.select([S >= cuts[2], S >= cuts[1], S >= cuts[0]], ["A", "B", "C"], "D")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(Xc, labels, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    classes = sorted(set(labels))
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.dataframe(pd.Series(labels).value_counts().sort_index().rename("Dossiers").to_frame(), width="stretch")
        st.metric("Exactitude sur 30 % de test", f"{accuracy_score(yte, pred):.1%}")
    with c2:
        cm = confusion_matrix(yte, pred, labels=classes)
        fig = go.Figure(go.Heatmap(z=cm, x=classes, y=classes, colorscale=[[0, "#19191b"], [1, ui.ACCENT]],
                                   text=cm, texttemplate="%{text}", textfont=dict(family=ui.MONO), showscale=False))
        fig.update_layout(title="Matrice de confusion", height=300, xaxis_title="Classe prédite",
                          yaxis=dict(title="Classe MCDM", autorange="reversed"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("**Nouveau dossier**")
    cols = st.columns(len(crit))
    new = [cols[k].number_input(c, float(lo[k]), float(hi[k]), float(np.median(Xc[:, k])), key=f"new-{k}")
           for k, c in enumerate(crit)]
    cls = clf.predict([new])[0]
    true_s = maut(new, ds.CREDIT_WEIGHTS, lo, hi, types)[1]
    ui.result(f"Le classifieur attribue la classe **{cls}** ; le calcul MCDM direct donne S = {true_s:.3f}.")

with tab2:
    st.markdown("Un modèle $\\hat S = f(x)$ est entraîné, puis SHAP décompose chaque prédiction : "
                "$\\hat S = \\phi_0 + \\sum_j \\phi_j$. L'importance globale $|\\phi_j|$ sert à réviser les poids :")
    st.latex(r"w_j^{new}=\alpha\,w_j^{MCDM}+(1-\alpha)\frac{\overline{|\phi_j|}}{\sum_k\overline{|\phi_k|}}")
    target = st.radio("Variable apprise", ["Score MCDM S (cours p.92)", "Décisions réelles du comité"], horizontal=True)
    alpha = st.slider("α : part des poids MCDM", 0.0, 1.0, 0.5, 0.05)

    @st.cache_data(show_spinner="Entraînement et calcul des valeurs SHAP…")
    def shap_importance(y, classify):
        import shap
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model = (RandomForestClassifier if classify else RandomForestRegressor)(n_estimators=200, random_state=0)
        model.fit(Xc, y)
        sv = shap.TreeExplainer(model).shap_values(Xc)
        sv = np.asarray(sv)
        if classify:  # contribution à la probabilité de la classe « accepté »
            sv = sv[..., 1] if sv.ndim == 3 else sv
        return np.abs(sv).mean(axis=0)

    classify = target.startswith("Décisions")
    phi = shap_importance(port["Accepté"].to_numpy() if classify else S, classify)
    w_new = shap_weight_update(ds.CREDIT_WEIGHTS, phi, alpha)
    fig = go.Figure()
    fig.add_bar(x=crit, y=ds.CREDIT_WEIGHTS, name="w MCDM", marker_color="#5a5a5f")
    fig.add_bar(x=crit, y=phi / phi.sum(), name="|φ| normalisé (SHAP)", marker_color="#a1a1a6")
    fig.add_bar(x=crit, y=w_new, name=f"w révisé (α = {alpha:.2f})", marker_color=ui.ACCENT)
    fig.update_layout(barmode="group", height=340, title="Poids MCDM, importance SHAP et poids révisés",
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    k = int(np.argmax(np.abs(w_new - np.array(ds.CREDIT_WEIGHTS))))
    ui.result(f"Le critère le plus révisé est **{crit[k]}** : {ds.CREDIT_WEIGHTS[k]:.3f} → {w_new[k]:.3f}. "
              + ("Les décisions du comité ne suivent pas exactement la pondération affichée par la banque."
                 if classify else "SHAP mesure l'effet réel de chaque critère, qui dépend aussi de sa dispersion "
                                  "dans les dossiers, pas seulement de son poids."))

with tab3:
    st.markdown("Les critères évoluent dans le temps : pour chaque action et chaque critère, on prévoit la valeur à "
                "$t+1$ ($\\hat x_{ij}(t+1)$), puis on applique la méthode MCDM à la matrice prévue.")
    st.caption("Prévision par lissage de Holt (le cours cite ARIMA, LSTM, GARCH ; la logique est la même). "
               "Séries mensuelles synthétiques")
    series = ds.stock_series()
    names = list(series)
    dyn_crit = list(series[names[0]].columns)
    shown = st.selectbox("Critère affiché", dyn_crit)
    forecast = np.array([[holt(series[a][c].to_numpy()) for c in dyn_crit] for a in names])
    last = np.array([[series[a][c].iloc[-1] for c in dyn_crit] for a in names])

    fig = go.Figure()
    shades = [ui.ACCENT, "#c4c4c8", "#85858b"]
    j = dyn_crit.index(shown)
    for i, a in enumerate(names):
        s = series[a][shown]
        fig.add_scatter(x=s.index, y=s.values, name=a, mode="lines", line=dict(color=shades[i], width=1.8))
        fig.add_scatter(x=[s.index[-1], "t+1"], y=[s.iloc[-1], forecast[i, j]], mode="lines+markers",
                        line=dict(color=shades[i], dash="dot"), showlegend=False)
    fig.update_layout(title=f"{shown} : historique et prévision t+1", height=320, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    w_dyn = np.array([0.45, 0.35, 0.20])
    t_dyn = [1, -1, 1]
    rc_now, rc_next = topsis(last, w_dyn, t_dyn)["RC"], topsis(forecast, w_dyn, t_dyn)["RC"]
    tbl = pd.DataFrame(forecast, index=names, columns=[f"{c}, t+1" for c in dyn_crit])
    tbl["RC* dernier mois"] = rc_now
    tbl["RC* prévu t+1"] = rc_next
    st.caption("TOPSIS, poids : rendement 0,45 (bénéfice), volatilité 0,35 (coût), liquidité 0,20 (bénéfice)")
    ui.table(tbl, digits=3)
    b_now, b_next = names[int(np.argmax(rc_now))], names[int(np.argmax(rc_next))]
    ui.result(f"Sur les valeurs prévues, TOPSIS retient **{b_next}**"
              + (" comme sur le dernier mois observé." if b_now == b_next else f", alors que le dernier mois désignait {b_now}.")
              + " Limite : l'erreur de prévision se propage dans la décision.")
