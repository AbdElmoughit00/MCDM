"""Éléments d'interface partagés : thème Plotly, éditeur de problème, graphique de classement, résultat encadré."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from mcdm.robustness import ranks

ACCENT = "#5b8def"
INK, MUTED, GRID = "#e4e4e7", "#a1a1a6", "#2a2a2d"
NEUTRALS = ["#c4c4c8", "#85858b", "#d4d4d8", "#5a5a5f", "#a1a1a6"]
MONO = "IBM Plex Mono, monospace"

pio.templates["td"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[ACCENT] + NEUTRALS,
    xaxis=dict(gridcolor=GRID, zerolinecolor="#5a5a5f", tickfont=dict(family=MONO, color=MUTED, size=12),
               linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor="#5a5a5f", tickfont=dict(family=MONO, color=MUTED, size=12),
               linecolor=GRID),
    margin=dict(l=8, r=8, t=40, b=8), hoverlabel=dict(font_family=MONO, bgcolor="#232325", bordercolor="#3a3a3d", font_color=INK),
    legend=dict(font=dict(color=MUTED)), title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
))
pio.templates.default = "td"

SENS = {1: "Bénéfice", -1: "Coût"}


def header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


def problem_editor(key, presets, show_weights=True):
    """Choix d'un jeu de données (ou import CSV), matrice modifiable, sens et poids des critères.

    Renvoie (DataFrame, types np.array, poids np.array normalisés ou None).
    """
    options = list(presets) + ["Importer un CSV"]
    choice = st.selectbox("Jeu de données", options, key=f"{key}-choice")
    if choice == "Importer un CSV":
        up = st.file_uploader("Fichier CSV : première colonne = alternatives, autres colonnes = critères numériques",
                              type="csv", key=f"{key}-upload")
        if up is None:
            st.stop()
        try:
            df = pd.read_csv(up, index_col=0).apply(pd.to_numeric, errors="raise").astype(float)
        except Exception as exc:  # fichier utilisateur : on explique au lieu de planter
            st.error(f"Lecture impossible : {exc}")
            st.stop()
        if df.shape[0] < 2 or df.shape[1] < 2 or df.isna().any().any():
            st.error("Il faut au moins 2 alternatives, 2 critères et aucune case vide")
            st.stop()
        types, weights = [1] * df.shape[1], [1 / df.shape[1]] * df.shape[1]
    else:
        p = presets[choice]
        p = p() if callable(p) else p  # jeu coûteux calculé seulement quand on le choisit
        st.caption(p["source"])
        df, types = p["matrix"], p["types"]
        weights = p["weights"] or [1 / df.shape[1]] * df.shape[1]

    X = st.data_editor(df, key=f"{key}-{choice}-matrix", width="stretch")
    if X.isna().any().any():
        st.error("La matrice contient des cases vides")
        st.stop()

    crit = pd.DataFrame({"Sens": [SENS[t] for t in types], "Poids": weights}, index=df.columns)
    cols = {"Sens": st.column_config.SelectboxColumn("Sens", options=list(SENS.values()), required=True),
            "Poids": st.column_config.NumberColumn("Poids", min_value=0.0, step=0.01, format="%.3f")}
    if not show_weights:
        crit = crit[["Sens"]]
    crit = st.data_editor(crit, key=f"{key}-{choice}-criteria", column_config=cols, width="stretch")
    types = np.array([1 if s == "Bénéfice" else -1 for s in crit["Sens"]])
    if not show_weights:
        return X, types, None
    w = crit["Poids"].to_numpy(dtype=float)
    if (w < 0).any() or w.sum() == 0:
        st.error("Les poids doivent être positifs et non tous nuls")
        st.stop()
    if not np.isclose(w.sum(), 1, atol=1e-3):
        st.caption(f"Poids saisis (somme {w.sum():.3f}) renormalisés pour sommer à 1")
        w = w / w.sum()
    return X, types, w


def table(data, digits=4, **kw):
    """Tableau numérique formaté (chiffres alignés à droite)."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    num = df.select_dtypes("float").columns  # les rangs (entiers) restent entiers
    st.dataframe(df.style.format({c: f"{{:.{digits}f}}" for c in num}), width="stretch", **kw)


def ranking_chart(names, scores, title, higher_is_better=True, fmt=".4f"):
    """Barres horizontales triées ; seule la meilleure alternative porte l'accent."""
    s = pd.Series(np.asarray(scores, dtype=float), index=list(names))
    s = s.sort_values(ascending=not higher_is_better)
    colors = [ACCENT] + ["#85858b"] * (len(s) - 1)
    # l'étiquette porte aussi l'accent : une meilleure valeur nulle (VIKOR Q = 0) n'a pas de barre visible
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                           text=[format(v, fmt) for v in s.values], textposition="outside",
                           textfont=dict(family=MONO, color=[ACCENT] + [MUTED] * (len(s) - 1))))
    fig.update_layout(title=title, height=80 + 36 * len(s), yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=_padded_range(s.values)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _padded_range(v):
    lo, hi = min(0.0, float(v.min())), max(0.0, float(v.max()))
    pad = (hi - lo) * 0.18 or 1
    return [lo - (2 * pad if lo < 0 else 0), hi + pad]  # place pour l'étiquette d'une barre négative


def padded(values, frac=0.15):
    """Plage d'axe avec marge, pour que les étiquettes des points extrêmes ne soient pas rognées."""
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = (hi - lo) * frac or abs(hi) * frac or 1
    return [lo - pad, hi + pad]


def result(text):
    """Signature de l'application : la conclusion encadrée, comme la réponse finale d'une copie corrigée."""
    with st.container(border=True):
        st.caption("Résultat")
        st.markdown(text)


def rank_series(names, scores, higher_is_better=True):
    return pd.Series(ranks(scores, higher_is_better), index=list(names), name="Rang")
