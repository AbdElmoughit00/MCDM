import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from mcdm import fuzzy as fz

ui.header("Ensembles flous", "Quand l'incertitude vient du jugement humain (épistémique) plutôt que du hasard : "
          "degrés d'appartenance, nombres flous et leurs extensions")


def parse(text, k):
    try:
        vals = tuple(float(v) for v in text.replace(";", ",").split(","))
    except ValueError:
        st.error(f"« {text} » : saisir {k} nombres séparés par des virgules")
        st.stop()
    if len(vals) != k or list(vals) != sorted(vals):
        st.error(f"« {text} » : il faut {k} nombres croissants")
        st.stop()
    return vals


tabs = st.tabs(["Fonctions d'appartenance", "Nombres flous et intervalles", "Intuitionniste, pythagoricien, q-ROF",
                "Pictural", "Hésitant"])

with tabs[0]:
    kind = st.radio("Forme", ["Trapézoïdale", "Triangulaire", "Gaussienne"], horizontal=True)
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        if kind == "Trapézoïdale":
            st.caption("Exemple du cours : l'ensemble « âge moyen »")
            a, b, c, d = parse(st.text_input("a, b, c, d", "25, 35, 55, 65"), 4)
            mu = lambda x: fz.trapezoid(x, a, b, c, d)
            lo_x, hi_x = a - 0.25 * (d - a) - 1, d + 0.25 * (d - a) + 1
            support = f"]{a:g} ; {d:g}["
        elif kind == "Triangulaire":
            a, b, c = parse(st.text_input("a, b, c", "20, 40, 60"), 3)
            mu = lambda x: fz.triangle(x, a, b, c)
            lo_x, hi_x = a - 0.25 * (c - a) - 1, c + 0.25 * (c - a) + 1
            support = f"]{a:g} ; {c:g}["
        else:
            m = st.number_input("Centre m", value=45.0)
            sg = st.number_input("Écart σ", value=10.0, min_value=0.1)
            mu = lambda x: fz.gaussian(x, m, sg)
            lo_x, hi_x = m - 4 * sg, m + 4 * sg
            support = "ℝ tout entier (μ > 0 partout)"
        x0 = st.slider("Valeur x", float(lo_x), float(hi_x), float((lo_x + hi_x) / 2))
        alpha = st.slider("Niveau α de la coupe", 0.0, 1.0, 0.5, 0.05)
    xs = np.linspace(lo_x, hi_x, 600)
    ys = mu(xs)
    cut = xs[ys >= alpha - 1e-12]
    with c2:
        fig = go.Figure(go.Scatter(x=xs, y=ys, line=dict(color=ui.INK, width=2), name="μ(x)"))
        if len(cut):
            fig.add_scatter(x=[cut.min(), cut.max()], y=[alpha, alpha], mode="lines",
                            line=dict(color=ui.ACCENT, width=5), name=f"α-coupe (α = {alpha:.2f})")
        fig.add_scatter(x=[x0], y=[float(mu(np.array([x0]))[0])], mode="markers",
                        marker=dict(size=11, color=ui.ACCENT), name="μ(x)")
        fig.update_layout(height=320, yaxis=dict(range=[-0.05, 1.1], title="μ(x)"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    core = xs[ys >= 1 - 1e-9]
    t = pd.DataFrame({"Notion": ["Hauteur h(A)", "Support", "Noyau", f"α-coupe (α = {alpha:.2f})"],
                      "Valeur": [f"{ys.max():.2f} ({'normalisé' if ys.max() > 1 - 1e-9 else 'non normalisé'})",
                                 support,
                                 f"[{core.min():.1f} ; {core.max():.1f}]" if len(core) else "vide",
                                 f"[{cut.min():.1f} ; {cut.max():.1f}]" if len(cut) else "vide"]})
    st.dataframe(t, hide_index=True, width="stretch")
    ui.result(f"μ({x0:.1f}) = **{float(mu(np.array([x0]))[0]):.2f}** : x appartient à l'ensemble à "
              f"{float(mu(np.array([x0]))[0]):.0%}, là où la logique booléenne n'admet que 0 ou 1.")

with tabs[1]:
    st.markdown("**Nombres flous trapézoïdaux** N(a, b, c, d) ; triangulaire si b = c")
    c1, c2, c3 = st.columns(3)
    A = parse(c1.text_input("A", "4, 5, 6, 7"), 4)
    B = parse(c2.text_input("B", "1, 2, 3, 4"), 4)
    op = c3.selectbox("Opération", ["A + B", "A − B", "A ⊗ B", "A ⊘ B", "λA"])
    lam = c3.number_input("λ", value=2.0) if op == "λA" else None
    try:
        R = {"A + B": lambda: fz.f_add(A, B), "A − B": lambda: fz.f_sub(A, B), "A ⊗ B": lambda: fz.f_mul(A, B),
             "A ⊘ B": lambda: fz.f_div(A, B), "λA": lambda: fz.f_scale(lam, A)}[op]()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    fig = go.Figure()
    for name, N, color in [("A", A, "#85858b"), ("B", B, "#5a5a5f"), (op, R, ui.ACCENT)]:
        if name == "B" and op == "λA":
            continue
        fig.add_scatter(x=list(N), y=[0, 1, 1, 0], name=name, line=dict(color=color, width=2.5 if color == ui.ACCENT else 1.8))
    fig.update_layout(height=280, yaxis=dict(range=[-0.05, 1.1], title="μ"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    ui.result(f"{op} = **({', '.join(f'{v:.3g}' for v in R)})**. La soustraction et la division croisent les "
              "bornes (a1 − b4, …), ce qui élargit toujours l'incertitude.")

    st.markdown("**Nombres d'intervalle** [a1, a2]")
    c1, c2 = st.columns(2)
    I1 = parse(c1.text_input("A = [a1, a2]", "2, 4"), 2)
    I2 = parse(c2.text_input("B = [b1, b2]", "1, 3"), 2)
    rows = [("A + B", (I1[0] + I2[0], I1[1] + I2[1])), ("A − B", (I1[0] - I2[1], I1[1] - I2[0])),
            ("A × B", fz.i_mul(I1, I2))]
    try:
        rows.append(("A / B", fz.i_div(I1, I2)))
    except ValueError as exc:
        rows.append(("A / B", str(exc)))
    st.dataframe(pd.DataFrame({"Opération": [r[0] for r in rows],
                               "Résultat": [r[1] if isinstance(r[1], str) else f"[{r[1][0]:.3g} ; {r[1][1]:.3g}]"
                                            for r in rows]}), hide_index=True, width="stretch")
    st.caption(f"Centre m(A) = {(I1[0] + I1[1]) / 2:g}, largeur w(A) = {(I1[1] - I1[0]) / 2:g} ; "
               f"A ≤ B ⇔ m(A) ≤ m(B), ici {'vrai' if (I1[0] + I1[1]) <= (I2[0] + I2[1]) else 'faux'}")

with tabs[2]:
    st.markdown("Un nombre flou à paire orthogonale donne un degré d'appartenance μ, de non-appartenance ν et "
                "d'hésitation π. La contrainte se relâche avec q : μ + ν ≤ 1 (intuitionniste, Atanassov 1986), "
                "μ² + ν² ≤ 1 (pythagoricien), μ^q + ν^q ≤ 1 (q-ROF).")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        mu_ = st.slider("μ : appartenance", 0.0, 1.0, 0.6, 0.01)
        nu_ = st.slider("ν : non-appartenance", 0.0, 1.0, 0.3, 0.01)
        rows = []
        for q, name in [(1, "Intuitionniste (q = 1)"), (2, "Pythagoricien (q = 2)"), (3, "q-ROF (q = 3)")]:
            ok = mu_ ** q + nu_ ** q <= 1 + 1e-12
            rows.append((name, "oui" if ok else "non", f"{fz.hesitation(mu_, nu_, q):.3f}" if ok else "-"))
        st.dataframe(pd.DataFrame(rows, columns=["Type", "Admissible", "Hésitation π"]), hide_index=True,
                     width="stretch")
    with c2:
        t = np.linspace(0, 1, 200)
        fig = go.Figure()
        for q, color in [(1, ui.INK), (2, "#a1a1a6"), (3, "#85858b")]:
            fig.add_scatter(x=t, y=(1 - t ** q) ** (1 / q), name=f"μ^{q} + ν^{q} = 1", line=dict(color=color))
        fig.add_scatter(x=[mu_], y=[nu_], mode="markers", marker=dict(size=12, color=ui.ACCENT), name="(μ, ν)")
        fig.update_layout(title="Espaces IFN, PFN et q-ROFN", height=340, xaxis_title="μ",
                          yaxis_title="ν", xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02], scaleanchor="x"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("**Exemple du vote** : 10 personnes, 5 pour, 3 contre, 2 abstentions : α = (0,5 ; 0,3), "
                "π = 0,2. Score S = μ − ν, précision H = μ + ν.")
    if mu_ + nu_ <= 1:
        a = (mu_, nu_)
        b = (0.5, 0.3)
        sa, sb = fz.ifn_score(a), fz.ifn_score(b)
        # score d'abord, précision pour départager
        key_a, key_b = (round(sa, 9), round(fz.ifn_accuracy(a), 9)), (round(sb, 9), round(fz.ifn_accuracy(b), 9))
        better = "α_saisi > α_vote" if key_a > key_b else "α_vote > α_saisi" if key_a < key_b else "égalité"
        agg = fz.ifwa([a, b], [0.5, 0.5])
        ui.result(f"α_saisi = ({mu_:.2f} ; {nu_:.2f}), S = {sa:.2f} ; α_vote = (0,50 ; 0,30), S = {sb:.2f}. "
                  f"Comparaison : **{better}**. Agrégation IFWA à poids égaux : ({agg[0]:.3f} ; {agg[1]:.3f}).")
    else:
        st.info("μ + ν > 1 : le point n'est pas un nombre intuitionniste, seulement pythagoricien ou q-ROF")

with tabs[3]:
    st.markdown("Pour un vote à quatre issues (pour, abstention, contre, refus de voter) : μ positif, π neutre, "
                "ν négatif, avec μ + π + ν ≤ 1 ; le reste est le degré de refus.")
    c1, c2, c3 = st.columns(3)
    pos = c1.number_input("Pour", 0, 1000, 45)
    neu = c2.number_input("Abstention", 0, 1000, 20)
    neg = c3.number_input("Contre", 0, 1000, 25)
    total = st.number_input("Votants inscrits", 1, 5000, 100)
    if pos + neu + neg > total:
        st.error("Pour + abstention + contre dépasse le nombre d'inscrits")
        st.stop()
    A = (pos / total, neg / total, neu / total)
    lam = st.slider("λ pour λA", 0.1, 5.0, 2.0, 0.1)
    s = fz.pfn_scale(lam, A)
    ui.result(f"A = (μ = {A[0]:.2f}, ν = {A[1]:.2f}, π = {A[2]:.2f}), refus = **{1 - sum(A):.2f}**. "
              f"λA = ({s[0]:.3f} ; {s[1]:.3f} ; {s[2]:.3f}).")

with tabs[4]:
    st.markdown("Plusieurs décideurs donnent chacun un degré : h = {γ1, γ2, …} (Torra, 2010). On compare par le score "
                "s(h) = moyenne, puis par la déviation σ(h) en cas d'égalité (la plus petite gagne).")
    defaults = ["0.2, 0.3, 0.5, 0.8", "0.4, 0.6, 0.8", "0.3, 0.45, 0.6"]
    hs = []
    cols = st.columns(3)
    for k, col in enumerate(cols):
        txt = col.text_input(f"h{k + 1}", defaults[k])
        try:
            h = sorted(float(v) for v in txt.split(","))
        except ValueError:
            st.error(f"h{k + 1} : nombres séparés par des virgules")
            st.stop()
        if not all(0 <= v <= 1 for v in h):
            st.error(f"h{k + 1} : les degrés doivent être dans [0, 1]")
            st.stop()
        hs.append(h)
    t = pd.DataFrame({"Score s(h)": [fz.hfe_score(h) for h in hs], "Déviation σ(h)": [fz.hfe_deviation(h) for h in hs]},
                     index=["h1", "h2", "h3"])
    ui.table(t)
    order = sorted(range(3), key=lambda i: (-round(t.iloc[i, 0], 10), round(t.iloc[i, 1], 10)))
    d = pd.DataFrame([[fz.hfe_hamming(hs[i], hs[j]) for j in range(3)] for i in range(3)],
                     index=["h1", "h2", "h3"], columns=["h1", "h2", "h3"])
    st.caption("Distance de Hamming hésitante (l'ensemble le plus court est complété par sa plus grande valeur)")
    ui.table(d, digits=3)
    ui.result("Ordre : **" + " > ".join(f"h{i + 1}" for i in order) + "**"
              + (". h1 et h3 ont le même score (0,45) : la déviation plus faible de h3 le place devant."
                 if [round(x, 6) for x in t.iloc[:, 0]] == [0.45, 0.6, 0.45] else "."))
