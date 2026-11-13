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


tabs = st.tabs(["Fonctions d'appartenance", "Variable linguistique", "Nombres flous et intervalles",
                "Intuitionniste, pythagoricien, q-ROF", "Pictural", "Hésitant"])
tab_mf, tab_ling, tab_num, tab_ifs, tab_pfs, tab_hfs = tabs

with tab_mf:
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
    st.caption("Ces trois formes sont des ensembles flous convexes : "
               "μ(λx₁ + (1 − λ)x₂) ≥ min(μ(x₁), μ(x₂)) pour tous x₁, x₂ et λ ∈ [0, 1]")
    ui.result(f"μ({x0:.1f}) = **{float(mu(np.array([x0]))[0]):.2f}** : x appartient à l'ensemble à "
              f"{float(mu(np.array([x0]))[0]):.0%}, là où la logique booléenne n'admet que 0 ou 1.")

with tab_ling:
    st.markdown("Une variable linguistique est un triplet (V, X, T_V) : une variable V, un domaine X et un ensemble "
                "T_V de termes, chacun décrit par un ensemble flou.")
    st.caption("Exemple du cours : V = pourboire, X = ℝ+, T_V = {faible, moyen, élevé}")
    terms = {"faible": (0, 5, 10), "moyen": (10, 15, 20), "élevé": (20, 25, 30)}
    x_tip = st.slider("Pourboire x", 0.0, 30.0, 12.5, 0.5)
    xs = np.linspace(0, 30, 601)
    fig = go.Figure()
    for (name, (a, b, c)), color in zip(terms.items(), ["#a1a1a6", ui.ACCENT, "#85858b"]):
        fig.add_scatter(x=xs, y=fz.triangle(xs, a, b, c), name=f"{name} ({a}, {b}, {c})", line=dict(color=color, width=2))
    fig.add_vline(x=x_tip, line=dict(color="#5a5a5f", dash="dot"))
    fig.update_layout(height=300, xaxis_title="Pourboire", yaxis=dict(range=[-0.05, 1.1], title="μ"),
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    degrees = {name: float(fz.triangle(x_tip, *abc)) for name, abc in terms.items()}
    st.dataframe(pd.DataFrame({"Terme": list(degrees), "Degré μ(x)": [f"{v:.2f}" for v in degrees.values()]}),
                 hide_index=True, width="stretch")
    best = max(degrees, key=degrees.get)
    ui.result(f"Un pourboire de {x_tip:g} est « {best} » au degré **{degrees[best]:.2f}**." if degrees[best] > 0
              else f"Un pourboire de {x_tip:g} n'appartient à aucun terme (degré 0 partout).")

with tab_num:
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
              "bornes : A − B = (a1 − b4, a2 − b3, a3 − b2, a4 − b1).")

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

with tab_ifs:
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
        st.markdown("**Opérations sur les nombres intuitionnistes** (α₁ = α_saisi, α₂ = α_vote)")
        lam_i = st.slider("λ", 0.1, 5.0, 2.0, 0.1, key="lam-ifn")
        ops = {"α₁ + α₂ = (μ₁ + μ₂ − μ₁μ₂ ; ν₁ν₂)": fz.ifn_add(a, b),
               "α₁ × α₂ = (μ₁μ₂ ; ν₁ + ν₂ − ν₁ν₂)": fz.ifn_mul(a, b),
               "λα₁ = (1 − (1 − μ₁)^λ ; ν₁^λ)": fz.ifn_scale(lam_i, a),
               "α₁^λ = (μ₁^λ ; 1 − (1 − ν₁)^λ)": fz.ifn_power(a, lam_i)}
        st.dataframe(pd.DataFrame({"Opération": list(ops), "Résultat (μ ; ν)": [f"({m:.3f} ; {n:.3f})" for m, n in ops.values()]}),
                     hide_index=True, width="stretch")
    else:
        st.info("μ + ν > 1 : le point n'est pas un nombre intuitionniste, seulement pythagoricien ou q-ROF")

    st.markdown("**q-ROF à valeur d'intervalle** : μ et ν deviennent des intervalles, avec μU^q + νU^q ≤ 1, "
                "et l'hésitation est elle aussi un intervalle")
    c1, c2, c3 = st.columns(3)
    mu_i = parse(c1.text_input("μ = [μL, μU]", "0.5, 0.7"), 2)
    nu_i = parse(c2.text_input("ν = [νL, νU]", "0.2, 0.4"), 2)
    q_i = c3.number_input("q", 1, 10, 3)
    try:
        pi_l, pi_u = fz.ivq_hesitation(mu_i, nu_i, q_i)
        st.caption(f"πL = (1 − μU^q − νU^q)^(1/q) = {pi_l:.3f} ; πU = (1 − μL^q − νL^q)^(1/q) = {pi_u:.3f} : "
                   f"π = [{pi_l:.3f} ; {pi_u:.3f}]")
    except ValueError as exc:
        st.error(str(exc))

with tab_pfs:
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
    st.markdown("**Opérations** (B = second vote : 30 pour, 40 contre, 10 abstentions sur 100)")
    B = (0.30, 0.40, 0.10)
    lam = st.slider("λ", 0.1, 5.0, 2.0, 0.1)
    ops = {"A ⊕ B = (μA + μB − μAμB ; νAνB ; πAπB)": fz.pfn_add(A, B),
           "λA = (1 − (1 − μA)^λ ; νA^λ ; πA^λ)": fz.pfn_scale(lam, A),
           "A^λ = (μA^λ ; 1 − (1 − νA)^λ ; 1 − (1 − πA)^λ)": fz.pfn_power(A, lam)}
    st.dataframe(pd.DataFrame({"Opération": list(ops),
                               "Résultat (μ ; ν ; π)": [f"({x:.3f} ; {y:.3f} ; {z:.3f})" for x, y, z in ops.values()]}),
                 hide_index=True, width="stretch")
    ui.result(f"A = (μ = {A[0]:.2f}, ν = {A[1]:.2f}, π = {A[2]:.2f}), degré de refus = **{1 - sum(A):.2f}**.")

with tab_hfs:
    st.markdown("Plusieurs décideurs donnent chacun un degré : h = {γ1, γ2, …} (Torra, 2010). On compare par le score "
                "s(h) = moyenne, puis par la déviation σ(h) en cas d'égalité (la plus petite gagne).")
    st.caption("Exemple du cours : comparer h1, h2 et h3")
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
    st.caption("Distance de Hamming hésitante : ||h1 − h2|| = (1/l) Σ |h1σ(k) − h2σ(k)|, l = max des tailles ; "
               "le cours ne précise pas comment compléter l'ensemble le plus court, ici par sa plus grande valeur")
    ui.table(d, digits=3)
    st.markdown("**Opérations** sur h1 et h2")
    a_h = st.slider("α", 0.1, 5.0, 2.0, 0.1, key="alpha-hfe")
    fmt = lambda h: "{" + ", ".join(f"{g:.3g}" for g in h) + "}"
    ops = {"h1^α = ∪ {γ^α}": fz.hfe_power(hs[0], a_h), "αh1 = ∪ {1 − (1 − γ)^α}": fz.hfe_scale(a_h, hs[0]),
           "h1 ⊕ h2 = ∪ {γ1 + γ2 − γ1γ2}": fz.hfe_add(hs[0], hs[1]), "h1 ⊗ h2 = ∪ {γ1γ2}": fz.hfe_mul(hs[0], hs[1])}
    st.dataframe(pd.DataFrame({"Opération": list(ops), "Résultat": [fmt(h) for h in ops.values()]}),
                 hide_index=True, width="stretch")
    ui.result("Ordre : **" + " > ".join(f"h{i + 1}" for i in order) + "**"
              + (". h1 et h3 ont le même score (0,45) : la déviation plus faible de h3 le place devant."
                 if [round(x, 6) for x in t.iloc[:, 0]] == [0.45, 0.6, 0.45] else "."))
