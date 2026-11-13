"""Surclassement : PROMETHEE I/II (Brans & Vincke, 1985) et ELECTRE III (Roy, 1978).

Les diapositives du cours ne détaillent pas ces méthodes : formulations de référence de la littérature.
"""
import numpy as np

PREFERENCE_FUNCTIONS = ["Usuelle", "En U", "En V", "Par paliers", "Linéaire", "Gaussienne"]


def preference(d, kind, q=0.0, p=1.0, s=1.0):
    """Fonction de préférence P(d) ∈ [0, 1] appliquée à l'écart orienté d (positif = a meilleur que b)."""
    d = np.asarray(d, dtype=float)
    pos = d > 0
    if kind == "Usuelle":
        out = pos.astype(float)
    elif kind == "En U":
        out = (d > q).astype(float)
    elif kind == "En V":
        out = np.clip(d / p, 0, 1) if p > 0 else pos.astype(float)
    elif kind == "Par paliers":
        out = np.where(d > p, 1.0, np.where(d > q, 0.5, 0.0))
    elif kind == "Linéaire":
        out = np.clip((d - q) / (p - q), 0, 1) if p > q else (d > q).astype(float)
    elif kind == "Gaussienne":
        out = np.where(pos, 1 - np.exp(-(d ** 2) / (2 * s ** 2)), 0.0)
    else:
        raise ValueError(f"Fonction de préférence inconnue : {kind}")
    return np.where(pos, out, 0.0)


def promethee(X, w, types, kinds, q, p, s):
    """Renvoie la matrice π, les flux φ+, φ-, φ et la relation PROMETHEE I."""
    X, w = np.asarray(X, dtype=float), np.asarray(w, dtype=float)
    m, n = X.shape
    sign = np.where(np.asarray(types) > 0, 1.0, -1.0)
    P = np.zeros((n, m, m))
    for j in range(n):
        d = sign[j] * (X[:, j][:, None] - X[:, j][None, :])
        P[j] = preference(d, kinds[j], q[j], p[j], s[j])
    pi = np.tensordot(w / w.sum(), P, axes=1)
    np.fill_diagonal(pi, 0)
    phi_plus = pi.sum(axis=1) / (m - 1)
    phi_minus = pi.sum(axis=0) / (m - 1)
    # PROMETHEE I : P (préférence), I (indifférence), R (incomparabilité)
    rel = np.full((m, m), "", dtype=object)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            ge = phi_plus[a] >= phi_plus[b] - 1e-12 and phi_minus[a] <= phi_minus[b] + 1e-12
            eq = np.isclose(phi_plus[a], phi_plus[b]) and np.isclose(phi_minus[a], phi_minus[b])
            le = phi_plus[a] <= phi_plus[b] + 1e-12 and phi_minus[a] >= phi_minus[b] - 1e-12
            rel[a, b] = "I" if eq else "P" if ge else "P⁻¹" if le else "R"
    return {"P": P, "pi": pi, "phi+": phi_plus, "phi-": phi_minus, "phi": phi_plus - phi_minus, "relation": rel}


def electre3(X, w, types, q, p, v, alpha=-0.15, beta=0.30):
    """Concordance, discordance, crédibilité, puis distillations descendante et ascendante."""
    X, w = np.asarray(X, dtype=float), np.asarray(w, dtype=float)
    q, p, v = (np.asarray(t, dtype=float) for t in (q, p, v))
    if ((q > p) | (p > v)).any():
        raise ValueError("Seuils incohérents : il faut q <= p <= v pour chaque critère")
    m, n = X.shape
    sign = np.where(np.asarray(types) > 0, 1.0, -1.0)
    c = np.zeros((n, m, m))
    d = np.zeros((n, m, m))
    for j in range(n):
        # diff[a, b] = avance de b sur a sur le critère j
        diff = sign[j] * (X[:, j][None, :] - X[:, j][:, None])
        span = p[j] - q[j]
        c[j] = np.where(diff <= q[j], 1.0, np.where(diff >= p[j], 0.0, (p[j] - diff) / (span if span else 1)))
        vspan = v[j] - p[j]
        d[j] = np.where(diff <= p[j], 0.0, np.where(diff >= v[j], 1.0, (diff - p[j]) / (vspan if vspan else 1)))
    C = np.tensordot(w / w.sum(), c, axes=1)
    S = C.copy()
    for a in range(m):
        for b in range(m):
            strong = d[:, a, b] > C[a, b]
            if strong.any():
                # d_j > C implique C < 1, le dénominateur ne s'annule pas
                S[a, b] = C[a, b] * np.prod((1 - d[strong, a, b]) / (1 - C[a, b]))
    np.fill_diagonal(C, 1)
    np.fill_diagonal(S, 1)
    s_fn = lambda lam: alpha * lam + beta  # seuil de discrimination s(λ)
    desc = _distill(S, s_fn, descending=True)
    asc = _distill(S, s_fn, descending=False)
    return {"concordance_j": c, "C": C, "discordance": d, "S": S, "descending": desc, "ascending": asc}


def _distill(S, s_fn, descending):
    """Distillation : renvoie la liste ordonnée des classes (meilleure en premier)."""
    remaining = list(range(S.shape[0]))
    classes = []
    while remaining:
        D = list(remaining)
        lam = max((S[a, b] for a in D for b in D if a != b), default=0.0)
        while len(D) > 1:
            cut = lam - s_fn(lam)
            lower = [S[a, b] for a in D for b in D if a != b and S[a, b] < cut]
            lam_next = max(lower, default=0.0)
            qual = []
            for a in D:
                strength = sum(1 for b in D if b != a and _outranks(S, a, b, lam_next, s_fn))
                weakness = sum(1 for b in D if b != a and _outranks(S, b, a, lam_next, s_fn))
                qual.append(strength - weakness)
            target = max(qual) if descending else min(qual)
            D = [a for a, qa in zip(D, qual) if qa == target]
            if lam_next == 0:
                break
            lam = lam_next
        classes.append(D)
        remaining = [a for a in remaining if a not in D]
    return classes if descending else classes[::-1]


def _outranks(S, a, b, lam, s_fn):
    return S[a, b] > lam and S[a, b] - S[b, a] > s_fn(S[a, b])


def class_rank(classes, m):
    """Rang (1 = meilleur) de chaque alternative à partir d'une liste de classes."""
    r = np.zeros(m, dtype=int)
    for k, cls in enumerate(classes, start=1):
        r[cls] = k
    return r
