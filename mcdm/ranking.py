"""Classement additif / compensatoire : WSM, WPM, WASPAS, TOPSIS, VIKOR."""
import numpy as np


def _check(X, w, types):
    X, w, types = np.asarray(X, dtype=float), np.asarray(w, dtype=float), np.asarray(types)
    if X.ndim != 2 or X.shape[1] != len(w) or len(w) != len(types):
        raise ValueError("Dimensions incohérentes entre matrice, poids et sens des critères")
    if (w < 0).any() or not np.isclose(w.sum(), 1, atol=1e-3):
        raise ValueError("Les poids doivent être positifs et sommer à 1")
    return X, w, types > 0


def linear_norm(X, types):
    """Cours p.55 : bénéfice x / max, coût min / x."""
    X = np.asarray(X, dtype=float)
    if (X <= 0).any():
        raise ValueError("Cette normalisation exige des valeurs strictement positives")
    benefit = np.asarray(types) > 0
    return np.where(benefit, X / X.max(axis=0), X.min(axis=0) / X)


def waspas(X, w, types, lam=0.5):
    """WSM (Q1), WPM (Q2) et WASPAS Q = λ Q1 + (1 - λ) Q2 (cours p.56-57)."""
    X, w, types = _check(X, w, types)
    R = linear_norm(X, types)
    q1 = R @ w
    q2 = np.prod(R ** w, axis=1)
    return {"normalized": R, "WSM": q1, "WPM": q2, "WASPAS": lam * q1 + (1 - lam) * q2}


def topsis(X, w, types):
    """TOPSIS (cours p.62-64) avec normalisation vectorielle."""
    X, w, benefit = _check(X, w, types)
    norm = np.sqrt((X ** 2).sum(axis=0))
    R = X / np.where(norm > 0, norm, 1)
    V = R * w
    ideal = np.where(benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(benefit, V.min(axis=0), V.max(axis=0))
    s_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    total = s_plus + s_minus
    rc = np.divide(s_minus, total, out=np.full_like(total, 0.5), where=total > 0)
    return {"normalized": R, "weighted": V, "ideal": ideal, "anti": anti, "S+": s_plus, "S-": s_minus, "RC": rc}


def vikor(X, w, types, v=0.5):
    """VIKOR (cours p.66-71) : S (utilité), R (regret), Q, puis conditions de compromis."""
    X, w, benefit = _check(X, w, types)
    best = np.where(benefit, X.max(axis=0), X.min(axis=0))
    worst = np.where(benefit, X.min(axis=0), X.max(axis=0))
    span = best - worst
    gap = np.divide(best - X, span, out=np.zeros_like(X), where=span != 0)
    term = w * gap
    S, R = term.sum(axis=1), term.max(axis=1)

    def scaled(a):
        return (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else np.zeros_like(a)

    Q = v * scaled(S) + (1 - v) * scaled(R)
    m = len(Q)
    order = np.argsort(Q, kind="stable")
    a1, a2 = order[0], order[1]
    dq = 1 / (m - 1)
    c1 = Q[a2] - Q[a1] >= dq - 1e-12
    c2 = S[a1] == S.min() and R[a1] == R.min()  # cours p.70 : meilleure en S et en R
    if c1 and c2:
        compromise = [a1]
    elif c1:
        compromise = [a1, a2]
    else:
        compromise = [int(i) for i in order if Q[i] - Q[a1] < dq - 1e-12]
    return {"best": best, "worst": worst, "terms": term, "S": S, "R": R, "Q": Q,
            "DQ": dq, "C1": bool(c1), "C2": bool(c2), "compromise": [int(i) for i in compromise]}
