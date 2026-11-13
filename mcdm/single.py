"""Évaluation absolue d'une alternative isolée (cours p.80-89) et outils IA associés (p.90-94)."""
import numpy as np


def utilities(x, lo, hi, types):
    """Utilités linéaires du cours p.82 : (x - min)/(max - min) en bénéfice, (max - x)/(max - min) en coût."""
    x, lo, hi = (np.asarray(a, dtype=float) for a in (x, lo, hi))
    if (hi <= lo).any():
        raise ValueError("Chaque critère doit avoir max > min")
    u = (x - lo) / (hi - lo)
    u = np.where(np.asarray(types) > 0, u, 1 - u)
    return np.clip(u, 0, 1)


def maut(x, w, lo, hi, types):
    u = utilities(x, lo, hi, types)
    return u, float(np.asarray(w, dtype=float) @ u)


def distance_to_ideal(u, w=None):
    """Indice C = D- / (D+ + D-) d'une alternative en utilités : idéal = 1, anti-idéal = 0 (cours p.89)."""
    u = np.asarray(u, dtype=float)
    w = np.ones_like(u) if w is None else np.asarray(w, dtype=float)
    d_plus = np.sqrt((w * (u - 1) ** 2).sum())
    d_minus = np.sqrt((w * u ** 2).sum())
    return d_plus, d_minus, d_minus / (d_plus + d_minus)


def threshold_table(scores, y, thresholds=None):
    """Pour chaque seuil τ : TPR, FPR, distance au point idéal d, indice de Youden J (cours p.86-88)."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        raise ValueError("Il faut des décisions passées des deux classes (acceptées et rejetées)")
    taus = np.unique(scores)[::-1] if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    for t in taus:
        pred = scores >= t
        tpr = (pred & y).sum() / y.sum()
        fpr = (pred & ~y).sum() / (~y).sum()
        rows.append((t, tpr, fpr, np.hypot(fpr, 1 - tpr), tpr - fpr))
    return np.array(rows)  # colonnes : tau, TPR, FPR, d, J


def holt(series, alpha=0.5, beta=0.3):
    """Lissage exponentiel double de Holt ; prévision à t+1.

    shortcut: pas d'ARIMA/LSTM comme dans le cours (p.94), passer à statsmodels si on veut des intervalles de prévision.
    """
    s = np.asarray(series, dtype=float)
    level, trend = s[0], s[1] - s[0]
    for x in s[1:]:
        prev = level
        level = alpha * x + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return level + trend


def shap_weight_update(w_mcdm, shap_abs_mean, alpha):
    """Cours p.92 : w_new = α w_MCDM + (1 - α) |φ_j| / Σ|φ|."""
    phi = np.asarray(shap_abs_mean, dtype=float)
    return alpha * np.asarray(w_mcdm, dtype=float) + (1 - alpha) * phi / phi.sum()
