"""Évaluation absolue d'une alternative isolée."""
import numpy as np


def utilities(x, lo, hi, types):
    """Utilités linéaires du cours : (x - min)/(max - min) en bénéfice, (max - x)/(max - min) en coût."""
    x, lo, hi = (np.asarray(a, dtype=float) for a in (x, lo, hi))
    if (hi <= lo).any():
        raise ValueError("Chaque critère doit avoir max > min")
    u = (x - lo) / (hi - lo)
    u = np.where(np.asarray(types) > 0, u, 1 - u)
    return np.clip(u, 0, 1)


def maut(x, w, lo, hi, types):
    u = utilities(x, lo, hi, types)
    return u, float(np.asarray(w, dtype=float) @ u)


def distance_to_ideal(u):
    """Indice C = D- / (D+ + D-) d'une alternative en utilités normalisées : idéal = 1, anti-idéal = 0."""
    u = np.asarray(u, dtype=float)
    d_plus = np.sqrt(((u - 1) ** 2).sum())
    d_minus = np.sqrt((u ** 2).sum())
    return d_plus, d_minus, d_minus / (d_plus + d_minus)


def threshold_table(scores, y, thresholds=None):
    """Pour chaque seuil τ : TPR, FPR, distance au point idéal d, indice de Youden J."""
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
