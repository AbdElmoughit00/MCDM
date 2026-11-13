"""Robustesse : rangs, corrélation de Spearman, sensibilité aux poids."""
import numpy as np
from scipy.stats import rankdata

from mcdm.ranking import topsis, vikor, waspas


def scores(method, X, w, types):
    """Score de chaque méthode de classement, orienté pour que « plus grand = meilleur »."""
    X = np.asarray(X, dtype=float)
    if method == "WSM":
        return waspas(X, w, types)["WSM"]
    if method == "WASPAS":
        return waspas(X, w, types)["WASPAS"]
    if method == "TOPSIS":
        return topsis(X, w, types)["RC"]
    if method == "VIKOR":
        return -vikor(X, w, types)["Q"]
    raise ValueError(method)


METHODS = ["WSM", "WASPAS", "TOPSIS", "VIKOR"]


def ranks(scores, higher_is_better=True):
    """Rang 1 = meilleure alternative ; ex aequo au rang minimum."""
    s = np.asarray(scores, dtype=float)
    return rankdata(-s if higher_is_better else s, method="min").astype(int)


def spearman(r1, r2):
    """ρ = 1 - 6 Σ d² / (n (n² - 1))."""
    d = np.asarray(r1) - np.asarray(r2)
    n = len(d)
    return 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1))


def spearman_matrix(rank_table):
    """rank_table : dict {méthode: rangs} -> matrice des ρ."""
    names = list(rank_table)
    M = np.array([[spearman(rank_table[a], rank_table[b]) for b in names] for a in names])
    return names, M


def vary_weight(w, j, value):
    """Fixe w_j = value et répartit 1 - value sur les autres critères au prorata de leurs poids initiaux."""
    w = np.asarray(w, dtype=float)
    rest = np.delete(w, j)
    rest = rest / rest.sum() if rest.sum() > 0 else np.full_like(rest, 1 / len(rest))
    return np.insert(rest * (1 - value), j, value)
