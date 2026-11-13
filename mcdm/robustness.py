"""Robustesse : rangs, corrélation de Spearman (cours p.77-79), sensibilité et Monte-Carlo sur les poids."""
import numpy as np
from scipy.stats import rankdata

from mcdm.outranking import promethee
from mcdm.ranking import topsis, vikor, waspas

# Valeurs critiques de ρ de Spearman, test bilatéral à 5 % (n = 6 : 0,886 comme dans le cours p.79)
SPEARMAN_CRITICAL = {5: 1.0, 6: 0.886, 7: 0.786, 8: 0.738, 9: 0.700, 10: 0.648, 11: 0.618, 12: 0.587}


def critical_rho(n):
    return SPEARMAN_CRITICAL.get(n, 1.96 / np.sqrt(n - 1))  # approximation normale au-delà de 12


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
    if method == "PROMETHEE II":
        span = X.max(0) - X.min(0)
        n = X.shape[1]
        return promethee(X, w, types, ["Linéaire"] * n, span * 0.1, span * 0.5, span * 0.3)["phi"]
    raise ValueError(method)


METHODS = ["WSM", "WASPAS", "TOPSIS", "VIKOR", "PROMETHEE II"]


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


def monte_carlo(score_fn, w, n_runs=1000, concentration=50.0, seed=0, higher_is_better=True):
    """Tire des poids de Dirichlet autour de w ; renvoie la fréquence de chaque rang par alternative."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float)
    samples = rng.dirichlet(np.maximum(w, 1e-6) * concentration, size=n_runs)
    m = len(score_fn(w))
    freq = np.zeros((m, m))
    for ws in samples:
        r = ranks(score_fn(ws), higher_is_better)
        freq[np.arange(m), r - 1] += 1
    return freq / n_runs, samples
