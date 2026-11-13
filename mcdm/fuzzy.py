"""Logique floue (cours p.95-130) et MCDM floue (p.131-150)."""
import numpy as np

# ---------- Fonctions d'appartenance et ensembles flous ----------


def trapezoid(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    up = np.divide(x - a, b - a, out=np.ones_like(x), where=b > a)
    down = np.divide(d - x, d - c, out=np.ones_like(x), where=d > c)
    return np.clip(np.minimum(up, down), 0, 1) * ((x >= a) & (x <= d))


def triangle(x, a, b, c):
    return trapezoid(x, a, b, b, c)


def gaussian(x, m, sigma):
    return np.exp(-((np.asarray(x, dtype=float) - m) ** 2) / (2 * sigma ** 2))


def alpha_cut_trapezoid(a, b, c, d, alpha):
    """[a + α(b - a), d - α(d - c)]."""
    return a + alpha * (b - a), d - alpha * (d - c)


# ---------- Arithmétique des nombres flous (triangulaires ou trapézoïdaux, cours p.109) ----------


def f_add(A, B):
    return tuple(np.add(A, B))


def f_sub(A, B):
    return tuple(np.subtract(A, B[::-1]))


def f_mul(A, B):
    return tuple(np.multiply(A, B))


def f_div(A, B):
    if min(B) <= 0:
        raise ValueError("La division floue exige un diviseur strictement positif")
    return tuple(np.divide(A, B[::-1]))


def f_scale(lam, A):
    return tuple(lam * np.asarray(A, dtype=float))


# ---------- Intervalles (cours p.110-111) ----------


def i_mul(A, B):
    prods = [A[0] * B[0], A[0] * B[1], A[1] * B[0], A[1] * B[1]]
    return min(prods), max(prods)


def i_div(A, B):
    if B[0] <= 0 <= B[1]:
        raise ValueError("0 ne doit pas appartenir au diviseur")
    return i_mul(A, (1 / B[1], 1 / B[0]))


# ---------- Nombres flous intuitionnistes (cours p.113-117) ----------


def ifn_add(a, b):
    return a[0] + b[0] - a[0] * b[0], a[1] * b[1]


def ifn_mul(a, b):
    return a[0] * b[0], a[1] + b[1] - a[1] * b[1]


def ifn_scale(lam, a):
    return 1 - (1 - a[0]) ** lam, a[1] ** lam


def ifn_power(a, lam):
    return a[0] ** lam, 1 - (1 - a[1]) ** lam


def ifn_score(a):
    return a[0] - a[1]


def ifn_accuracy(a):
    return a[0] + a[1]


def ifwa(alphas, weights):
    """IFWA = (1 - Π(1 - μ_j)^σ_j, Π ν_j^σ_j) ; alphas de forme (..., k, 2), poids sur l'avant-dernier axe."""
    A = np.asarray(alphas, dtype=float)
    s = np.asarray(weights, dtype=float)
    mu = 1 - np.prod((1 - A[..., 0]) ** s, axis=-1)
    nu = np.prod(A[..., 1] ** s, axis=-1)
    return np.stack([mu, nu], axis=-1)


def hesitation(mu, nu, q=1):
    """π = (1 - μ^q - ν^q)^(1/q) ; q = 1 intuitionniste, q = 2 pythagoricien, q >= 3 q-ROF."""
    rest = 1 - mu ** q - nu ** q
    if rest < -1e-12:
        raise ValueError(f"μ^{q} + ν^{q} > 1 : ce n'est pas un nombre flou de rang {q}")
    return max(rest, 0) ** (1 / q)


# ---------- Ensembles flous picturaux (cours p.122-124) ----------


def pfn_add(A, B):
    return A[0] + B[0] - A[0] * B[0], A[1] * B[1], A[2] * B[2]


def pfn_scale(lam, A):
    return 1 - (1 - A[0]) ** lam, A[1] ** lam, A[2] ** lam


# ---------- Ensembles flous hésitants (cours p.125-130) ----------


def hfe_score(h):
    return float(np.mean(h))


def hfe_deviation(h):
    h = np.asarray(h, dtype=float)
    return float(np.sqrt(((h - h.mean()) ** 2).mean()))


def hfe_compare(h1, h2):
    """+1 si h1 > h2, -1 si h1 < h2, 0 si égaux (score puis déviation)."""
    s1, s2 = hfe_score(h1), hfe_score(h2)
    if not np.isclose(s1, s2):
        return 1 if s1 > s2 else -1
    d1, d2 = hfe_deviation(h1), hfe_deviation(h2)
    if np.isclose(d1, d2):
        return 0
    return 1 if d1 < d2 else -1


def hfe_hamming(h1, h2):
    """Distance de Hamming hésitante ; le plus court est complété par sa plus grande valeur (hypothèse optimiste)."""
    a, b = sorted(h1), sorted(h2)
    l = max(len(a), len(b))
    a += [a[-1]] * (l - len(a))
    b += [b[-1]] * (l - len(b))
    return float(np.abs(np.subtract(a, b)).mean())


# ---------- MCDM floue ----------


def fuzzy_single(tfns, w, c_max):
    """Cours p.134 : normalisation (a,b,c) / c_max, agrégation Σ w x̃, défuzzification centre de gravité."""
    T = np.asarray(tfns, dtype=float) / np.asarray(c_max, dtype=float)[:, None]
    agg = np.asarray(w, dtype=float) @ T
    return T, agg, agg.mean()


# Tableau 1 du cours (Chen, 2000)
RATING_SCALE = {"VP": (0, 0, 1), "P": (0, 1, 3), "MP": (1, 3, 5), "F": (3, 5, 7),
                "MG": (5, 7, 9), "G": (7, 9, 10), "VG": (9, 10, 10)}
WEIGHT_SCALE = {"VL": (0, 0, 0.1), "L": (0, 0.1, 0.3), "ML": (0.1, 0.3, 0.5), "M": (0.3, 0.5, 0.7),
                "MH": (0.5, 0.7, 0.9), "H": (0.7, 0.9, 1), "VH": (0.9, 1, 1)}


def fuzzy_topsis(ratings, weights, types):
    """Fuzzy TOPSIS (Chen, 2000), cours p.135-139.

    ratings : (m, n, K, 3) nombres triangulaires par décideur ; weights : (n, K, 3) ; types : +1 / -1.
    """
    X = np.asarray(ratings, dtype=float).mean(axis=2)  # agrégation 1/K Σ
    W = np.asarray(weights, dtype=float).mean(axis=1)
    benefit = np.asarray(types) > 0
    c_star = X[:, :, 2].max(axis=0)
    a_minus = X[:, :, 0].min(axis=0)
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        if benefit[j]:
            R[:, j] = X[:, j] / c_star[j]
        else:
            R[:, j] = a_minus[j] / X[:, j, ::-1]
    V = R * W[None, :, :]
    d_plus = np.sqrt(((V - 1) ** 2).mean(axis=2)).sum(axis=1)  # FPIS = (1, 1, 1)
    d_minus = np.sqrt((V ** 2).mean(axis=2)).sum(axis=1)       # FNIS = (0, 0, 0)
    cc = d_minus / (d_plus + d_minus)
    return {"aggregated": X, "weights": W, "normalized": R, "weighted": V, "D+": d_plus, "D-": d_minus, "CC": cc}


def if_topsis(dm_matrices, dm_weights, dm_criteria, types):
    """TOPSIS intuitionniste (cours p.143-150).

    dm_matrices : (K, m, n, 2) ; dm_weights : (K,) ; dm_criteria : (K, n, 2) importances des critères en IFN.
    """
    M = np.moveaxis(np.asarray(dm_matrices, dtype=float), 0, -2)   # (m, n, K, 2)
    D = ifwa(M, dm_weights)                                         # (m, n, 2)
    Wc = ifwa(np.moveaxis(np.asarray(dm_criteria, dtype=float), 0, -2), dm_weights)  # (n, 2)
    mu = D[..., 0] * Wc[:, 0]
    nu = D[..., 1] + Wc[:, 1] - D[..., 1] * Wc[:, 1]
    pi = 1 - mu - nu
    benefit = np.asarray(types) > 0
    mu_p = np.where(benefit, mu.max(0), mu.min(0))
    nu_p = np.where(benefit, nu.min(0), nu.max(0))
    mu_n = np.where(benefit, mu.min(0), mu.max(0))
    nu_n = np.where(benefit, nu.max(0), nu.min(0))
    pi_p, pi_n = 1 - mu_p - nu_p, 1 - mu_n - nu_n
    n = mu.shape[1]
    s_plus = np.sqrt(((mu - mu_p) ** 2 + (nu - nu_p) ** 2 + (pi - pi_p) ** 2).sum(1) / (2 * n))
    s_minus = np.sqrt(((mu - mu_n) ** 2 + (nu - nu_n) ** 2 + (pi - pi_n) ** 2).sum(1) / (2 * n))
    return {"aggregated": D, "criteria_weights": Wc, "weighted": np.stack([mu, nu], -1),
            "I+": np.stack([mu_p, nu_p], -1), "I-": np.stack([mu_n, nu_n], -1),
            "S+": s_plus, "S-": s_minus, "C": s_minus / (s_plus + s_minus)}
