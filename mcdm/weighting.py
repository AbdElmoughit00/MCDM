"""Pondération des critères : AHP, BWM (subjectives), Entropie, CRITIC (objectives)."""
import numpy as np
from scipy.optimize import linprog

# Indice aléatoire de Saaty (table du cours, p.29) ; n = 1, 2 sont toujours cohérents.
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


def ahp(A):
    """Méthode approximative du cours : sommes de colonnes, normalisation, moyenne des lignes, test de cohérence."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or (A <= 0).any():
        raise ValueError("La matrice de comparaison doit être carrée et strictement positive")
    if not np.allclose(A * A.T, 1, atol=1e-2):
        raise ValueError("La matrice doit être réciproque : a_ji = 1 / a_ij")
    s = A.sum(axis=0)
    N = A / s
    w = N.mean(axis=1)
    lam = float(s @ w)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    if n > 10:
        raise ValueError("La table RI du cours s'arrête à n = 10")
    ri = RI[n]
    cr = ci / ri if ri else 0.0
    return {"col_sums": s, "normalized": N, "weights": w, "lambda_max": lam, "CI": ci, "RI": ri, "CR": cr}


def ahp_global(criteria_weights, local_priorities):
    """local_priorities[j] = priorités des alternatives selon le critère j ; renvoie les priorités finales."""
    L = np.column_stack(local_priorities)
    return L @ np.asarray(criteria_weights, dtype=float)


def bwm(best, worst, a_bo, a_ow):
    """BWM linéaire (cours p.38-43) : min ξ s.c. |w_B - a_Bj w_j| <= ξ, |w_j - a_jW w_W| <= ξ, Σw = 1."""
    a_bo, a_ow = np.asarray(a_bo, dtype=float), np.asarray(a_ow, dtype=float)
    n = len(a_bo)
    if a_bo[best] != 1 or a_ow[worst] != 1:
        raise ValueError("a_BB et a_WW doivent valoir 1")
    rows = []
    for j in range(n):
        for coef in (1, -1):
            r = np.zeros(n + 1)
            r[best] += coef
            r[j] -= coef * a_bo[j]
            r[n] = -1
            rows.append(r)
            r = np.zeros(n + 1)
            r[j] += coef
            r[worst] -= coef * a_ow[j]
            r[n] = -1
            rows.append(r)
    A_ub = np.array(rows)
    A_eq = np.r_[np.ones(n), 0].reshape(1, -1)
    c = np.r_[np.zeros(n), 1]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(rows)), A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * (n + 1))
    if not res.success:
        raise ValueError(f"Programme linéaire non résolu : {res.message}")
    return {"weights": res.x[:n], "xi": res.x[n], "A_ub": A_ub, "c": c}


def minmax(X, types):
    """Normalisation min-max orientée (cours CRITIC p.47) ; types[j] = +1 bénéfice, -1 coût. Colonne constante -> 0."""
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    benefit = np.asarray(types) > 0
    return np.where(benefit, (X - lo) / span, (hi - X) / span)


def entropy(D):
    """Entropie (cours p.44-45) sur une matrice déjà normalisée et positive D (m alternatives x n critères)."""
    D = np.asarray(D, dtype=float)
    if (D < 0).any():
        raise ValueError("L'entropie exige des valeurs positives : normaliser d'abord (min-max)")
    m = D.shape[0]
    col = D.sum(axis=0)
    P = np.divide(D, col, out=np.full_like(D, 1 / m), where=col > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -plogp.sum(axis=0) / np.log(m)
    div = 1 - E
    return {"P": P, "E": E, "diversification": div, "weights": div / div.sum()}


def critic(X, types):
    """CRITIC (cours p.46-49) : écart type x indépendance (1 - |ρ|)."""
    R = minmax(X, types)
    rho = np.corrcoef(R, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0)  # colonne constante : aucune corrélation exploitable
    sigma = R.std(axis=0, ddof=1)
    independence = (1 - np.abs(rho)).sum(axis=0)
    C = sigma * independence
    if C.sum() == 0:
        raise ValueError("Toutes les colonnes sont constantes : CRITIC ne peut pas discriminer")
    return {"normalized": R, "corr": rho, "sigma": sigma, "independence": independence, "C": C, "weights": C / C.sum()}
