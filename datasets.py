"""Jeux de données : exercices du cours (transcrits des diapositives) et cas réalistes construits (graine fixe).

Chaque problème multicritère est un dict : matrix (DataFrame alternatives x critères), types (+1 bénéfice / -1 coût),
weights (optionnel), source (d'où viennent les chiffres).
"""
import numpy as np
import pandas as pd


def problem(rows, index, columns, types, weights=None, source=""):
    return {"matrix": pd.DataFrame(rows, index=index, columns=columns, dtype=float),
            "types": list(types), "weights": None if weights is None else list(weights), "source": source}


# ---------------- AHP ----------------

AHP_CAR = {
    "criteria": ["Coût", "Confort", "Sécurité"],
    "alternatives": ["Voiture 1", "Voiture 2"],
    "A": [[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]],
    "local": [[[1, 7], [1 / 7, 1]], [[1, 0.2], [5, 1]], [[1, 9], [1 / 9, 1]]],
    "source": "Exercice du cours : achat d'une voiture",
}

AHP_HOSPITAL = {
    "criteria": ["Population desservie", "Accessibilité", "Coût du terrain", "Infrastructures"],
    "alternatives": ["Kénitra", "Khémisset", "Settat", "Béni Mellal"],
    "A": [[1, 3, 5, 3], [1 / 3, 1, 3, 1], [1 / 5, 1 / 3, 1, 1 / 3], [1 / 3, 1, 3, 1]],
    "local": [
        [[1, 3, 2, 1 / 2], [1 / 3, 1, 1 / 2, 1 / 4], [1 / 2, 2, 1, 1 / 3], [2, 4, 3, 1]],
        [[1, 2, 1 / 2, 3], [1 / 2, 1, 1 / 3, 2], [2, 3, 1, 4], [1 / 3, 1 / 2, 1 / 4, 1]],
        [[1, 1 / 3, 1 / 2, 2], [3, 1, 2, 4], [2, 1 / 2, 1, 3], [1 / 2, 1 / 4, 1 / 3, 1]],
        [[1, 3, 2, 1], [1 / 3, 1, 1 / 2, 1 / 3], [1 / 2, 2, 1, 1 / 2], [1, 3, 2, 1]],
    ],
    "source": "Cas construit à partir de l'exemple de l'introduction (hôpital, 4 villes candidates)",
}

# Même problème, mais jugements contradictoires : Population > Accessibilité > Coût, et pourtant Coût >> Population
AHP_HOSPITAL_INCONSISTENT = dict(AHP_HOSPITAL, A=[[1, 3, 1 / 5, 3], [1 / 3, 1, 5, 1], [5, 1 / 5, 1, 1 / 3], [1 / 3, 1, 3, 1]],
                                 source="Variante volontairement incohérente pour illustrer CR > 0,1")

# ---------------- BWM ----------------

BWM_SUPPLIER = {"criteria": ["Qualité", "Prix", "Délai de livraison", "Engagement RSE"],
                "best": 0, "worst": 3, "BO": [1, 3, 4, 8], "OW": [8, 5, 3, 1],
                "source": "Exercice du cours : choix d'un fournisseur"}

BWM_FINTECH = {"criteria": ["Sécurité", "Coût", "Évolutivité", "Conformité réglementaire"],
               "best": 0, "worst": 2, "BO": [1, 4, 6, 2], "OW": [6, 2, 1, 4],
               "source": "Cas construit : choix d'une plateforme Fintech"}

# ---------------- Matrices de décision ----------------

FINANCE_COLUMNS = ["Rendement moyen (%)", "Écart-type (%)", "Volume", "Bêta", "EPS", "ROE (%)",
                   "P/B", "P/S", "ROA (%)"]
FINANCE = problem(
    [[0.2137, 2.19, 1382.93, 0.51, 16.72, 9.36, 0.95, 0.66, 5.20],
     [-0.1037, 1.94, 2302.18, 0.68, 55.08, 3.63, 0.28, 0.67, 1.90],
     [-0.3592, 2.67, 864.13, 0.37, 17.17, 5.63, 1.47, 3.24, 3.29],
     [0.2753, 3.06, 420.82, 1.12, 82.84, 12.28, 1.28, 0.54, 5.39],
     [0.1222, 5.90, 762.22, 0.06, 58.38, 14.42, 1.58, 2.44, 5.70],
     [-1.2446, 7.06, 4504.48, 0.68, -3.29, -38.62, 2.29, 0.38, -5.35],
     [-0.5443, 5.65, 10656.87, 1.42, 0.50, 5.23, 1.98, 0.32, 0.94],
     [-0.1648, 2.69, 162.01, 0.70, 50.98, 28.82, 6.81, 1.08, 8.26],
     [-0.1194, 1.60, 7220.56, 0.46, 9.85, 6.40, 1.07, 1.74, 5.11],
     [0.1794, 5.22, 13159.67, 1.77, 0.95, 12.02, 0.46, 0.72, 1.50],
     [-0.2807, 2.34, 374.91, 0.81, 42.03, 4.35, 0.74, 0.65, 2.91],
     [-0.3532, 2.44, 194.47, 0.79, 22.17, 4.57, 0.95, 0.62, 2.51],
     [0.1381, 2.79, 311.65, 1.33, 14.82, 10.33, 1.95, 2.78, 5.50],
     [-0.4341, 5.26, 28452.85, 1.18, -0.85, -303.61, 9.70, 0.38, -8.96],
     [-0.0184, 2.54, 1419.53, 1.03, 3.35, 0.82, 0.66, 0.47, 0.47],
     [-0.0424, 2.66, 31328.13, 0.70, 1.94, 9.74, 2.19, 3.14, 4.91],
     [-0.4791, 8.55, 25731.26, 1.87, 2.18, 14.86, 0.94, 0.48, 9.08],
     [-0.0144, 3.73, 6927.05, 1.47, 3.26, 5.76, 0.93, 3.11, 0.82]],
    ["ADPL", "ADRS2", "ARNT", "ATGR", "ATPL", "DDJH", "DLKV", "ERNT", "HT", "INGR", "KOEI", "KRAS",
     "MAIS", "OPTE", "PODR", "RIVP", "VLEN", "ZABA"],
    FINANCE_COLUMNS, [1, -1, 1, -1, 1, 1, -1, -1, 1],
    weights=[0.0370, 0.0189, 0.0533, 0.0259, 0.2182, 0.3070, 0.0764, 0.1543, 0.1089],
    source="Données réelles de Vuković et al. (2020), 18 actions")

CRITIC_MACHINES = problem(
    [[30, 0.1, 1, 20], [100, 0.7, 1, 40], [50, 1, 2, 10], [300, 2, 3, 35]],
    ["A1", "A2", "A3", "A4"],
    ["Épaisseur max. (C1)", "Largeur de coupe min. (C2)", "Qualité de découpe (C3)", "Coût de maintenance (C4)"],
    [-1, -1, 1, -1], source="Exercice du cours : machines de découpage")

WASPAS_MACHINES = problem(
    [[0.035, 847, 0.335, 1.760, 0.590], [0.027, 834, 0.335, 1.680, 0.665],
     [0.037, 808, 0.590, 2.400, 0.500], [0.028, 821, 0.500, 1.590, 0.410]],
    ["A1", "A2", "A3", "A4"], ["C1", "C2", "C3", "C4", "C5"], [-1, -1, 1, -1, 1],
    weights=[0.331, 0.181, 0.369, 0.072, 0.047], source="Exercice du cours : choix d'une machine")

INVESTMENTS = problem(
    [[12.4, 14.2, 4.5, 8, 6], [9.8, 17.5, 3.2, 6, 7], [15.1, 11.8, 5.8, 9, 4],
     [7.2, 19.4, 2.6, 5, 8], [11.0, 13.1, 4.0, 7, 5], [13.6, 12.5, 6.2, 9, 3]],
    ["Usine de dessalement", "Ferme solaire", "Plateforme logistique", "Application mobile",
     "Extension d'usine", "Centre de données"],
    ["VAN (M DH)", "TRI (%)", "Délai de récupération (ans)", "Alignement stratégique (/10)", "Risque (/10)"],
    [1, 1, -1, 1, -1], weights=[0.30, 0.20, 0.15, 0.20, 0.15],
    source="Cas construit : budgétisation des capitaux, chiffres plausibles")

TOPSIS_CARS = problem(
    [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]], ["M1", "M2", "M3", "M4"],
    ["Style", "Fiabilité", "Économie de carburant", "Coût"], [1, 1, 1, -1], weights=[0.2, 0.1, 0.4, 0.3],
    source="Exercice du cours : choix d'une voiture")

VIKOR_MACHINES = problem([[1, 3000], [2, 3750], [5, 4500]], ["A1", "A2", "A3"], ["C1", "C2"], [-1, 1],
                         weights=[0.5, 0.5], source="Exercice du cours : choix d'une machine")

SUPPLIERS6 = problem(
    [[82, 4.2, 12, 7], [75, 3.6, 9, 8], [90, 5.1, 15, 6], [68, 3.1, 7, 5], [85, 4.6, 10, 9], [79, 3.9, 14, 6]],
    ["F1", "F2", "F3", "F4", "F5", "F6"],
    ["Qualité (/100)", "Prix unitaire (€)", "Délai (jours)", "Service (/10)"], [1, -1, -1, 1],
    weights=[0.35, 0.30, 0.15, 0.20], source="Cas construit : 6 fournisseurs")

# Rangs publiés dans le cours (A1 à A6)
COURSE_RANKS = {"TOPSIS": [2, 1, 3, 5, 4, 6], "VIKOR": [1, 2, 4, 3, 5, 6], "PROMETHEE II": [2, 1, 3, 6, 4, 5]}

# ---------------- Alternative unique ----------------

PAST_DECISIONS = pd.DataFrame({"Score S": [0.85, 0.78, 0.71, 0.69, 0.62, 0.54],
                               "Décision": ["Accepter"] * 3 + ["Rejeter"] * 3},
                              index=["A1", "A2", "A3", "A4", "A5", "A6"])

CREDIT_CRITERIA = ["Revenu mensuel (DH)", "Taux d'endettement (%)", "Ancienneté emploi (ans)",
                   "Incidents de paiement", "Apport personnel (%)"]
CREDIT_TYPES = [1, -1, 1, -1, 1]
CREDIT_BOUNDS = {"min": [3000, 0, 0, 0, 0], "max": [40000, 60, 25, 6, 50]}
CREDIT_WEIGHTS = [0.25, 0.25, 0.15, 0.20, 0.15]


def credit_portfolio(n=300, seed=7):
    """Dossiers de crédit synthétiques ; la décision passée dépend des critères + bruit (comité humain imparfait)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        CREDIT_CRITERIA[0]: np.round(rng.lognormal(9.4, 0.5, n), -2).clip(3000, 40000),
        CREDIT_CRITERIA[1]: rng.uniform(5, 55, n).round(1),
        CREDIT_CRITERIA[2]: rng.gamma(2, 3, n).round(1).clip(0, 25),
        CREDIT_CRITERIA[3]: rng.poisson(0.8, n).clip(0, 6),
        CREDIT_CRITERIA[4]: rng.uniform(0, 40, n).round(1),
    })
    z = (np.log(df.iloc[:, 0]) - 9.4) * 2 - (df.iloc[:, 1] - 30) / 10 + df.iloc[:, 2] / 6 \
        - df.iloc[:, 3] * 0.9 + df.iloc[:, 4] / 15 + rng.normal(0, 0.8, n)
    df["Accepté"] = (z > 0).astype(int)
    df.index = [f"D{i:03d}" for i in range(1, n + 1)]
    return df


# ---------------- Flou ----------------

FUZZY_SINGLE = {"criteria": ["ROA (%)", "Liquidité"], "weights": [0.6, 0.4],
                "tfn": [[1.0, 1.2, 1.5], [110, 120, 130]], "c_max": [1.5, 130],
                "source": "Exemple du cours : évaluation floue d'une seule alternative"}

FUZZY_TOPSIS_HIRING = {
    "alternatives": ["A1", "A2", "A3"],
    "criteria": ["Stabilité émotionnelle", "Communication orale", "Personnalité", "Expérience passée", "Confiance en soi"],
    "types": [1, 1, 1, 1, 1],
    "decision_makers": ["DM1", "DM2", "DM3"],
    # ratings[alternative][décideur][critère]
    "ratings": [[["MG", "G", "F", "VG", "F"], ["G", "MG", "G", "G", "F"], ["MG", "F", "G", "VG", "F"]],
                [["G", "VG", "VG", "VG", "VG"], ["G", "VG", "VG", "VG", "MG"], ["MG", "VG", "G", "VG", "G"]],
                [["VG", "MG", "G", "G", "G"], ["G", "G", "MG", "VG", "G"], ["F", "VG", "VG", "MG", "MG"]]],
    # weights[décideur][critère]
    "weights": [["H", "VH", "VH", "VH", "M"], ["VH", "VH", "H", "VH", "MH"], ["MH", "VH", "H", "VH", "MH"]],
    "source": "Exercice du cours : recrutement d'un ingénieur (Chen, 2000)",
}

IF_CLOUD = {
    "alternatives": ["Fournisseur Atlas", "Fournisseur Borea", "Fournisseur Cedre", "Fournisseur Dunes"],
    "criteria": ["Sécurité", "Coût", "Évolutivité", "Conformité"],
    "types": [1, -1, 1, 1],
    "decision_makers": ["DSI", "Responsable sécurité", "Directeur financier"],
    "dm_weights": [0.40, 0.35, 0.25],
    # (μ, ν) par décideur, alternative, critère
    "matrices": [
        [[(0.7, 0.2), (0.5, 0.4), (0.6, 0.3), (0.8, 0.1)], [(0.6, 0.3), (0.3, 0.6), (0.8, 0.1), (0.6, 0.3)],
         [(0.8, 0.1), (0.7, 0.2), (0.5, 0.4), (0.7, 0.2)], [(0.5, 0.4), (0.2, 0.7), (0.7, 0.2), (0.5, 0.3)]],
        [[(0.8, 0.1), (0.5, 0.3), (0.5, 0.4), (0.7, 0.2)], [(0.5, 0.4), (0.4, 0.5), (0.7, 0.2), (0.5, 0.4)],
         [(0.9, 0.05), (0.6, 0.3), (0.6, 0.3), (0.8, 0.1)], [(0.4, 0.5), (0.3, 0.6), (0.6, 0.3), (0.4, 0.4)]],
        [[(0.6, 0.3), (0.6, 0.3), (0.6, 0.2), (0.7, 0.2)], [(0.6, 0.2), (0.2, 0.7), (0.7, 0.2), (0.6, 0.3)],
         [(0.7, 0.2), (0.8, 0.1), (0.5, 0.3), (0.6, 0.3)], [(0.6, 0.3), (0.1, 0.8), (0.8, 0.1), (0.5, 0.4)]],
    ],
    # importance des critères (μ, ν) par décideur
    "criteria_ifn": [[(0.9, 0.05), (0.6, 0.3), (0.5, 0.4), (0.7, 0.2)],
                     [(0.95, 0.03), (0.4, 0.5), (0.5, 0.3), (0.8, 0.1)],
                     [(0.7, 0.2), (0.9, 0.05), (0.6, 0.3), (0.6, 0.3)]],
    "source": "Cas construit : choix d'un fournisseur cloud, 3 décideurs, nombres flous intuitionnistes",
}
