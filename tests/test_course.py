"""Vérifie le moteur contre les résultats chiffrés donnés dans le cours (et quelques invariants)."""
import numpy as np
import pytest

import datasets as ds
from mcdm import fuzzy, ranking, robustness, single, weighting


def test_ahp_car():
    res = weighting.ahp(ds.AHP_CAR["A"])
    assert res["weights"].sum() == pytest.approx(1)
    assert np.argmax(res["weights"]) == 0  # le coût domine
    assert res["CR"] < 0.1
    local = [weighting.ahp(L)["weights"] for L in ds.AHP_CAR["local"]]
    final = weighting.ahp_global(res["weights"], local)
    assert final.sum() == pytest.approx(1)
    assert final[0] > final[1]  # voiture 1


def test_ahp_inconsistent_detected():
    assert weighting.ahp(ds.AHP_HOSPITAL["A"])["CR"] < 0.1
    assert weighting.ahp(ds.AHP_HOSPITAL_INCONSISTENT["A"])["CR"] > 0.1


def test_ahp_rejects_non_reciprocal():
    with pytest.raises(ValueError):
        weighting.ahp([[1, 3], [3, 1]])


def test_bwm_supplier_course_values():
    b = ds.BWM_SUPPLIER
    res = weighting.bwm(b["best"], b["worst"], b["BO"], b["OW"])
    assert res["xi"] == pytest.approx(0.08, abs=5e-3)  # ξ* du cours p.42
    assert res["weights"] == pytest.approx([0.5629, 0.2156, 0.1617, 0.0599], abs=1e-3)
    # Les poids imprimés p.42 ne respectent pas le modèle : leur écart maximal dépasse ξ*
    w = np.array([0.552, 0.207, 0.155, 0.086])
    gap = max(np.abs(w[0] - np.array(b["BO"]) * w).max(), np.abs(w - np.array(b["OW"]) * w[3]).max())
    assert gap > res["xi"] + 0.1


def test_entropy_and_critic_sum_to_one():
    X = ds.FINANCE["matrix"].to_numpy()
    w_e = weighting.entropy(weighting.minmax(X, ds.FINANCE["types"]))["weights"]
    w_c = weighting.critic(X, ds.FINANCE["types"])["weights"]
    assert w_e.sum() == pytest.approx(1) and w_c.sum() == pytest.approx(1)
    assert (w_e >= 0).all() and (w_c >= 0).all()


def test_entropy_uniform_column_has_zero_weight():
    D = np.array([[1, 1.0], [1, 5.0], [1, 9.0]])
    w = weighting.entropy(D)["weights"]
    assert w[0] == pytest.approx(0)


def test_waspas_lambda_extremes():
    p = ds.WASPAS_MACHINES
    res0 = ranking.waspas(p["matrix"], p["weights"], p["types"], lam=0)
    res1 = ranking.waspas(p["matrix"], p["weights"], p["types"], lam=1)
    assert res0["WASPAS"] == pytest.approx(res0["WPM"])
    assert res1["WASPAS"] == pytest.approx(res1["WSM"])


def test_topsis_cars():
    p = ds.TOPSIS_CARS
    rc = ranking.topsis(p["matrix"], p["weights"], p["types"])["RC"]
    assert ((rc >= 0) & (rc <= 1)).all()
    assert rc == pytest.approx([0.4545, 0.5678, 0.3703, 0.5527], abs=1e-4)  # recalculé à la main : M2


def test_vikor_course_exercise():
    p = ds.VIKOR_MACHINES
    res = ranking.vikor(p["matrix"], p["weights"], p["types"])
    assert res["S"] == pytest.approx([0.5, 0.375, 0.5])
    assert res["R"] == pytest.approx([0.5, 0.25, 0.5])
    assert res["Q"] == pytest.approx([1, 0, 1])
    assert res["compromise"] == [1] and res["C1"] and res["C2"]


def test_spearman_course_matrix():
    r = ds.COURSE_RANKS
    assert robustness.spearman(r["TOPSIS"], r["VIKOR"]) == pytest.approx(0.77, abs=5e-3)
    assert robustness.spearman(r["TOPSIS"], r["PROMETHEE II"]) == pytest.approx(0.94, abs=5e-3)
    assert robustness.spearman(r["VIKOR"], r["PROMETHEE II"]) == pytest.approx(0.60, abs=5e-3)


def test_roc_course_table():
    df = ds.PAST_DECISIONS
    y = (df["Décision"] == "Accepter").to_numpy()
    t = single.threshold_table(df["Score S"], y, thresholds=[0.75, 0.65, 0.60])
    assert t[:, 1] == pytest.approx([2 / 3, 1, 1])        # TPR
    assert t[:, 2] == pytest.approx([0, 1 / 3, 2 / 3])    # FPR
    assert t[:, 3] == pytest.approx([1 / 3, 1 / 3, 2 / 3])  # d
    assert t[:, 4] == pytest.approx([2 / 3, 2 / 3, 1 / 3])  # J


def test_maut_bounds():
    u, s = single.maut([20000, 30, 5, 1, 20], ds.CREDIT_WEIGHTS, ds.CREDIT_BOUNDS["min"], ds.CREDIT_BOUNDS["max"],
                       ds.CREDIT_TYPES)
    assert 0 <= s <= 1 and ((u >= 0) & (u <= 1)).all()


def test_fuzzy_single_course():
    f = ds.FUZZY_SINGLE
    _, agg, s = fuzzy.fuzzy_single(f["tfn"], f["weights"], f["c_max"])
    assert agg == pytest.approx([0.74, 0.85, 1.00], abs=5e-3)
    assert s == pytest.approx(0.86, abs=5e-3)


def test_membership_age_moyen():
    assert fuzzy.trapezoid(30, 25, 35, 55, 65) == pytest.approx(0.5)
    assert fuzzy.trapezoid(45, 25, 35, 55, 65) == pytest.approx(1)


def test_hesitant_course_comparison():
    h1, h2, h3 = [0.2, 0.3, 0.5, 0.8], [0.4, 0.6, 0.8], [0.3, 0.45, 0.6]
    assert fuzzy.hfe_compare(h2, h3) == 1 and fuzzy.hfe_compare(h3, h1) == 1


def test_fuzzy_topsis_chen_order():
    h = ds.FUZZY_TOPSIS_HIRING
    R = [[[fuzzy.RATING_SCALE[x] for x in dm] for dm in alt] for alt in h["ratings"]]  # (m, K, n, 3)
    R = np.transpose(np.array(R, dtype=float), (0, 2, 1, 3))
    W = np.transpose(np.array([[fuzzy.WEIGHT_SCALE[x] for x in dm] for dm in h["weights"]], dtype=float), (1, 0, 2))
    cc = fuzzy.fuzzy_topsis(R, W, h["types"])["CC"]
    assert list(np.argsort(-cc)) == [1, 2, 0]  # A2 > A3 > A1 (Chen, 2000)


def test_if_topsis_cloud():
    c = ds.IF_CLOUD
    res = fuzzy.if_topsis(c["matrices"], c["dm_weights"], c["criteria_ifn"], c["types"])
    assert ((res["C"] >= 0) & (res["C"] <= 1)).all()
    assert (res["aggregated"].sum(-1) <= 1 + 1e-9).all()


def test_ifwa_identity():
    a = (0.6, 0.3)
    assert fuzzy.ifwa([a, a], [0.5, 0.5]) == pytest.approx(a)
