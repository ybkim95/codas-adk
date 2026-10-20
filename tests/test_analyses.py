"""The agents' analysis tools (codas.core.analyses) on synthetic data with planted structure: each must
recover what was planted and stay quiet where nothing was."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codas.core import analyses


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"age": rng.normal(50, 12, n), "female": rng.integers(0, 2, n).astype(float),
                      "x": rng.normal(size=n), "noise": rng.normal(size=n), "u": rng.normal(size=n)})
    return f, rng


def test_rebuild_feature_matches_the_engine_names():
    f = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 0.0, 2.0]})
    assert analyses.rebuild_feature(f, "a_over_b", {"components": ["a", "b"]}).tolist()[::2] == [0.5, 2.0]
    assert np.isnan(analyses.rebuild_feature(f, "a_over_b", {"components": ["a", "b"]})[1])
    prod = analyses.rebuild_feature(f, "a_x_b", {"components": ["a", "b"]})
    za, zb = (f.a - f.a.mean()) / f.a.std(ddof=0), (f.b - f.b.mean()) / f.b.std(ddof=0)
    assert np.allclose(prod, za * zb)
    assert analyses.rebuild_feature(f, "d", {"op": "difference", "a": "a", "b": "b"}).tolist() == [-1.0, 2.0, 2.0]
    assert analyses.rebuild_feature(f, "a", {}).tolist() == [1.0, 2.0, 4.0]
    assert analyses.rebuild_feature(f, "missing", {}) is None


def test_survival_screen_ranks_the_real_hazard_and_corrects_for_multiplicity():
    f, rng = _frame()
    hazard = np.exp(0.5 * f.x + 0.03 * (f.age - 50))
    t = rng.exponential(1 / (0.1 * hazard))
    c = rng.uniform(0, 15, len(f))
    f["time"], f["event"] = np.minimum(t, c), (t <= c).astype(float)
    rows = analyses.survival_screen(f, ["noise", "x", "u"], "event", "time", confounders=["age"])
    assert [r["feature"] for r in rows][0] == "x"
    x = rows[0]
    assert 1.4 < x["hr_per_sd"] < 1.9 and x["q"] < 1e-6 and x["ci95"][0] > 1
    assert all(r["q"] > 0.01 for r in rows[1:])


def test_subgroup_analysis_finds_planted_heterogeneity_and_not_absent_one():
    f, rng = _frame()
    f["y"] = np.where(f.female == 1, 0.6, 0.0) * f.x + 0.02 * f.age + rng.normal(size=len(f))
    het = analyses.subgroup_analysis(f, "x", "y", "female", confounders=["age"])
    strata = {s["stratum"]: s for s in het["strata"]}
    assert set(strata) == {"female=0", "female=1"}
    assert strata["female=1"]["rho"] > 0.3 and abs(strata["female=0"]["rho"]) < 0.08
    assert het["heterogeneity_p"] < 1e-6 and het["i2"] > 0.9
    f["y2"] = 0.4 * f.x + rng.normal(size=len(f))
    flat = analyses.subgroup_analysis(f, "x", "y2", "age", confounders=["female"])
    assert len(flat["strata"]) == 3 and flat["strata"][0]["stratum"].startswith("age tertile 1")
    assert flat["heterogeneity_p"] > 0.01


def test_multivariable_model_reports_independent_effects_and_incremental_value():
    f, rng = _frame()
    f["y"] = 0.05 * f.age + 0.5 * f.x + rng.normal(size=len(f))
    f.loc[rng.random(len(f)) < 0.3, "age"] = np.nan  # a covariate often missing: imputed, not dropped
    fit = analyses.multivariable_model(f, ["x", "noise"], "y", confounders=["age", "female"], imputations=5)
    coef = {c["term"]: c for c in fit["coefficients"]}
    assert coef["x"]["p"] < 1e-10 and abs(coef["noise"]["estimate"]) < 0.06
    assert fit["n"] == len(f) and fit["metric"] == "r2"
    inc = fit["incremental"]
    assert inc["full"] > inc["base"] and inc["delta_ci95"][0] > 0.05
    f["b"] = (rng.random(len(f)) < 1 / (1 + np.exp(-(0.8 * f.x - 1)))).astype(float)
    logit = analyses.multivariable_model(f, ["x"], "b", confounders=["female"])
    assert logit["metric"] == "auc" and logit["incremental"]["delta_ci95"][0] > 0
    assert {c["term"]: c for c in logit["coefficients"]}["x"]["odds_ratio_per_sd"] > 1.8


def test_multivariable_survival_model_uses_cox_and_concordance():
    f, rng = _frame()
    t = rng.exponential(1 / (0.1 * np.exp(0.6 * f.x)))
    c = rng.uniform(0, 15, len(f))
    f["time"], f["event"] = np.minimum(t, c), (t <= c).astype(float)
    fit = analyses.multivariable_model(f, ["x", "noise"], "event", confounders=["age"], time="time")
    assert fit["metric"] == "c_index" and fit["incremental"]["full"] > fit["incremental"]["base"] + 0.05
    assert {c["term"]: c for c in fit["coefficients"]}["x"]["hr_per_sd"] > 1.5


def test_concordance_index_hand_computed():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 0, 1])
    risk = np.array([4.0, 3.0, 2.0, 3.0])
    # comparable pairs (i has the event, t_j > t_i): (0,1)(0,2)(0,3) (1,2)(1,3): concordant when
    # risk_i > risk_j: 3 + 1 (1,2) + tie (1,3) = 4.5 of 5
    assert analyses.concordance(time, event, risk) == pytest.approx(4.5 / 5)


def test_nonlinear_model_detects_a_u_shape_the_linear_model_misses():
    f, rng = _frame(4000)
    f["y"] = f.x ** 2 + 0.3 * f.noise + rng.normal(scale=0.5, size=len(f))
    fit = analyses.nonlinear_model(f, ["x", "noise", "u"], "y", confounders=["age"])
    assert fit["nonlinear"] > fit["linear"] + 0.3
    shapes = {s["feature"]: s["shape"] for s in fit["shapes"]}
    assert shapes["x"] == "U-shaped" and shapes["noise"] in {"increasing", "flat"}
    assert fit["importance"][0]["feature"] == "x"


def test_spline_dose_response_tests_nonlinearity():
    f, rng = _frame(4000)
    f["y"] = np.where(f.x > 0.5, 1.0, 0.0) * (f.x - 0.5) * 2 + rng.normal(size=len(f))  # a threshold
    curve = analyses.spline_dose_response(f, "x", "y", confounders=["age"])
    assert curve["nonlinearity_p"] < 1e-6 and len(curve["curve"]) == 5
    f["y2"] = 0.5 * f.x + rng.normal(size=len(f))
    assert analyses.spline_dose_response(f, "x", "y2", confounders=["age"])["nonlinearity_p"] > 0.01
    t = rng.exponential(1 / (0.1 * np.exp(0.5 * f.x)))
    c = rng.uniform(0, 15, len(f))
    f["time"], f["event"] = np.minimum(t, c), (t <= c).astype(float)
    cox = analyses.spline_dose_response(f, "x", "event", confounders=["age"], time="time")
    assert cox["measure"].startswith("hazard ratio") and cox["nonlinearity_p"] > 0.01
    assert cox["curve"][-1]["effect"] > 1.5  # 90th vs 50th percentile
