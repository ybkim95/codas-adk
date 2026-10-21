"""The statistical and ML method pools (codas.core.methods), ported from the paper's CoDaS: each method
recovers planted structure, and the default exploration order and request validation behave."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codas.core import methods


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"age": rng.normal(50, 10, n), "female": rng.integers(0, 2, n).astype(float),
                      "x": rng.normal(size=n), "noise": rng.normal(size=n)})
    return f, rng


def test_registry_lists_both_pools_with_descriptions():
    names = methods.available(survival=False)
    assert {"partial_correlation", "robust_regression", "mutual_information", "subgroup_stratified", "mediation",
            "propensity_stratification"} <= set(names["stat"])
    assert {"elastic_net", "ridge", "random_forest", "gradient_boosting"} <= set(names["ml"])
    assert "survival_screen" in methods.available(survival=True)["stat"]
    assert all(methods.describe(m) for pool in names.values() for m in pool)


def test_next_methods_follow_requests_then_the_default_order():
    first = methods.next_methods(explored=[], requested=[], survival=False)
    assert [m["method"] for m in first] == ["partial_correlation", "gradient_boosting"]
    later = methods.next_methods(explored=["partial_correlation", "gradient_boosting"],
                                 requested=[{"method": "mediation", "mediator": "age"}], survival=False)
    assert later[0] == {"method": "mediation", "mediator": "age"} and later[1]["method"] == "elastic_net"
    everything = [m for pool in methods.available(False).values() for m in pool]
    assert methods.next_methods(everything, [], False) == []
    assert methods.exhausted(everything, survival=False)


def test_robust_regression_resists_outliers():
    f, rng = _frame()
    f["y"] = 0.5 * f.x + rng.normal(size=len(f))
    f.loc[f.index[:40], "y"] = 60.0  # gross outliers
    out = methods.run("robust_regression", f, ["x", "noise"], "y", confounders=["age"])
    rows = {r["feature"]: r for r in out["results"]}
    assert 0.4 < rows["x"]["estimate"] < 0.6 and rows["x"]["ci95"][0] > 0.3
    assert rows["noise"]["q"] > 0.05


def test_mutual_information_finds_a_nonmonotone_dependence_spearman_misses():
    f, rng = _frame(3000)
    f["y"] = f.x ** 2 + rng.normal(scale=0.5, size=len(f))
    out = methods.run("mutual_information", f, ["x", "noise"], "y", confounders=["age"])
    rows = {r["feature"]: r for r in out["results"]}
    assert rows["x"]["p"] < 0.05 and rows["noise"]["p"] > 0.05 and rows["x"]["mi"] > rows["noise"]["mi"]


def test_mediation_recovers_the_indirect_effect():
    f, rng = _frame(4000)
    f["m"] = 0.6 * f.x + rng.normal(size=len(f))
    f["y"] = 0.5 * f.m + 0.2 * f.x + rng.normal(size=len(f))
    out = methods.run("mediation", f, ["x"], "y", confounders=["age"], mediator="m")
    r = out["results"][0]
    sd_x = f.x.std(ddof=0)
    assert r["indirect"] == pytest.approx(0.3 * sd_x, abs=0.05) and r["indirect_ci95"][0] > 0
    assert 0.5 < r["proportion_mediated"] < 0.8
    assert "error" in methods.run("mediation", f, ["x"], "y", confounders=["age"])  # needs a mediator


def test_propensity_stratification_removes_confounding():
    f, rng = _frame(5000)
    f["x"] = 0.8 * (f.age - 50) / 10 + rng.normal(size=len(f))  # exposure driven by age
    f["y"] = 0.05 * f.age + rng.normal(size=len(f))              # no effect of x
    crude = f.loc[f.x > f.x.median(), "y"].mean() - f.loc[f.x <= f.x.median(), "y"].mean()
    out = methods.run("propensity_stratification", f, ["x"], "y", confounders=["age"])
    r = out["results"][0]
    assert crude > 0.3 and abs(r["effect"]) < 0.12 and r["ci95"][0] < 0 < r["ci95"][1]


def test_ml_methods_use_nested_cv_and_a_permutation_null():
    f, rng = _frame(1500)
    f["y"] = 0.6 * f.x + rng.normal(size=len(f))
    for name in ("elastic_net", "random_forest"):
        out = methods.run(name, f, ["x", "noise"], "y", confounders=["age"])
        assert out["metric"] == "r2" and out["value"] > 0.15 and out["null_p"] < 0.15, name
        assert out["outer_folds"] == 5 and out["best_params"] and out["importance"][0]["feature"] == "x"
    f["b"] = (rng.random(len(f)) < 1 / (1 + np.exp(-1.5 * f.x))).astype(float)
    out = methods.run("logistic", f, ["x", "noise"], "b")
    assert out["metric"] == "auc" and out["value"] > 0.7


def test_subgroup_stratified_uses_the_requested_or_a_categorical_covariate():
    f, rng = _frame()
    f["y"] = np.where(f.female == 1, 0.6, 0.0) * f.x + rng.normal(size=len(f))
    out = methods.run("subgroup_stratified", f, ["x"], "y", confounders=["age", "female"])
    assert out["by"] == "female" and out["results"][0]["heterogeneity_p"] < 1e-6


def test_every_method_has_a_headline():
    f, rng = _frame(800)
    f["y"] = 0.5 * f.x + rng.normal(size=len(f))
    for name in ("partial_correlation", "robust_regression", "ridge"):
        assert methods.headline(methods.run(name, f, ["x"], "y", confounders=["age"]))
