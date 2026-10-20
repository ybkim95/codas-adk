"""Multiple imputation of missing adjustment covariates: chained equations with posterior draws, pooled
with Rubin's rules. A covariate missing at random given the outcome biases a complete-case analysis;
the pooled estimate should recover the truth."""

from __future__ import annotations

import numpy as np
import pytest

from codas.core.imputation import (completed_covariates, nelson_aalen, pool, pooled_cox, pooled_partial_spearman)
from codas.core.statistics import partial_spearman
from codas.core.survival import cox_ph


def _confounded(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)                       # the confounder
    x = 0.8 * c + rng.normal(size=n)             # exposure driven by it
    y = 0.6 * c + 0.25 * x + rng.normal(size=n)  # outcome: a real, modest effect of x
    return x, y, c, rng


def test_pool_follows_rubins_rules():
    est, se, df = pool(np.array([1.0, 1.2, 0.8]), np.array([0.04, 0.04, 0.04]))
    within, between = 0.04, np.var([1.0, 1.2, 0.8], ddof=1)
    assert est == pytest.approx(1.0)
    assert se == pytest.approx(np.sqrt(within + (1 + 1 / 3) * between))
    assert df > 2
    est1, se1, df1 = pool(np.array([0.5]), np.array([0.01]))  # one draw: nothing to pool
    assert (est1, se1) == (0.5, pytest.approx(0.1)) and np.isinf(df1)


def test_completed_covariates_fill_only_the_gaps_and_keep_binary_columns_binary():
    x, y, c, rng = _confounded(500)
    b = (c > 0).astype(float)
    cov = np.column_stack([c, b])
    cov[rng.random(500) < 0.4, 0] = np.nan
    cov[rng.random(500) < 0.2, 1] = np.nan
    sets = completed_covariates(cov, np.column_stack([x, y]), m=3, seed=1)
    assert len(sets) == 3
    for s in sets:
        assert np.isfinite(s).all()
        observed = np.isfinite(cov)
        assert np.array_equal(s[observed], cov[observed])
        assert set(np.unique(s[:, 1])) <= {0.0, 1.0}
    assert not np.array_equal(sets[0], sets[1])  # posterior draws differ
    again = completed_covariates(cov, np.column_stack([x, y]), m=3, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(sets, again))  # and are reproducible


def test_pooled_partial_spearman_recovers_what_complete_case_loses():
    x, y, c, rng = _confounded()
    full_rho, _, _ = partial_spearman(x, y, c.reshape(-1, 1))
    # Missing at random given the outcome: the confounder is recorded mostly for high-outcome people.
    prob = 1 / (1 + np.exp(-2.5 * (y - np.median(y))))
    c_obs = np.where(rng.random(len(c)) < prob, c, np.nan).reshape(-1, 1)
    cc_rho, _, cc_n = partial_spearman(x, y, c_obs)
    mi_rho, mi_p, mi_n = pooled_partial_spearman(x, y, c_obs, m=10, seed=3)
    assert mi_n == len(x) > cc_n
    assert abs(mi_rho - full_rho) < abs(cc_rho - full_rho)
    assert abs(mi_rho - full_rho) < 0.03 and mi_p < 1e-6


def test_pooled_partial_spearman_without_gaps_is_the_plain_partial():
    x, y, c, _ = _confounded(300)
    assert pooled_partial_spearman(x, y, c.reshape(-1, 1), m=5, seed=0) == partial_spearman(x, y, c.reshape(-1, 1))


def test_nelson_aalen_hand_computed():
    time = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    event = np.array([1, 1, 0, 1, 0])
    # at t=1: 1/5; t=2: one event among 4 at risk; t=3: 1/2; t=5: no event
    expected = np.array([0.2, 0.45, 0.45, 0.95, 0.95])
    assert np.allclose(nelson_aalen(time, event), expected)


def test_pooled_cox_recovers_the_full_data_hazard_ratio():
    rng = np.random.default_rng(5)
    n = 3000
    c = rng.normal(size=n)
    x = 0.7 * c + rng.normal(size=n)
    hazard = np.exp(0.6 * c + 0.3 * x)
    t_event = rng.exponential(1 / (0.05 * hazard))
    t_cens = rng.uniform(0, 30, size=n)
    time, event = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float)
    full = cox_ph(time, event, np.column_stack([x, c])).coef[0]
    prob = np.where(event == 1, 0.85, 0.3)  # covariate recorded far more often for cases
    c_obs = np.where(rng.random(n) < prob, c, np.nan)
    cc = cox_ph(time, event, np.column_stack([x, c_obs])).coef[0]
    fit = pooled_cox(time, event, x, c_obs.reshape(-1, 1), m=10, seed=2)
    assert fit["n"] == n and abs(fit["coef"] - full) < abs(cc - full) and abs(fit["coef"] - full) < 0.05
    assert fit["se"] > 0 and fit["imputations"] == 10


def test_discovery_adjusted_tests_keep_rows_with_a_missing_confounder_when_asked():
    import pandas as pd

    from codas.core.discovery import DiscoveryRequest, run_discovery

    x, y, c, rng = _confounded(1500, seed=9)
    c_obs = np.where(rng.random(len(c)) < 0.5, c, np.nan)
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=len(x)), "c": c_obs, "y": y})

    def adjusted_n(imputations):
        report = run_discovery(frame, DiscoveryRequest(target_column="y", confounder_columns=["c"], top_k=3,
                                                       validation_resamples=200,
                                                       covariate_imputations=imputations))
        cand = next(c for c in report.candidates if c.feature == "x")
        test = next(t for t in cand.tests if t.name == "confounder_adjusted_robustness")
        return int(test.details.split("partial_spearman_n=")[1].split(",")[0])

    assert adjusted_n(0) < 900          # complete-case drops about half
    assert adjusted_n(5) == len(x)      # imputation keeps everyone
