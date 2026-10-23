"""Cox regression stays defined where a Newton step overshoots: rare events with a few extreme values."""

import numpy as np

from codas.core.survival import cox_ph


def _rare_events_with_outliers(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    x = rng.normal(size=n)
    x[:5] = rng.normal(0, 80, 5)
    t = rng.integers(1, 1500, n).astype(float)
    e = (rng.random(n) < 0.004).astype(float)
    e[np.argmax(x)] = 1
    return t, e, (x - x.mean()) / x.std()


def test_an_overshooting_newton_step_is_halved_instead_of_failing():
    # seed 108 made exp() underflow over a late risk set, so log(0) raised "math domain error"
    fit = cox_ph(*_rare_events_with_outliers(108))
    assert fit.coef.shape == (1,) and (np.isfinite(fit.coef[0]) or np.isnan(fit.coef[0]))


def test_ordinary_fits_are_unchanged():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.5 * x))
    c = rng.exponential(1.0, n)
    fit = cox_ph(np.minimum(t, c), (t <= c).astype(float), x)
    assert fit.converged and abs(fit.coef[0] - 0.5) < 0.06


def test_a_duplicated_or_constant_covariate_is_dropped_not_fatal():
    rng = np.random.default_rng(1)
    n = 4000
    x, days, age = rng.normal(size=n), rng.normal(150, 40, n), rng.normal(50, 15, n)
    t, c = rng.exponential(1 / np.exp(0.4 * x)), rng.exponential(1.0, n)
    time, event = np.ceil(np.minimum(t, c) * 365), (t <= c).astype(float)
    clean = cox_ph(time, event, np.column_stack([x, age, days]))
    dup = cox_ph(time, event, np.column_stack([x, age, days, days, np.ones(n), 2 * age + days]))
    assert dup.dropped == [3, 4, 5] and clean.dropped == []
    assert abs(dup.coef[0] - clean.coef[0]) < 1e-6 and np.isfinite(dup.se[0])
    assert np.isnan(dup.coef[3]) and np.isnan(dup.se[4])
