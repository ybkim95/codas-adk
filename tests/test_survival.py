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
