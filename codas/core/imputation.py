"""Multiple imputation of missing adjustment covariates.

An analysis adjusted for a covariate that is missing for many participants either drops them
(complete-case: biased whenever missingness depends on the outcome or exposure, and it wastes power)
or imputes them. Here the gaps are filled ``m`` times by chained equations with draws from the
posterior predictive distribution (``IterativeImputer(sample_posterior=True)``), using the exposure
and the outcome as auxiliary variables so the imputation model is congenial with the analysis. The
analysis is repeated on each completed copy and the estimates are combined with Rubin's rules.

For a time-to-event outcome the auxiliaries are the event indicator and the Nelson-Aalen cumulative
hazard at each participant's time (White & Royston, Stat Med 2009), which approximates the outcome
model better than the raw time does.

Only adjustment covariates are imputed. Rows missing the exposure or the outcome stay out of the
analysis, as before. Opt-in: callers pass a number of imputations; zero keeps complete-case.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats

from codas.core.statistics import partial_spearman


def pool(estimates: np.ndarray, variances: np.ndarray) -> tuple[float, float, float]:
    """Rubin's rules: (pooled estimate, total standard error, degrees of freedom) from ``m`` estimates
    and their squared standard errors."""
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = len(q)
    est, within = float(q.mean()), float(u.mean())
    if m < 2:
        return est, math.sqrt(within), math.inf
    between = float(q.var(ddof=1))
    total = within + (1 + 1 / m) * between
    if between <= 0:
        return est, math.sqrt(total), math.inf
    df = (m - 1) * (1 + within / ((1 + 1 / m) * between)) ** 2
    return est, math.sqrt(total), float(df)


def completed_covariates(covariates: np.ndarray, auxiliary: np.ndarray, m: int, seed: int) -> list[np.ndarray]:
    """``m`` copies of ``covariates`` (n x k) with the missing cells drawn from their posterior
    predictive given the other covariates and the ``auxiliary`` columns (n x a, complete). Observed
    cells are unchanged; a column whose observed values are all 0/1 stays 0/1."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401 - registers IterativeImputer
    from sklearn.impute import IterativeImputer

    cov = np.asarray(covariates, dtype=float)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    aux = np.asarray(auxiliary, dtype=float)
    if aux.ndim == 1:
        aux = aux.reshape(-1, 1)
    k = cov.shape[1]
    gaps = ~np.isfinite(cov)
    binary = [j for j in range(k) if set(np.unique(cov[~gaps[:, j], j])) <= {0.0, 1.0}]
    out = []
    for i in range(m):
        imputer = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence notices for near-constant columns
            filled = imputer.fit_transform(np.column_stack([cov, aux]))[:, :k]
        filled[~gaps] = cov[~gaps]
        for j in binary:
            filled[:, j] = np.clip(np.round(filled[:, j]), 0, 1)
        out.append(filled)
    return out


def _wald_p(z: float, se: float, df: float) -> float:
    if not np.isfinite(z) or not np.isfinite(se) or se <= 0:
        return math.nan
    return float(2 * (stats.norm.sf(abs(z / se)) if math.isinf(df) else stats.t.sf(abs(z / se), df)))


def pooled_partial_spearman(x, y, covariates: np.ndarray, m: int = 5, seed: int = 17) -> tuple[float, float, int]:
    """Partial Spearman of ``x`` and ``y`` given ``covariates`` whose missing values are multiply imputed;
    the Fisher-z transformed estimates are pooled. Same return as ``partial_spearman``. Without gaps
    among the analysable rows it is exactly ``partial_spearman``."""
    fit = pooled_partial_spearman_ci(x, y, covariates, m=m, seed=seed)
    return fit["rho"], fit["p"], fit["n"]


def pooled_partial_spearman_ci(x, y, covariates: np.ndarray, m: int = 5, seed: int = 17,
                               alpha: float = 0.05) -> dict:
    """``pooled_partial_spearman`` with its (1 - alpha) Fisher-z interval and the number of imputations
    used (0 when nothing was missing: the plain partial Spearman with a Fisher interval)."""
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    cov = np.asarray(covariates, dtype=float)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    k = cov.shape[1]
    rows = np.isfinite(x_arr) & np.isfinite(y_arr)
    if m < 1 or np.isfinite(cov[rows]).all():
        rho, p, n = partial_spearman(x_arr, y_arr, cov)
        z = math.atanh(float(np.clip(rho, -0.999999, 0.999999))) if np.isfinite(rho) else math.nan
        se, df, used = (1.0 / math.sqrt(n - 3) if n > 3 else math.nan), math.inf, 0
    else:
        x_arr, y_arr, cov = x_arr[rows], y_arr[rows], cov[rows]
        zs, variances, n = [], [], 0
        for completed in completed_covariates(cov, np.column_stack([x_arr, y_arr]), m, seed):
            rho, _p, n = partial_spearman(x_arr, y_arr, completed)
            if not np.isfinite(rho) or n - 3 - k <= 0:
                return {"rho": math.nan, "p": math.nan, "n": int(n), "ci95": [math.nan, math.nan], "imputations": m}
            zs.append(math.atanh(float(np.clip(rho, -0.999999, 0.999999))))
            variances.append(1.0 / (n - 3 - k))
        z, se, df = pool(np.array(zs), np.array(variances))
        rho, p, used = float(math.tanh(z)), _wald_p(z, se, df), m
    if not (np.isfinite(z) and np.isfinite(se)):
        return {"rho": rho, "p": p, "n": int(n), "ci95": [math.nan, math.nan], "imputations": used}
    crit = stats.norm.ppf(1 - alpha / 2) if math.isinf(df) else stats.t.ppf(1 - alpha / 2, df)
    return {"rho": float(rho), "p": float(p), "n": int(n), "ci95": [math.tanh(z - crit * se), math.tanh(z + crit * se)],
            "imputations": used}


def nelson_aalen(time, event) -> np.ndarray:
    """Nelson-Aalen cumulative hazard evaluated at each participant's own time."""
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=float)
    times = np.unique(t)
    at_risk = np.array([(t >= s).sum() for s in times], dtype=float)
    events = np.array([d[t == s].sum() for s in times], dtype=float)
    cumulative = np.cumsum(np.where(at_risk > 0, events / at_risk, 0.0))
    return cumulative[np.searchsorted(times, t)]


def pooled_cox(time, event, x, covariates: np.ndarray | None, m: int = 10, seed: int = 17,
               alpha: float = 0.05) -> dict:
    """Cox log hazard ratio of ``x`` (first column of the model) adjusted for ``covariates`` with their
    missing values multiply imputed; pooled with Rubin's rules. Returns coef, se, p, the HR with its
    (1 - alpha) CI, n, events and the number of imputations used (0 when nothing was missing)."""
    from codas.core.survival import cox_ph

    t, e, xv = (np.asarray(v, dtype=float) for v in (time, event, x))
    xv = xv.reshape(-1, 1) if xv.ndim == 1 else xv
    rows = np.isfinite(t) & np.isfinite(e) & np.all(np.isfinite(xv), axis=1) & (t > 0)
    cov = None if covariates is None else np.asarray(covariates, dtype=float)
    if cov is not None and cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    if cov is None or m < 1 or np.isfinite(cov[rows]).all():
        fit = cox_ph(t, e, xv if cov is None else np.column_stack([xv, cov]))
        sets, used = [fit], 0
    else:
        t, e, xv, cov = t[rows], e[rows], xv[rows], cov[rows]
        aux = np.column_stack([xv, e, nelson_aalen(t, e)])
        sets = [cox_ph(t, e, np.column_stack([xv, c])) for c in completed_covariates(cov, aux, m, seed)]
        used = m
    coefs = np.array([f.coef[0] for f in sets])
    variances = np.array([f.se[0] ** 2 for f in sets])
    if not (np.isfinite(coefs).all() and np.isfinite(variances).all()):
        return {"coef": math.nan, "se": math.nan, "p": math.nan, "hr": math.nan, "ci95": [math.nan, math.nan],
                "n": sets[0].n, "events": sets[0].events, "imputations": used}
    coef, se, df = pool(coefs, variances)
    crit = stats.norm.ppf(1 - alpha / 2) if math.isinf(df) else stats.t.ppf(1 - alpha / 2, df)
    return {"coef": coef, "se": se, "p": _wald_p(coef, se, df), "hr": math.exp(coef),
            "ci95": [math.exp(coef - crit * se), math.exp(coef + crit * se)], "n": sets[0].n,
            "events": sets[0].events, "imputations": used}
