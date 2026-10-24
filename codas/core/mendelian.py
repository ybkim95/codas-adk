"""Genetic triangulation of an exposure-outcome association: Mendelian randomization within one cohort.

Genetic variants associated with an exposure in published studies (instruments) are allocated at conception, so
they are not caused by the outcome or by most confounders of the exposure. Here, within one cohort:

* ``allele_score``: each person's weighted count of exposure-raising alleles (published weights).
* ``one_sample``: the score's association with the exposure (first stage, least squares with the covariates;
  its F statistic says whether the instrument is strong enough, conventionally F >= 10) and with the outcome
  (reduced form: Cox for a time to event, least squares for a value), and their ratio (Wald), the outcome effect
  per unit of exposure the genetic variation implies, with a delta-method interval. In one sample a weak
  instrument biases the ratio toward the confounded estimate, so F is reported beside it.
* ``per_variant``: the same two associations for each variant, combined by inverse-variance weighting (IVW),
  MR-Egger (whose intercept tests directional pleiotropy) and the weighted median (valid when at least half of
  the weight comes from valid instruments; Bowden et al. 2016), each with an interval.

Covariates should include genetic principal components, so ancestry does not confound the score. Nothing here
names a dataset, an exposure or an outcome.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from codas.core.survival import cox_ph

_Z = 1.959963984540054


def allele_score(dosages: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted allele count per person (rows); a missing dosage counts as the variant's mean."""
    d = np.asarray(dosages, dtype=float)
    means = np.nanmean(d, axis=0)
    d = np.where(np.isfinite(d), d, means)
    return d @ np.asarray(weights, dtype=float)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, standard errors and residual variance of least squares with an intercept (x without it)."""
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return beta[1:], np.sqrt(np.clip(np.diag(cov)[1:], 0, None)), sigma2


def _association(y: np.ndarray, g: np.ndarray, cov: np.ndarray | None, time: np.ndarray | None) -> tuple[float, float]:
    """(estimate, SE) of ``g`` on ``y`` given ``cov``: Cox when ``time`` is given, else least squares."""
    x = g[:, None] if cov is None else np.column_stack([g, cov])
    if time is not None:
        fit = cox_ph(time, y, x)
        return float(fit.coef[0]), float(fit.se[0])
    beta, se, _ = _ols(y, x)
    return float(beta[0]), float(se[0])


def _complete(*arrays: np.ndarray | None) -> np.ndarray:
    ok = None
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a, dtype=float)
        m = np.isfinite(a) if a.ndim == 1 else np.all(np.isfinite(a), axis=1)
        ok = m if ok is None else ok & m
    return ok


def one_sample(exposure: Any, outcome: Any, score: Any, covariates: Any = None, time: Any = None) -> dict[str, Any]:
    """The allele score's first stage, reduced form and Wald ratio (module doc); ratio per unit of exposure."""
    x, y, g = (np.asarray(v, dtype=float) for v in (exposure, outcome, score))
    cov = None if covariates is None else np.asarray(covariates, dtype=float)
    t = None if time is None else np.asarray(time, dtype=float)
    ok = _complete(x, y, g, cov, t)
    x, y, g = x[ok], y[ok], g[ok]
    cov = None if cov is None else cov[ok]
    t = None if t is None else t[ok]
    bx, sx = _association(x, g, cov, None)
    by, sy = _association(y, g, cov, t)
    f = (bx / sx) ** 2 if sx > 0 else math.nan
    ratio = by / bx if bx else math.nan
    se = math.sqrt(sy ** 2 / bx ** 2 + by ** 2 * sx ** 2 / bx ** 4) if bx else math.nan
    return {"n": int(ok.sum()), "first_stage": bx, "first_stage_se": sx, "f_statistic": f, "reduced_form": by,
            "reduced_form_se": sy, "ratio": ratio, "ratio_se": se,
            "ci95": [ratio - _Z * se, ratio + _Z * se] if np.isfinite(se) else None,
            "p": float(2 * stats.norm.sf(abs(ratio / se))) if se and np.isfinite(se) else math.nan,
            "scale": "log hazard ratio" if t is not None else "outcome units"}


def per_variant(exposure: Any, outcome: Any, dosages: Any, covariates: Any = None, time: Any = None,
                draws: int = 500, seed: int = 20251024) -> dict[str, Any]:
    """Per-variant associations and their IVW, MR-Egger and weighted-median combinations (module doc)."""
    x, y = np.asarray(exposure, dtype=float), np.asarray(outcome, dtype=float)
    d = np.asarray(dosages, dtype=float)
    cov = None if covariates is None else np.asarray(covariates, dtype=float)
    t = None if time is None else np.asarray(time, dtype=float)
    ok = _complete(x, y, cov, t)
    bx, sx, by, sy = [], [], [], []
    for j in range(d.shape[1]):
        keep = ok & np.isfinite(d[:, j])
        c = None if cov is None else cov[keep]
        a = _association(x[keep], d[keep, j], c, None)
        b = _association(y[keep], d[keep, j], c, None if t is None else t[keep])
        bx.append(a[0]), sx.append(a[1]), by.append(b[0]), sy.append(b[1])
    bx, sx, by, sy = (np.asarray(v) for v in (bx, sx, by, sy))
    use = np.isfinite(bx) & np.isfinite(by) & (sy > 0) & (np.abs(bx) > 0)
    bx, sx, by, sy = bx[use], sx[use], by[use], sy[use]
    out: dict[str, Any] = {"variants": int(use.sum())}
    if len(bx) < 2:
        return out
    w = bx ** 2 / sy ** 2
    ivw = float((bx * by / sy ** 2).sum() / w.sum())
    se_ivw = math.sqrt(1 / w.sum())
    out["ivw"] = {"estimate": ivw, "se": se_ivw, "ci95": [ivw - _Z * se_ivw, ivw + _Z * se_ivw],
                  "p": float(2 * stats.norm.sf(abs(ivw / se_ivw)))}
    if len(bx) >= 3:
        sign = np.sign(bx)  # orient every variant to raise the exposure (InSIDE-consistent Egger)
        design = np.column_stack([np.ones(len(bx)), np.abs(bx)])
        weights = 1 / sy ** 2
        wd = design * weights[:, None]
        beta = np.linalg.solve(design.T @ wd, wd.T @ (by * sign))
        resid = by * sign - design @ beta
        dof = max(len(bx) - 2, 1)
        phi = max(float((weights * resid ** 2).sum()) / dof, 1.0)
        cov = np.linalg.inv(design.T @ wd) * phi
        se = np.sqrt(np.diag(cov))
        out["egger"] = {"estimate": float(beta[1]), "se": float(se[1]),
                        "ci95": [float(beta[1] - _Z * se[1]), float(beta[1] + _Z * se[1])],
                        "intercept": float(beta[0]), "intercept_p": float(2 * stats.t.sf(abs(beta[0] / se[0]), dof))}
    ratio, ratio_se = by / bx, sy / np.abs(bx)
    est = _weighted_median(ratio, 1 / ratio_se ** 2)
    rng = np.random.default_rng(seed)
    boot = [_weighted_median(rng.normal(by, sy) / rng.normal(bx, sx), 1 / ratio_se ** 2) for _ in range(draws)]
    se_wm = float(np.std(boot, ddof=1))
    out["weighted_median"] = {"estimate": est, "se": se_wm, "ci95": [est - _Z * se_wm, est + _Z * se_wm]}
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Bowden et al.'s weighted median: the 50th percentile of the weight distribution, interpolated."""
    order = np.argsort(values)
    v, w = values[order], weights[order] / weights.sum()
    cum = np.cumsum(w) - 0.5 * w
    below = np.searchsorted(cum, 0.5) - 1
    if below < 0:
        return float(v[0])
    if below >= len(v) - 1:
        return float(v[-1])
    return float(v[below] + (v[below + 1] - v[below]) * (0.5 - cum[below]) / (cum[below + 1] - cum[below]))
