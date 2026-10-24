"""Effect sizes a reader can weigh, beyond a p-value.

For a value outcome: the change in the outcome's own units per SD and per natural unit of the feature, the
standardised coefficient, the variance the feature explains beyond the covariates (partial delta R^2), and
the adjusted difference between the top and bottom quarters of the feature. For a time to event: the hazard
ratio per SD and per natural unit, the hazard ratio of each quarter against the lowest with a test for
trend, the crude cumulative incidence by quarter at a horizon with the risk difference between the extreme
quarters, and the E-value of the estimate and of its confidence bound. Every effect gets a magnitude class,
so "significant but negligible" is said, not left to the reader.

Quarter cut-points and the SD are frozen on the discovery share and applied unchanged to the holdout.
``fit_frozen`` fits the prediction models on discovery (covariates alone, with the feature, with an
incumbent, with both); ``evaluate_frozen`` scores them on the holdout: the gain in discrimination (C for a
time to event or a binary outcome, R^2 for a value) with a bootstrap interval, and head to head against the
incumbent. Missing covariate values are filled with discovery medians, which never look at the outcome:
imputing with the outcome would inflate a prediction model's discrimination.

Nothing here names a dataset, an outcome or a feature.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from codas.core.survival import cox_ph, e_value

# Where an effect stops being negligible, becomes moderate and becomes large (conventions, docs/AOU.md): |rho|
# and a standardised coefficient from 0.1 (Cohen's small); a hazard ratio per SD beyond 0.95-1.05 (either way:
# a ratio and its inverse are the same size); a C or R^2 gain of 0.005, about what HDL cholesterol or CRP add
# to established cardiovascular risk models.
CLASSES = {"rho": (0.1, 0.2, 0.3), "std_beta": (0.1, 0.2, 0.3), "hr_per_sd": (1.05, 1.2, 1.5),
           "delta_c": (0.005, 0.01, 0.02), "delta_r2": (0.005, 0.01, 0.02)}
BOOTSTRAP = 200
_SEED = 20251024


# --- discrimination ------------------------------------------------------------------------------

def _left_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each position of an integer array, how many earlier positions hold a smaller value and how many
    an equal one: bottom-up merging, each level counted for all blocks at once (O(n log^2 n) in numpy)."""
    n = len(values)
    less = np.zeros(n, dtype=np.int64)
    equal = np.zeros(n, dtype=np.int64)
    if n < 2:
        return less, equal
    pos = np.arange(n)
    width = int(values.max()) + 1
    size = 1
    while size < n:
        block = pos // (2 * size)
        right = (pos % (2 * size)) >= size
        left_keys = np.sort(block[~right] * width + values[~right])
        keys = block[right] * width + values[right]
        start = np.searchsorted(left_keys, block[right] * width, side="left")
        lo = np.searchsorted(left_keys, keys, side="left")
        hi = np.searchsorted(left_keys, keys, side="right")
        less[right] += lo - start
        equal[right] += hi - lo
        size *= 2
    return less, equal


def concordance(time: Any, event: Any, risk: Any) -> float:
    """Harrell's C: among pairs where the earlier time is an event, the share in which the earlier one has the
    higher risk (ties in risk count half; equal times are not comparable). Exact, O(n log^2 n)."""
    t, e, r = (np.asarray(v, dtype=float) for v in (time, event, risk))
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[ok], e[ok] > 0, r[ok]
    if not e.any():
        return math.nan
    order = np.argsort(t, kind="mergesort")
    t, e, r = t[order], e[order], r[order]
    rank = np.unique(r, return_inverse=True)[1].astype(np.int64)
    group = np.unique(t, return_inverse=True)[1].astype(np.int64)
    n = len(t)
    # Later positions with a smaller or equal risk, then those in the same time group taken out again.
    less_after, equal_after = (c[::-1] for c in _left_counts(rank[::-1]))
    keys = group * (int(rank.max()) + 1) + rank
    less_same, equal_same = (c[::-1] for c in _left_counts(keys[::-1]))
    later = n - np.searchsorted(t, t, side="right")
    concordant = (less_after - less_same) + 0.5 * (equal_after - equal_same)
    comparable = later[e].sum()
    return float(concordant[e].sum() / comparable) if comparable else math.nan


def auc(y: Any, score: Any) -> float:
    """The area under the ROC curve (C for a binary outcome), ties counted half."""
    y, s = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok] > 0, s[ok]
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def r_squared(y: Any, prediction: Any) -> float:
    """Out-of-sample R^2 of a frozen prediction: 1 - SSE / SST around the sample's own mean."""
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - p) ** 2).sum()) / sst if sst > 0 else math.nan


# --- descriptive survival ----------------------------------------------------------------------------

def cumulative_incidence(time: Any, event: Any, horizon: float) -> float:
    """1 - the Kaplan-Meier survival at ``horizon`` (crude; censoring treated as non-informative)."""
    t, e = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    ok = np.isfinite(t) & np.isfinite(e)
    t, e = t[ok], e[ok] > 0
    if not len(t):
        return math.nan
    times, at = np.unique(t[e & (t <= horizon)], return_counts=True)
    if not len(times):
        return 0.0
    at_risk = len(t) - np.searchsorted(np.sort(t), times, side="left")
    return float(1 - np.prod(1 - at / at_risk))


# --- effects ---------------------------------------------------------------------------------------

def _filled(cov: pd.DataFrame | None, medians: dict[str, float] | None = None) -> np.ndarray | None:
    """Covariates with missing values filled by ``medians`` (the discovery share's, or this frame's) and a
    missing-value indicator for any column with gaps; None without covariates."""
    if cov is None or not cov.shape[1]:
        return None
    blocks = []
    for c in cov.columns:
        v = pd.to_numeric(cov[c], errors="coerce").to_numpy(dtype=float)
        miss = ~np.isfinite(v)
        fill = (medians or {}).get(c, np.nanmedian(v) if (~miss).any() else 0.0)
        blocks.append(np.where(miss, fill, v))
        if miss.any():
            blocks.append(miss.astype(float))
    return np.column_stack(blocks)


def _ols(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, their standard errors and R^2 of y on [1, design]."""
    x = np.column_stack([np.ones(len(y)), design])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    sst = float(((y - y.mean()) ** 2).sum())
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), (1 - float(resid @ resid) / sst if sst > 0 else math.nan)


def quarter(x: Any, cutpoints: list[float]) -> np.ndarray:
    """Quarter 0-3 of each value by the frozen cut-points (NaN where the value is missing)."""
    v = np.asarray(x, dtype=float)
    out = np.searchsorted(np.asarray(cutpoints, dtype=float), v, side="right").astype(float)
    out[~np.isfinite(v)] = np.nan
    return out


def magnitude(measure: str, value: float | None) -> str | None:
    """negligible, small, moderate or large by CLASSES (a hazard ratio and its inverse are the same size)."""
    if value is None or not np.isfinite(value) or measure not in CLASSES or (measure == "hr_per_sd" and value <= 0):
        return None
    size = abs(math.log(value)) if measure == "hr_per_sd" else abs(value)
    small, moderate, large = (math.log(b) if measure == "hr_per_sd" else b for b in CLASSES[measure])
    return "negligible" if size < small else "small" if size < moderate else "moderate" if size < large else "large"


def _ratio(log_value: float) -> float | None:
    """exp of a log ratio, rounded; None when it is not finite or beyond any estimable size."""
    return round(math.exp(log_value), 4) if np.isfinite(log_value) and abs(log_value) <= 20 else None


def e_value_bound(lo: float, hi: float) -> float:
    """The E-value of the confidence bound nearer the null (1 when the interval includes it)."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 1 <= hi:
        return 1.0
    return float(e_value(lo if lo > 1 else hi))


def value_effects(y: Any, x: Any, cov: pd.DataFrame | None, *, sd: float, unit: float | None,
                  cutpoints: list[float] | None) -> dict[str, Any]:
    """For a value outcome: change in its units per SD (frozen ``sd``) and per natural ``unit`` of the
    feature, the standardised coefficient, the partial delta R^2 and the adjusted top-minus-bottom quarter
    difference (frozen ``cutpoints``), with 95% intervals; covariates median-filled with indicators."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    c = _filled(cov)
    ok = np.isfinite(y) & np.isfinite(x)
    if c is not None:
        ok &= np.isfinite(c).all(axis=1)
    y, x, c = y[ok], x[ok], (c[ok] if c is not None else np.empty((ok.sum(), 0)))
    if len(y) < 30 or not np.isfinite(sd) or sd <= 0:
        return {}
    z = (x - np.nanmean(x)) / sd
    beta, se, r2_full = _ols(y, np.column_stack([z, c]))
    r2_base = _ols(y, c)[2] if c.shape[1] else 0.0
    per_sd, half = float(beta[1]), 1.959964 * float(se[1])
    sd_y = float(np.std(y, ddof=1))
    out = {"per_sd": round(per_sd, 4), "per_sd_ci95": [round(per_sd - half, 4), round(per_sd + half, 4)],
           "std_beta": round(per_sd / sd_y, 4) if sd_y > 0 else None,
           "partial_delta_r2": round(max(r2_full - r2_base, 0.0), 5), "n": int(len(y))}
    if unit:
        out.update(unit=unit, per_unit=round(per_sd * unit / sd, 4),
                   per_unit_ci95=[round((per_sd - half) * unit / sd, 4), round((per_sd + half) * unit / sd, 4)])
    if cutpoints:
        q = quarter(x, cutpoints)
        dummies = np.column_stack([(q == k).astype(float) for k in (1, 2, 3)])
        if all(dummies[:, j].sum() >= 20 for j in range(3)) and (q == 0).sum() >= 20:
            b, s, _ = _ols(y, np.column_stack([dummies, c]))
            out["top_vs_bottom_quarter"] = round(float(b[3]), 4)
            out["top_vs_bottom_quarter_ci95"] = [round(float(b[3] - 1.959964 * s[3]), 4),
                                                 round(float(b[3] + 1.959964 * s[3]), 4)]
    out["magnitude"] = magnitude("std_beta", out["std_beta"])
    return out


def survival_effects(time: Any, event: Any, x: Any, cov: pd.DataFrame | None, *, hr_per_sd: float,
                     ci95: list[float], sd: float, unit: float | None, cutpoints: list[float] | None,
                     horizon: float, draws: int = BOOTSTRAP, unit_label: str | None = None,
                     min_events: int = 0) -> dict[str, Any]:
    """For a time to event, beyond the hazard ratio per SD (given): per ``unit`` (``sd`` is the SD the hazard
    ratio is per, in the same scale; ``unit_label`` names it, e.g. "a doubling" on a log scale), by quarter
    (frozen ``cutpoints``) against the lowest with a test for trend, the crude cumulative incidence by quarter
    at ``horizon`` days with the top-minus-bottom risk difference (bootstrap interval), and the E-value of the
    confidence bound. A quarter whose events number 1 to ``min_events`` gets no cumulative incidence, which
    would reveal the count."""
    t, e, x = (np.asarray(v, dtype=float) for v in (time, event, x))
    lo, hi = ci95
    out: dict[str, Any] = {"e_value_ci": round(e_value_bound(lo, hi), 3), "magnitude": magnitude("hr_per_sd", hr_per_sd)}
    if unit and sd and np.isfinite(sd) and sd > 0:
        out.update(unit=unit_label or unit, hr_per_unit=round(float(np.exp(np.log(hr_per_sd) * unit / sd)), 4),
                   hr_per_unit_ci95=[round(float(np.exp(np.log(b) * unit / sd)), 4) for b in (lo, hi)])
    if not cutpoints:
        return out
    q = quarter(x, cutpoints)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(q)
    if any(((q[ok] == k).sum() < 20) for k in range(4)):
        return out
    c = _filled(cov)
    dummies = np.column_stack([(q == k).astype(float) for k in (1, 2, 3)])
    fit = cox_ph(t, e, dummies if c is None else np.column_stack([dummies, c]))
    if fit.converged:  # a quarter with too few events to estimate gets None, not an overflowing ratio
        out["quarter_hr"] = [1.0] + [_ratio(b) for b in fit.coef[:3]]
        out["quarter_hr_ci95"] = [[1.0, 1.0]] + [[_ratio(b - 1.959964 * s), _ratio(b + 1.959964 * s)]
                                                 for b, s in zip(fit.coef[:3], fit.se[:3])]
    trend = cox_ph(t, e, q[:, None] if c is None else np.column_stack([q, c]))
    if trend.converged:
        out["p_trend"] = float(trend.p[0])
    shown = [not 1 <= int((e[ok & (q == k)] > 0).sum()) <= min_events for k in range(4)]
    risks = [cumulative_incidence(t[ok & (q == k)], e[ok & (q == k)], horizon) for k in range(4)]
    out.update(horizon_days=horizon, cumulative_incidence=[round(r, 4) if s else None for r, s in zip(risks, shown)])
    if not (shown[0] and shown[3]):
        return out
    out["risk_difference"] = round(risks[3] - risks[0], 4)
    rng = np.random.default_rng(_SEED)
    top, bottom = np.flatnonzero(ok & (q == 3)), np.flatnonzero(ok & (q == 0))
    diffs = [cumulative_incidence(t[a], e[a], horizon) - cumulative_incidence(t[b], e[b], horizon)
             for a, b in ((rng.choice(top, len(top)), rng.choice(bottom, len(bottom))) for _ in range(draws))]
    out["risk_difference_ci95"] = ([round(float(np.quantile(diffs, 0.025)), 4), round(float(np.quantile(diffs, 0.975)), 4)]
                                   if diffs else None)
    return out


# --- frozen prediction models -----------------------------------------------------------------------

def _design(frame: pd.DataFrame, columns: list[str], frozen: dict[str, Any]) -> np.ndarray:
    """The model's columns, standardised by the discovery means and SDs, missing values set to the discovery
    medians (outcome-free)."""
    out = []
    for c in columns:
        v = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) if c in frame.columns else \
            np.full(len(frame), np.nan)
        v = np.where(np.isfinite(v), v, frozen["medians"][c])
        out.append((v - frozen["means"][c]) / frozen["sds"][c])
    return np.column_stack(out) if out else np.empty((len(frame), 0))


def fit_frozen(frame: pd.DataFrame, *, target: str, time: str | None, covariates: list[str], feature: str,
               incumbent: str | None = None) -> dict[str, Any] | None:
    """Prediction models fitted on the discovery share and frozen for the holdout: covariates alone, with the
    feature, and with an incumbent alone and with the feature (Cox for a time to event, else least squares,
    which for a binary outcome is a linear risk score)."""
    columns = list(dict.fromkeys([*covariates, feature, *([incumbent] if incumbent else [])]))
    columns = [c for c in columns if c in frame.columns and c not in (target, time)]
    if feature not in columns:
        return None
    values = {c: pd.to_numeric(frame[c], errors="coerce") for c in columns}
    frozen: dict[str, Any] = {"kind": "cox" if time else "linear", "feature": feature, "incumbent": incumbent,
                              "medians": {c: float(v.median()) for c, v in values.items()},
                              "means": {}, "sds": {}, "models": {}}
    for c, v in values.items():
        filled = v.fillna(frozen["medians"][c])
        sd = float(filled.std(ddof=0))
        frozen["means"][c], frozen["sds"][c] = float(filled.mean()), (sd if sd > 0 else 1.0)
    base = [c for c in columns if c in covariates and c not in (feature, incumbent)]
    specs = {"base": base, "feature": base + [feature]}
    if incumbent and incumbent in columns:
        specs.update(incumbent=base + [incumbent], both=base + [incumbent, feature])
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(frame[time], errors="coerce").to_numpy(dtype=float) if time else None
    for name, cols in specs.items():
        design = _design(frame, cols, frozen)
        if time:
            if not cols:
                frozen["models"][name] = {"columns": [], "coef": []}
                continue
            fit = cox_ph(t, y, design)
            if not fit.converged:
                return None
            frozen["models"][name] = {"columns": cols, "coef": [float(b) if np.isfinite(b) else 0.0 for b in fit.coef]}
        else:
            ok = np.isfinite(y)
            x = np.column_stack([np.ones(ok.sum()), design[ok]])
            beta, *_ = np.linalg.lstsq(x, y[ok], rcond=None)
            frozen["models"][name] = {"columns": cols, "coef": [float(b) for b in beta]}
    return frozen


def _score(frozen: dict[str, Any], name: str, frame: pd.DataFrame) -> np.ndarray:
    model = frozen["models"][name]
    design = _design(frame, model["columns"], frozen)
    coef = np.asarray(model["coef"], dtype=float)
    if frozen["kind"] == "cox":
        return design @ coef if len(coef) else np.zeros(len(frame))
    return coef[0] + design @ coef[1:]


def evaluate_frozen(frozen: dict[str, Any], frame: pd.DataFrame, *, target: str, time: str | None,
                    draws: int = BOOTSTRAP) -> dict[str, Any]:
    """The frozen models scored on the holdout: C (time to event, binary) or R^2 (value) of each, the gain from
    the feature with a bootstrap 95% interval, and with an incumbent the gain beyond it and the feature head
    to head against it."""
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(frame[time], errors="coerce").to_numpy(dtype=float) if time else None
    binary = not time and set(np.unique(y[np.isfinite(y)])) <= {0.0, 1.0}
    metric_name = "C" if time or binary else "R2"
    scores = {name: _score(frozen, name, frame) for name in frozen["models"]}

    def metric(idx: np.ndarray, s: np.ndarray) -> float:
        if time:
            return concordance(t[idx], y[idx], s[idx])
        return auc(y[idx], s[idx]) if binary else r_squared(y[idx], s[idx])

    every = np.arange(len(frame))
    out: dict[str, Any] = {"metric": metric_name, **{name: round(metric(every, s), 4) for name, s in scores.items()}}
    pairs = {"gain": ("feature", "base")}
    if "both" in scores:
        pairs.update(gain_beyond_incumbent=("both", "incumbent"), head_to_head=("feature", "incumbent"))
    rng = np.random.default_rng(_SEED)
    boot = {k: [] for k in pairs}
    for _ in range(draws):
        idx = rng.integers(0, len(frame), len(frame))
        for k, (a, b) in pairs.items():
            boot[k].append(metric(idx, scores[a]) - metric(idx, scores[b]))
    for k, (a, b) in pairs.items():
        values = np.asarray([v for v in boot[k] if np.isfinite(v)])
        out[k] = round(out[a] - out[b], 4)
        out[f"{k}_ci95"] = ([round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]
                            if len(values) else None)  # no draws: the point estimate only
    out["magnitude"] = magnitude("delta_c" if metric_name == "C" else "delta_r2", out["gain"])
    out["draws"] = draws
    return out
