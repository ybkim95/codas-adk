"""Analyses an agent can choose beyond the discovery screen, each deterministic and aggregate-only.

The discovery round answers one question (which features correlate with the outcome, robustly). An
expert analyst asks several more, and each has a function here:

* ``survival_screen``: for a time-to-event outcome, each feature's Cox hazard ratio per SD with the
  confounders, so candidates are ranked the way an incident outcome is finally tested;
* ``subgroup_analysis``: the association within strata of a covariate (its levels, or tertiles of a
  continuous one), with Cochran's Q and I^2 for heterogeneity;
* ``multivariable_model``: features fitted jointly with the confounders (OLS, logistic or Cox), each
  feature's independent effect, and the cross-validated incremental value over a confounders-only
  model (delta R^2, AUC or concordance, bootstrap CI): what the wearable adds to what is known;
* ``nonlinear_model``: gradient boosting against the linear model on the same inputs, held-out
  permutation importance, and the shape of each feature's partial dependence (increasing, decreasing,
  U-shaped, inverted U, non-monotone or flat);
* ``spline_dose_response``: a restricted cubic spline (4 knots, Harrell) with a Wald test of
  non-linearity and the effect at the 10th-90th percentiles relative to the median.

Missing confounder values are multiply imputed when ``imputations`` > 0 (``codas.core.imputation``);
rows missing a feature or the outcome are left out. Outputs hold estimates, intervals, p-values and
counts only, never a row.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .imputation import completed_covariates, nelson_aalen, pool, pooled_cox, pooled_partial_spearman_ci
from .statistics import benjamini_hochberg, safe_spearman
from .survival import cox_ph

_Z = stats.norm.ppf(0.975)


# --- features -------------------------------------------------------------------------------------

_OPS = {"ratio": lambda a, b: a / b.replace(0.0, np.nan), "product": lambda a, b: a * b,
        "difference": lambda a, b: a - b, "sum": lambda a, b: a + b}


def rebuild_feature(frame: pd.DataFrame, name: str, definition: dict[str, Any] | None = None) -> pd.Series | None:
    """A feature as a column of ``frame``: a raw column, a proposed ``{op, a, b}`` feature, or an engine
    ratio (``<a>_over_<b>``) or standardised product (``<a>_x_<b>``) of two ``components``."""
    definition = definition or {}
    num = lambda col: pd.to_numeric(frame[col], errors="coerce")  # noqa: E731
    if "op" in definition:
        if definition["op"] in _OPS and {definition["a"], definition["b"]} <= set(frame.columns):
            return _OPS[definition["op"]](num(definition["a"]), num(definition["b"])).replace([np.inf, -np.inf], np.nan)
        return None
    if name in frame.columns:
        return num(name)
    parts = definition.get("components") or []
    if len(parts) == 2 and set(parts) <= set(frame.columns):
        a, b = num(parts[0]), num(parts[1])
        if f"{parts[0]}_over_{parts[1]}" == name:
            return a / b.replace(0.0, np.nan)
        if f"{parts[0]}_x_{parts[1]}" == name:
            return ((a - a.mean()) / a.std(ddof=0)) * ((b - b.mean()) / b.std(ddof=0))
    return None


# --- shared helpers -------------------------------------------------------------------------------

def _num(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    return np.column_stack([_num(frame, c) for c in columns]) if columns else None


def _is_binary(y: np.ndarray) -> bool:
    values = np.unique(y[np.isfinite(y)])
    return len(values) == 2 and set(values.tolist()) <= {0.0, 1.0}


def _z(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * np.nan


def _p(estimate: float, se: float) -> float:
    return float(2 * stats.norm.sf(abs(estimate / se))) if np.isfinite(se) and se > 0 else math.nan


def association(frame: pd.DataFrame, feature: str, target: str, confounders: list[str] | tuple[str, ...] = (),
                time: str | None = None, imputations: int = 0, seed: int = 17) -> dict[str, Any]:
    """The feature's association with the outcome: Cox HR per SD (with ``time``), else partial Spearman
    given the confounders (plain Spearman without any)."""
    x, y = _num(frame, feature), _num(frame, target)
    cov = _matrix(frame, list(confounders))
    if time:
        fit = pooled_cox(_num(frame, time), y, _z(x), cov, m=imputations, seed=seed)
        return {"feature": feature, "test": "cox", "hr_per_sd": fit["hr"], "ci95": fit["ci95"], "p": fit["p"],
                "n": fit["n"], "events": fit["events"], "log_estimate": fit["coef"], "se": fit["se"]}
    if cov is not None:
        fit = pooled_partial_spearman_ci(x, y, cov, m=imputations, seed=seed)
        rho, p, n, ci = fit["rho"], fit["p"], fit["n"], fit["ci95"]
    else:
        rho, p, n = safe_spearman(x, y)
        half = _Z / math.sqrt(max(n - 3, 1))
        z = math.atanh(float(np.clip(rho, -0.999999, 0.999999))) if np.isfinite(rho) else math.nan
        ci = [math.tanh(z - half), math.tanh(z + half)]
    return {"feature": feature, "test": "partial_spearman" if cov is not None else "spearman", "rho": rho,
            "ci95": ci, "p": p, "n": n}


def _transformed(res: dict[str, Any]) -> tuple[float, float]:
    """(estimate, SE) on the scale where it is approximately normal: log HR, or Fisher z of rho."""
    lo, hi = res["ci95"]
    if res["test"] == "cox":
        if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0):
            return math.nan, math.nan
        return math.log(res["hr_per_sd"]), (math.log(hi) - math.log(lo)) / (2 * _Z)
    clip = lambda r: math.atanh(float(np.clip(r, -0.999999, 0.999999)))  # noqa: E731
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return math.nan, math.nan
    return clip(res["rho"]), (clip(hi) - clip(lo)) / (2 * _Z)


# --- survival screen ------------------------------------------------------------------------------

def survival_screen(frame: pd.DataFrame, features: list[str], target: str, time: str,
                    confounders: list[str] | tuple[str, ...] = (), imputations: int = 0, seed: int = 17) -> list[dict[str, Any]]:
    """Cox HR per SD of each feature with the confounders, BH-corrected across ``features``, strongest first."""
    rows = [association(frame, f, target, confounders, time, imputations, seed) for f in features]
    for row, q in zip(rows, benjamini_hochberg([r["p"] if np.isfinite(r["p"]) else 1.0 for r in rows])):
        row["q"] = q
        row.pop("log_estimate", None)
        row.pop("se", None)
    return sorted(rows, key=lambda r: (r["p"] if np.isfinite(r["p"]) else 2.0))


# --- subgroups -----------------------------------------------------------------------------------

def _label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def subgroup_analysis(frame: pd.DataFrame, feature: str, target: str, by: str,
                      confounders: list[str] | tuple[str, ...] = (), time: str | None = None,
                      imputations: int = 0, seed: int = 17, max_levels: int = 6) -> dict[str, Any]:
    """The association within each level of ``by`` (tertiles when it is continuous), and whether it
    differs between them (Cochran's Q, I^2)."""
    group = pd.to_numeric(frame[by], errors="coerce")
    levels = sorted(group.dropna().unique())
    if len(levels) > max_levels:
        edges = group.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        codes = pd.cut(group, edges, labels=False, include_lowest=True, duplicates="drop")
        names = {k: f"{by} tertile {k + 1} ({edges[k]:.3g} to {edges[k + 1]:.3g})" for k in range(len(edges) - 1)}
    else:
        codes = group.map({v: k for k, v in enumerate(levels)})
        names = {k: f"{by}={_label(v)}" for k, v in enumerate(levels)}
    others = [c for c in confounders if c != by]
    strata, est, se = [], [], []
    for k, name in names.items():
        part = frame.loc[codes == k]
        res = association(part, feature, target, others, time, imputations, seed)
        res.pop("feature", None)
        theta, s = _transformed(res)
        res.pop("log_estimate", None)
        res.pop("se", None)
        strata.append({"stratum": name, **res})
        if np.isfinite(theta) and np.isfinite(s) and s > 0:
            est.append(theta)
            se.append(s)
    out: dict[str, Any] = {"feature": feature, "by": by, "strata": strata}
    if len(est) >= 2:
        w = 1 / np.square(se)
        mean = float(np.sum(w * np.array(est)) / np.sum(w))
        q = float(np.sum(w * np.square(np.array(est) - mean)))
        df = len(est) - 1
        out.update(heterogeneity_q=q, heterogeneity_p=float(stats.chi2.sf(q, df)),
                   i2=max(0.0, (q - df) / q) if q > 0 else 0.0)
    return out


# --- model fitting ------------------------------------------------------------------------------

def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    sigma2 = float(resid @ resid) / max(len(y) - design.shape[1], 1)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return beta[1:], cov[1:, 1:]


def _logit(x: np.ndarray, y: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(y)), x])
    beta = np.zeros(design.shape[1])
    info = np.eye(design.shape[1])
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-np.clip(design @ beta, -30, 30)))
        w = mu * (1 - mu)
        info = design.T @ (design * w[:, None]) + 1e-8 * np.eye(design.shape[1])
        step = np.linalg.solve(info, design.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.pinv(info)
    return beta[1:], cov[1:, 1:]


def _fit(kind: str, x: np.ndarray, y: np.ndarray, t: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if kind == "cox":
        res = cox_ph(t, y, x)
        return res.coef, res.cov
    return (_logit if kind == "logistic" else _ols)(x, y)


def _pooled_fit(kind: str, exposures: np.ndarray, cov: np.ndarray | None, y: np.ndarray, t: np.ndarray | None,
                imputations: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Coefficients and covariance of [exposures, covariates], with missing covariates multiply imputed
    and pooled by Rubin's multivariate rules. Rows missing an exposure or the outcome are dropped."""
    rows = np.all(np.isfinite(exposures), axis=1) & np.isfinite(y)
    if t is not None:
        rows &= np.isfinite(t) & (t > 0)
    exposures, y = exposures[rows], y[rows]
    t = t[rows] if t is not None else None
    if cov is None:
        coef, var = _fit(kind, exposures, y, t)
        return coef, var, int(rows.sum())
    cov = cov[rows]
    if imputations < 1 or np.isfinite(cov).all():
        keep = np.all(np.isfinite(cov), axis=1)
        coef, var = _fit(kind, np.column_stack([exposures[keep], cov[keep]]), y[keep], None if t is None else t[keep])
        return coef, var, int(keep.sum())
    aux = [exposures, y.reshape(-1, 1)] + ([t.reshape(-1, 1), nelson_aalen(t, y).reshape(-1, 1)] if t is not None else [])
    fits = [_fit(kind, np.column_stack([exposures, c]), y, t)
            for c in completed_covariates(cov, np.column_stack(aux), imputations, seed)]
    coefs = np.array([f[0] for f in fits])
    within = np.mean([f[1] for f in fits], axis=0)
    between = np.cov(coefs, rowvar=False, ddof=1).reshape(within.shape)
    return coefs.mean(axis=0), within + (1 + 1 / imputations) * between, int(rows.sum())


# --- prediction metrics ---------------------------------------------------------------------------

def concordance(time, event, risk) -> float:
    """Harrell's C: among pairs where the earlier time is an event, the share in which the earlier one
    has the higher risk (ties in risk count half). O(n log n)."""
    t, e, r = (np.asarray(v, dtype=float) for v in (time, event, risk))
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[ok], e[ok], r[ok]
    order = np.argsort(-t, kind="mergesort")
    seen: list[float] = []  # risks of everyone with a later time, kept sorted
    concordant = comparable = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and t[order[j]] == t[order[i]]:
            j += 1
        group = order[i:j]  # equal times are not comparable with each other
        for k in group:
            if e[k] and seen:
                below = bisect_left(seen, r[k])
                equal = bisect_left(seen, np.nextafter(r[k], np.inf)) - below
                concordant += below + 0.5 * equal
                comparable += len(seen)
        for k in group:
            insort(seen, r[k])
        i = j
    return concordant / comparable if comparable else math.nan


def _metric(kind: str, y: np.ndarray, pred: np.ndarray, t: np.ndarray | None) -> float:
    if kind == "cox":
        return concordance(t, y, pred)
    if kind == "logistic":
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else math.nan
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else math.nan


def _oof_predictions(kind: str, x: np.ndarray | None, y: np.ndarray, t: np.ndarray | None,
                     folds: int, seed: int) -> np.ndarray:
    """Out-of-fold linear predictors (risk score for Cox, probability for logistic, mean otherwise).
    Missing inputs are filled with the training fold's medians."""
    from sklearn.model_selection import KFold, StratifiedKFold

    splitter = (StratifiedKFold(folds, shuffle=True, random_state=seed) if kind != "ols"
                else KFold(folds, shuffle=True, random_state=seed))
    pred = np.zeros(len(y))
    for train, test in splitter.split(np.zeros(len(y)), y if kind != "ols" else None):
        if x is None or x.shape[1] == 0:
            pred[test] = 0.0 if kind == "cox" else y[train].mean()
            continue
        med = np.nanmedian(x[train], axis=0)
        fill = lambda a: np.where(np.isfinite(a), a, med)  # noqa: E731
        xtr, xte = fill(x[train]), fill(x[test])
        mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
        sd[sd == 0] = 1
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        if kind == "cox":
            pred[test] = xte @ cox_ph(t[train], y[train], xtr).coef
        elif kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            pred[test] = LogisticRegression(C=1e6, max_iter=1000).fit(xtr, y[train]).predict_proba(xte)[:, 1]
        else:
            from sklearn.linear_model import LinearRegression

            pred[test] = LinearRegression().fit(xtr, y[train]).predict(xte)
    return pred


def multivariable_model(frame: pd.DataFrame, features: list[str], target: str,
                        confounders: list[str] | tuple[str, ...] = (), time: str | None = None,
                        imputations: int = 0, seed: int = 17, folds: int = 5, resamples: int = 200) -> dict[str, Any]:
    """Features fitted jointly with the confounders, and their cross-validated incremental value."""
    y = _num(frame, target)
    t = _num(frame, time) if time else None
    kind = "cox" if time else ("logistic" if _is_binary(y) else "ols")
    feats = np.column_stack([_z(_num(frame, f)) for f in features])
    cov = _matrix(frame, list(confounders))
    coef, var, n = _pooled_fit(kind, feats, cov, y, t, imputations, seed)
    se = np.sqrt(np.clip(np.diag(var), 0, None))
    terms = []
    for k, term in enumerate([*features, *confounders]):
        row = {"term": term, "estimate": float(coef[k]), "se": float(se[k]), "p": _p(coef[k], se[k]),
               "ci95": [float(coef[k] - _Z * se[k]), float(coef[k] + _Z * se[k])]}
        if k < len(features) and kind != "ols":
            ratio = "hr_per_sd" if kind == "cox" else "odds_ratio_per_sd"
            row[ratio] = math.exp(coef[k])
            row[f"{ratio}_ci95"] = [math.exp(v) for v in row["ci95"]]
        terms.append(row)

    rows = np.all(np.isfinite(feats), axis=1) & np.isfinite(y) & (np.isfinite(t) & (t > 0) if t is not None else True)
    yy, tt = y[rows], (t[rows] if t is not None else None)
    base_x = cov[rows] if cov is not None else None
    full_x = np.column_stack([feats[rows]] + ([cov[rows]] if cov is not None else []))
    base = _oof_predictions(kind, base_x, yy, tt, folds, seed)
    full = _oof_predictions(kind, full_x, yy, tt, folds, seed)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples if kind != "cox" else min(resamples, 100)):
        idx = rng.integers(0, len(yy), len(yy))
        tb = tt[idx] if tt is not None else None
        deltas.append(_metric(kind, yy[idx], full[idx], tb) - _metric(kind, yy[idx], base[idx], tb))
    deltas = np.array([d for d in deltas if np.isfinite(d)])
    base_m, full_m = _metric(kind, yy, base, tt), _metric(kind, yy, full, tt)
    return {"model": kind, "metric": {"cox": "c_index", "logistic": "auc", "ols": "r2"}[kind], "n": n,
            "events": int(np.nansum(yy)) if kind != "ols" else None, "coefficients": terms,
            "incremental": {"base": base_m, "full": full_m, "delta": full_m - base_m,
                            "delta_ci95": ([float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
                                           if len(deltas) else [math.nan, math.nan]),
                            "folds": folds, "base_terms": list(confounders)}}


# --- nonlinear model ------------------------------------------------------------------------------

def _shape(values: np.ndarray, scale: float) -> str:
    span = float(values.max() - values.min())
    if not np.isfinite(span) or span < 0.25 * scale:
        return "flat"
    trend, _p, _n = safe_spearman(np.arange(len(values), dtype=float), values)
    if trend >= 0.9:
        return "increasing"
    if trend <= -0.9:
        return "decreasing"
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    inner = range(2, len(values) - 2)
    if lo in inner and min(values[0], values[-1]) - values[lo] > 0.4 * span:
        return "U-shaped"
    if hi in inner and values[hi] - max(values[0], values[-1]) > 0.4 * span:
        return "inverted U"
    return "non-monotone"


def nonlinear_model(frame: pd.DataFrame, features: list[str], target: str,
                    confounders: list[str] | tuple[str, ...] = (), seed: int = 17, folds: int = 5) -> dict[str, Any]:
    """Gradient boosting against the linear model on features + confounders (held-out R^2 or AUC; a
    time-to-event outcome is modelled as its event indicator), held-out permutation importance, and the
    shape of each feature's partial dependence over its 5th-95th percentiles."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    columns = [*features, *confounders]
    y = _num(frame, target)
    rows = np.isfinite(y)
    x, y = np.column_stack([_num(frame, c) for c in columns])[rows], y[rows]
    binary = _is_binary(y)
    make_gbm = lambda: (HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor)(  # noqa: E731
        max_iter=300, learning_rate=0.05, early_stopping=True, random_state=seed)
    make_lin = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                                     LogisticRegression(max_iter=1000) if binary else LinearRegression())
    splitter = (StratifiedKFold if binary else KFold)(folds, shuffle=True, random_state=seed)
    pred_gbm, pred_lin = np.zeros(len(y)), np.zeros(len(y))
    importance = np.zeros(len(columns))
    score = "roc_auc" if binary else "r2"
    for train, test in splitter.split(x, y if binary else None):
        gbm, lin = make_gbm().fit(x[train], y[train]), make_lin().fit(x[train], y[train])
        pred_gbm[test] = gbm.predict_proba(x[test])[:, 1] if binary else gbm.predict(x[test])
        pred_lin[test] = lin.predict_proba(x[test])[:, 1] if binary else lin.predict(x[test])
        imp = permutation_importance(gbm, x[test], y[test], scoring=score, n_repeats=3, random_state=seed)
        importance += imp.importances_mean / folds
    kind = "logistic" if binary else "ols"
    full = make_gbm().fit(x, y)
    sample = x[np.random.default_rng(seed).choice(len(x), min(len(x), 2000), replace=False)]
    predict = (lambda a: full.predict_proba(a)[:, 1]) if binary else full.predict
    scale = float(np.std(predict(sample)))
    shapes = []
    for k, name in enumerate(features):
        grid = np.nanpercentile(x[:, k], np.linspace(5, 95, 10))
        values = []
        for g in grid:
            probe = sample.copy()
            probe[:, k] = g
            values.append(float(predict(probe).mean()))
        values = np.array(values)
        shapes.append({"feature": name, "shape": _shape(values, scale),
                       "range_of_effect": float(values.max() - values.min())})
    ranked = sorted(({"feature": c, "importance": float(v)} for c, v in zip(columns, importance)),
                    key=lambda r: -r["importance"])
    return {"metric": "auc" if binary else "r2", "n": int(len(y)), "linear": _metric(kind, y, pred_lin, None),
            "nonlinear": _metric(kind, y, pred_gbm, None), "importance": ranked, "shapes": shapes}


# --- spline dose-response -------------------------------------------------------------------------

def _rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (x plus k-2 non-linear terms), linear beyond the outer knots."""
    k = len(knots)
    pos = lambda v: np.where(v > 0, v, 0.0) ** 3  # noqa: E731
    norm = (knots[-1] - knots[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        cols.append((pos(x - knots[j])
                     - pos(x - knots[-2]) * (knots[-1] - knots[j]) / (knots[-1] - knots[-2])
                     + pos(x - knots[-1]) * (knots[-2] - knots[j]) / (knots[-1] - knots[-2])) / norm)
    return np.column_stack(cols)


def spline_dose_response(frame: pd.DataFrame, feature: str, target: str,
                         confounders: list[str] | tuple[str, ...] = (), time: str | None = None,
                         imputations: int = 0, seed: int = 17) -> dict[str, Any]:
    """Restricted cubic spline of the feature (knots at its 5th, 35th, 65th and 95th percentiles) with the
    confounders; Wald test that the non-linear terms are zero; and the effect at the 10th, 25th, 50th,
    75th and 90th percentiles against the median (mean difference, odds ratio or hazard ratio)."""
    x = _num(frame, feature)
    y = _num(frame, target)
    t = _num(frame, time) if time else None
    kind = "cox" if time else ("logistic" if _is_binary(y) else "ols")
    knots = np.nanpercentile(x, [5, 35, 65, 95])
    basis = _rcs(x, knots)
    coef, var, n = _pooled_fit(kind, basis, _matrix(frame, list(confounders)), y, t, imputations, seed)
    b, v = coef[:basis.shape[1]], var[:basis.shape[1], :basis.shape[1]]
    nl, vnl = b[1:], v[1:, 1:]
    try:
        wald = float(nl @ np.linalg.solve(vnl, nl))
        p_nl = float(stats.chi2.sf(wald, len(nl)))
    except np.linalg.LinAlgError:
        p_nl = math.nan
    ref = _rcs(np.array([np.nanmedian(x)]), knots)[0]
    curve = []
    for pct in (10, 25, 50, 75, 90):
        value = float(np.nanpercentile(x, pct))
        d = _rcs(np.array([value]), knots)[0] - ref
        est, se = float(d @ b), float(math.sqrt(max(d @ v @ d, 0.0)))
        lo, hi = est - _Z * se, est + _Z * se
        if kind != "ols":
            est, lo, hi = math.exp(est), math.exp(lo), math.exp(hi)
        curve.append({"percentile": pct, "value": value, "effect": est, "ci95": [lo, hi]})
    measure = {"cox": "hazard ratio vs the median", "logistic": "odds ratio vs the median",
               "ols": "mean difference vs the median"}[kind]
    return {"feature": feature, "model": kind, "measure": measure, "n": n, "knots": knots.tolist(),
            "nonlinearity_p": p_nl, "curve": curve}


# --- one-line summaries (reports and notebooks) --------------------------------------------------

def _g(value: Any) -> str:
    return f"{value:.3g}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"


def _effect(res: dict[str, Any]) -> str:
    ci = res.get("ci95") or [math.nan, math.nan]
    if "hr_per_sd" in res:
        return f"HR/SD {_g(res['hr_per_sd'])} ({_g(ci[0])} to {_g(ci[1])})"
    return f"rho {_g(res.get('rho'))} ({_g(ci[0])} to {_g(ci[1])})"


def headline(tool: str, result: dict[str, Any]) -> str:
    """A one-line reading of an analysis result, for reports and notebook tables."""
    if "error" in result:
        return f"not run: {result['error']}"
    if tool == "survival_screen":
        top = result.get("results") or []
        return "; ".join(f"{r['feature']}: {_effect(r)}, q {_g(r.get('q'))}" for r in top[:4])
    if tool == "subgroup_analysis":
        strata = "; ".join(f"{s['stratum']}: {_effect(s)}" for s in result.get("strata") or [])
        return (f"{strata}. Heterogeneity p {_g(result.get('heterogeneity_p'))}, I² {_g(result.get('i2'))}")
    if tool == "multivariable_model":
        inc = result.get("incremental") or {}
        ci = inc.get("delta_ci95") or [math.nan, math.nan]
        independent = [c["term"] for c in result.get("coefficients") or []
                       if c["term"] not in (inc.get("base_terms") or []) and isinstance(c.get("p"), float) and c["p"] < 0.05]
        return (f"{result.get('metric')} {_g(inc.get('base'))} with covariates only, {_g(inc.get('full'))} with the "
                f"features (delta {_g(inc.get('delta'))}, 95% CI {_g(ci[0])} to {_g(ci[1])}); independent effects: "
                + (", ".join(independent) or "none"))
    if tool == "nonlinear_model":
        shapes = ", ".join(f"{s['feature']} {s['shape']}" for s in result.get("shapes") or [])
        top = (result.get("importance") or [{}])[0].get("feature")
        return (f"{result.get('metric')} linear {_g(result.get('linear'))} vs boosting {_g(result.get('nonlinear'))}; "
                f"shapes: {shapes}; most important: {top}")
    if tool == "dose_response":
        curve = result.get("curve") or []
        end = curve[-1] if curve else {}
        ci = end.get("ci95") or [math.nan, math.nan]
        return (f"non-linearity p {_g(result.get('nonlinearity_p'))}; 90th vs 50th percentile: "
                f"{_g(end.get('effect'))} ({_g(ci[0])} to {_g(ci[1])}) {result.get('measure', '')}")
    return ""
