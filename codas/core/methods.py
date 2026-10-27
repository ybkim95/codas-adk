"""Statistical and machine-learning method pools, as in the paper's CoDaS discovery loop.

Each round a deterministic runner applies one method from each pool to the round's candidates, and a
gap-finding agent chooses what to run next from what the previous rounds left open (a method, and for
some methods a stratifying covariate or a mediator). The pools:

* statistics: ``partial_correlation`` (partial Spearman, or Cox for a time to event),
  ``subgroup_stratified`` (effects by levels or tertiles of a covariate, Cochran's Q), ``robust_regression``
  (Huber M-estimation, bootstrap CI), ``propensity_stratification`` (high vs low exposure compared within
  propensity-score quintiles, adjusted within strata), ``mutual_information`` (k-NN estimate on
  confounder-residualised ranks, permutation p: catches non-monotone dependence), ``mediation``
  (product of coefficients with a bootstrap CI; the mediator is named by the agent), and
  ``survival_screen`` when the outcome has a follow-up time;
* machine learning: ``gradient_boosting``, ``elastic_net``, ``random_forest``, ``lasso``, ``ridge``,
  ``huber`` (continuous outcomes) and ``logistic`` (binary outcomes), each with nested cross-validation
  (5 outer folds, a 3-fold inner grid search), a bootstrap CI for the held-out metric, a permutation
  null, and feature importance. A time-to-event outcome is modelled as its event indicator here.

Every result is aggregate (estimates, intervals, p-values, counts). Gradient boosting is scikit-learn's
histogram implementation, the same algorithm family as LightGBM; XGBoost and LightGBM themselves are
not dependencies.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from . import analyses
from .statistics import benjamini_hochberg

STAT_ORDER = ("partial_correlation", "subgroup_stratified", "robust_regression", "propensity_stratification",
              "mutual_information")
ML_ORDER = ("gradient_boosting", "elastic_net", "random_forest", "lasso", "ridge", "huber", "logistic")
_DESCRIPTIONS = {
    "partial_correlation": "partial Spearman correlation with the confounders (Cox HR per SD for a time to event)",
    "subgroup_stratified": "effect within levels or tertiles of a covariate ('by'), with Cochran's Q and I^2",
    "robust_regression": "Huber robust regression per SD of the feature with the confounders, bootstrap CI",
    "propensity_stratification": "above- vs below-median exposure compared within propensity-score quintiles",
    "mutual_information": "mutual information with the outcome after removing the confounders; catches "
                          "non-monotone dependence",
    "mediation": "indirect effect through a named 'mediator' (product of coefficients, bootstrap CI)",
    "survival_screen": "Cox hazard ratio per SD with the confounders, BH across features",
    "gradient_boosting": "histogram gradient boosting, nested CV",
    "elastic_net": "elastic-net regression (penalised logistic for a binary outcome), nested CV",
    "random_forest": "random forest, nested CV",
    "lasso": "lasso (L1-penalised logistic for a binary outcome), nested CV",
    "ridge": "ridge (L2-penalised logistic for a binary outcome), nested CV",
    "huber": "Huber regression, nested CV (continuous outcomes)",
    "logistic": "logistic regression, nested CV (binary outcomes)",
}


def available(survival: bool) -> dict[str, list[str]]:
    stat = (["survival_screen"] if survival else []) + list(STAT_ORDER) + ["mediation"]
    return {"stat": stat, "ml": list(ML_ORDER)}


def describe(name: str) -> str:
    return _DESCRIPTIONS.get(name, "")


def pool_of(name: str) -> str | None:
    if name in ML_ORDER:
        return "ml"
    if name in STAT_ORDER or name in {"mediation", "survival_screen"}:
        return "stat"
    return None


def _default_order(survival: bool, binary: bool) -> dict[str, list[str]]:
    stat = (["survival_screen"] if survival else []) + list(STAT_ORDER)
    ml = [m for m in ML_ORDER if not (m == "huber" and (binary or survival)) and not (m == "logistic" and not (binary or survival))]
    return {"stat": stat, "ml": ml}


def next_methods(explored: list[str], requested: list[dict[str, Any]], survival: bool, binary: bool = False,
                 per_pool: int = 1) -> list[dict[str, Any]]:
    """What to run next: the requested methods first, then the next unexplored method of each pool that
    no request covers (``per_pool`` per pool). Mediation runs only on request (it needs a mediator)."""
    out = [dict(r) for r in requested if pool_of(str(r.get("method"))) is not None]
    covered = Counter(pool_of(r["method"]) for r in out)
    for pool, order in _default_order(survival, binary).items():
        todo = [m for m in order if m not in explored and m not in {r["method"] for r in out}]
        out += [{"method": m} for m in todo[:max(per_pool - covered[pool], 0)]]
    return out


def exhausted(explored: list[str], survival: bool, binary: bool = False) -> bool:
    """Every method of both default pools has been run."""
    return all(m in explored for order in _default_order(survival, binary).values() for m in order)


# --- helpers --------------------------------------------------------------------------------------

def _num(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def _filled(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Covariates with missing values set to the column median, standardised (empty matrix without any)."""
    if not columns:
        return np.zeros((len(frame), 0))
    m = np.column_stack([_num(frame, c) for c in columns])
    med = np.nanmedian(m, axis=0)
    m = np.where(np.isfinite(m), m, med)
    sd = m.std(axis=0)
    sd[sd == 0] = 1
    return (m - m.mean(axis=0)) / sd


def _with_bh(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row, q in zip(rows, benjamini_hochberg([r.get("p") if np.isfinite(r.get("p", np.nan)) else 1.0 for r in rows])):
        row["q"] = q
    return rows


def _kind(y: np.ndarray, time: str | None) -> str:
    return "cox" if time else ("logistic" if analyses._is_binary(y) else "ols")


# --- statistics ----------------------------------------------------------------------------------

def _partial_correlation(frame, features, target, confounders, time, imputations, seed, **_):
    rows = [analyses.association(frame, f, target, confounders, time, imputations, seed) for f in features]
    for r in rows:
        r.pop("log_estimate", None)
        r.pop("se", None)
    return {"results": _with_bh(rows)}


def _subgroup(frame, features, target, confounders, time, imputations, seed, by: str | None = None, **_):
    if not by:
        small = [c for c in confounders if pd.to_numeric(frame[c], errors="coerce").nunique() <= 6]
        by = (small or list(confounders) or [None])[0]
    if not by:
        raise ValueError("subgroup_stratified needs a 'by' covariate (none declared).")
    rows = [analyses.subgroup_analysis(frame, f, target, by, confounders, time, imputations, seed) for f in features]
    return {"by": by, "results": rows}


def _robust(frame, features, target, confounders, time, imputations, seed, resamples: int = 100, **_):
    from sklearn.linear_model import HuberRegressor

    y = _num(frame, target)
    if time or analyses._is_binary(y):
        raise ValueError("robust_regression applies to a continuous outcome.")
    cov = _filled(frame, list(confounders))
    rng = np.random.default_rng(seed)
    rows = []
    for f in features:
        x = analyses._z(_num(frame, f))
        ok = np.isfinite(x) & np.isfinite(y)
        design, yy = np.column_stack([x[ok], cov[ok]]), y[ok]
        fit = lambda idx: HuberRegressor(max_iter=500).fit(design[idx], yy[idx]).coef_[0]  # noqa: E731
        est = float(fit(np.arange(len(yy))))
        boots = np.array([fit(rng.integers(0, len(yy), len(yy))) for _ in range(resamples)])
        se = float(boots.std(ddof=1))
        rows.append({"feature": f, "estimate": est, "se": se, "p": analyses._p(est, se), "n": int(ok.sum()),
                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]})
    return {"measure": "change in outcome per SD (Huber)", "results": _with_bh(rows)}


def _rank_residual(v: np.ndarray, cov: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v)
    if cov.shape[1] == 0:
        return r
    design = np.column_stack([np.ones(len(r)), cov])
    beta, *_ = np.linalg.lstsq(design, r, rcond=None)
    return r - design @ beta


def _mutual_information(frame, features, target, confounders, time, imputations, seed, permutations: int = 49,
                        max_rows: int = 3000, **_):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    y_all = _num(frame, target)
    binary = analyses._is_binary(y_all) or bool(time)
    rng = np.random.default_rng(seed)
    cov_all = _filled(frame, list(confounders))
    rows = []
    for f in features:
        x = _num(frame, f)
        ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y_all))
        idx = rng.choice(ok, min(len(ok), max_rows), replace=False)
        cov = cov_all[idx]
        xr = _rank_residual(x[idx], cov).reshape(-1, 1)
        y = y_all[idx] if binary else _rank_residual(y_all[idx], cov)
        estimate = (lambda yy: mutual_info_classif(xr, yy.astype(int), random_state=seed)[0]) if binary else \
                   (lambda yy: mutual_info_regression(xr, yy, random_state=seed)[0])
        mi = float(estimate(y))
        null = np.array([estimate(rng.permutation(y)) for _ in range(permutations)])
        rows.append({"feature": f, "mi": mi, "p": float((1 + np.sum(null >= mi)) / (permutations + 1)),
                     "null_mean": float(null.mean()), "n": int(len(idx))})
    return {"results": _with_bh(rows)}


def _coef(kind: str, x: np.ndarray, y: np.ndarray, t: np.ndarray | None) -> np.ndarray:
    return analyses._fit(kind, x, y, t)[0]


def _mediation(frame, features, target, confounders, time, imputations, seed, mediator: str | None = None,
               resamples: int = 200, **_):
    if not mediator or mediator not in frame.columns:
        raise ValueError("mediation needs a 'mediator' column.")
    y = _num(frame, target)
    t = _num(frame, time) if time else None
    kind = _kind(y, time)
    m = _num(frame, mediator)
    cov = _filled(frame, [c for c in confounders if c != mediator])
    rng = np.random.default_rng(seed)
    rows = []
    for f in features:
        x = analyses._z(_num(frame, f))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(m) & (np.isfinite(t) & (t > 0) if t is not None else True)
        xx, mm, yy, cc = x[ok], m[ok], y[ok], cov[ok]
        tt = t[ok] if t is not None else None

        def effects(idx):
            a = analyses._ols(np.column_stack([xx[idx], cc[idx]]), mm[idx])[0][0]
            outcome = _coef(kind, np.column_stack([xx[idx], mm[idx], cc[idx]]), yy[idx], None if tt is None else tt[idx])
            total = _coef(kind, np.column_stack([xx[idx], cc[idx]]), yy[idx], None if tt is None else tt[idx])[0]
            return a * outcome[1], outcome[0], total

        indirect, direct, total = effects(np.arange(len(yy)))
        boots = np.array([effects(rng.integers(0, len(yy), len(yy)))[0] for _ in range(resamples)])
        rows.append({"feature": f, "mediator": mediator, "indirect": float(indirect), "direct": float(direct),
                     "total": float(total), "indirect_ci95": [float(np.percentile(boots, 2.5)),
                                                              float(np.percentile(boots, 97.5))],
                     "proportion_mediated": float(indirect / total) if total else math.nan, "n": int(ok.sum())})
    scale = {"ols": "outcome units per SD", "logistic": "log-odds per SD (product-of-coefficients approximation)",
             "cox": "log hazard per SD (product-of-coefficients approximation)"}[kind]
    return {"measure": scale, "results": rows}


def _propensity(frame, features, target, confounders, time, imputations, seed, strata: int = 5, **_):
    from sklearn.linear_model import LogisticRegression

    if not confounders:
        raise ValueError("propensity_stratification needs declared confounders.")
    y = _num(frame, target)
    t = _num(frame, time) if time else None
    kind = _kind(y, time)
    cov = _filled(frame, list(confounders))
    rows = []
    for f in features:
        x = _num(frame, f)
        ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(t) & (t > 0) if t is not None else True)
        treat = (x[ok] > np.median(x[ok])).astype(float)
        cc, yy = cov[ok], y[ok]
        tt = t[ok] if t is not None else None
        score = LogisticRegression(max_iter=1000).fit(cc, treat).predict_proba(cc)[:, 1]
        bins = pd.qcut(score, strata, labels=False, duplicates="drop")
        est, var, weights = [], [], []
        for k in np.unique(bins):
            s = bins == k
            if treat[s].sum() < 5 or (1 - treat[s]).sum() < 5:
                continue
            coef, cv = analyses._fit(kind, np.column_stack([treat[s], cc[s]]), yy[s], None if tt is None else tt[s])
            if np.isfinite(coef[0]) and np.isfinite(cv[0, 0]):
                est.append(coef[0])
                var.append(cv[0, 0])
                weights.append(s.sum())
        if not est:
            rows.append({"feature": f, "effect": math.nan, "ci95": [math.nan, math.nan], "p": math.nan})
            continue
        est, var = np.array(est), np.array(var)
        if kind == "ols":  # stratum-size weights: the average effect in the population
            w = np.array(weights) / np.sum(weights)
            effect, se = float(w @ est), float(math.sqrt(np.sum(w ** 2 * var)))
        else:  # ratio measures: inverse-variance pooling of the log scale
            w = 1 / var
            effect, se = float(np.sum(w * est) / np.sum(w)), float(math.sqrt(1 / np.sum(w)))
        lo, hi = effect - 1.959964 * se, effect + 1.959964 * se
        row = {"feature": f, "strata": int(len(est)), "n": int(ok.sum()), "p": analyses._p(effect, se)}
        if kind == "ols":
            row.update(effect=effect, ci95=[lo, hi])
        else:
            row.update(effect=math.exp(effect), ci95=[math.exp(lo), math.exp(hi)])
        rows.append(row)
    measure = {"ols": "mean difference, above vs below median", "logistic": "odds ratio, above vs below median",
               "cox": "hazard ratio, above vs below median"}[kind]
    return {"measure": measure, "results": _with_bh(rows)}


def _survival_screen(frame, features, target, confounders, time, imputations, seed, **_):
    if not time:
        raise ValueError("survival_screen needs a follow-up time.")
    return {"results": analyses.survival_screen(frame, features, target, time, confounders, imputations, seed)}


# --- machine learning ----------------------------------------------------------------------------

def _logistic(l1_ratio: float | None = None, **kwargs: Any):
    """``LogisticRegression`` with an L1 (``l1_ratio=1``) or elastic-net (0 < ``l1_ratio`` < 1) penalty.
    scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio`` (removal in 1.10) and warns on every
    fit that passes it; earlier versions need ``penalty``. The same model is built on either."""
    import sklearn
    from sklearn.linear_model import LogisticRegression

    if l1_ratio is None:
        return LogisticRegression(**kwargs)
    version = tuple(int("".join(c for c in part if c.isdigit()) or 0) for part in sklearn.__version__.split(".")[:2])
    if version >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, **kwargs)
    if l1_ratio == 1:
        return LogisticRegression(penalty="l1", **kwargs)
    return LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, **kwargs)


def _model(name: str, binary: bool, seed: int):
    from sklearn import ensemble, linear_model

    grids: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "ridge": ((linear_model.LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1, 10]}) if binary else
                  (linear_model.Ridge(), {"alpha": [0.1, 1, 10, 100]})),
        "lasso": ((_logistic(1.0, solver="liblinear", max_iter=2000),
                   {"C": [0.01, 0.1, 1, 10]}) if binary else
                  (linear_model.Lasso(max_iter=5000), {"alpha": [0.001, 0.01, 0.1]})),
        "elastic_net": ((_logistic(0.5, solver="saga", max_iter=3000, tol=1e-3),
                         {"C": [0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}) if binary else
                        (linear_model.ElasticNet(max_iter=5000), {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]})),
        # Each tree sees a quarter of the rows: at cohort scale a full forest per inner fold costs minutes.
        "random_forest": ((ensemble.RandomForestClassifier if binary else ensemble.RandomForestRegressor)(
            n_estimators=100, max_depth=10, max_samples=0.25, n_jobs=-1, random_state=seed),
            {"min_samples_leaf": [10, 40]}),
        "gradient_boosting": ((ensemble.HistGradientBoostingClassifier if binary else ensemble.HistGradientBoostingRegressor)(
            max_iter=200, random_state=seed), {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
        "huber": (None if binary else linear_model.HuberRegressor(max_iter=1000), {"epsilon": [1.35, 2.0]}),
        "logistic": (linear_model.LogisticRegression(max_iter=2000) if binary else None, {"C": [0.1, 1, 10]}),
    }
    return grids[name]


def _ml(name: str):
    def run(frame, features, target, confounders, time, imputations, seed, outer: int = 5, resamples: int = 200, **_):
        from sklearn.base import clone
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        columns = [*features, *confounders]
        y = _num(frame, target)
        ok = np.isfinite(y)
        x, y = np.column_stack([_num(frame, c) for c in columns])[ok], y[ok]
        binary = analyses._is_binary(y) or bool(time)
        estimator, grid = _model(name, binary, seed)
        if estimator is None:
            raise ValueError(f"{name} does not apply to a {'binary' if binary else 'continuous'} outcome.")
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])
        grid = {f"model__{k}": v for k, v in grid.items()}
        scoring = "roc_auc" if binary else "r2"
        splitter = (StratifiedKFold if binary else KFold)(outer, shuffle=True, random_state=seed)
        pred, chosen = np.zeros(len(y)), []
        for train, test in splitter.split(x, y if binary else None):
            search = GridSearchCV(clone(pipe), grid, cv=3, scoring=scoring).fit(x[train], y[train])
            chosen.append(json.dumps({k.removeprefix("model__"): v for k, v in search.best_params_.items()}, sort_keys=True))
            best = search.best_estimator_
            pred[test] = best.predict_proba(x[test])[:, 1] if binary else best.predict(x[test])
        metric = lambda yy, pp: analyses._metric("logistic" if binary else "ols", yy, pp, None)  # noqa: E731
        value = metric(y, pred)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(resamples):
            idx = rng.integers(0, len(y), len(y))
            boots.append(metric(y[idx], pred[idx]))
        boots = np.array([b for b in boots if np.isfinite(b)])
        params = json.loads(Counter(chosen).most_common(1)[0][0])
        fixed = clone(pipe).set_params(**{f"model__{k}": v for k, v in params.items()})
        n_perm = 19  # the smallest attainable permutation p is 1/(n_perm + 1) = 0.05
        simple = (StratifiedKFold if binary else KFold)(outer, shuffle=True, random_state=seed + 1)
        method = "predict_proba" if binary else "predict"
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            pp = cross_val_predict(fixed, x, yp, cv=simple, method=method)
            null.append(metric(yp, pp[:, 1] if binary else pp))
        final = clone(fixed)
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed,
                                              stratify=y if binary else None)
        final.fit(xtr, ytr)
        imp = permutation_importance(final, xte, yte, scoring=scoring, n_repeats=3, random_state=seed)
        importance = sorted(({"feature": c, "importance": float(v)} for c, v in zip(columns, imp.importances_mean)),
                            key=lambda r: -r["importance"])
        return {"metric": "auc" if binary else "r2", "value": value,
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else [math.nan] * 2,
                "null_p": float((1 + np.sum(np.array(null) >= value)) / (n_perm + 1)), "null_draws": n_perm,
                "outer_folds": outer, "best_params": params, "importance": importance, "n": int(len(y)),
                "note": "time to event modelled as its event indicator" if time else ""}
    return run


_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "partial_correlation": _partial_correlation, "subgroup_stratified": _subgroup, "robust_regression": _robust,
    "propensity_stratification": _propensity, "mutual_information": _mutual_information, "mediation": _mediation,
    "survival_screen": _survival_screen, **{m: _ml(m) for m in ML_ORDER},
}


def run(name: str, frame: pd.DataFrame, features: list[str], target: str, confounders: list[str] | tuple[str, ...] = (),
        time: str | None = None, imputations: int = 0, seed: int = 17, **options: Any) -> dict[str, Any]:
    """Run one method on ``features``; returns its aggregate result, or ``{"error": ...}``."""
    head = {"method": name, "pool": pool_of(name), "features": list(features)}
    if name not in _RUNNERS:
        return {**head, "error": f"unknown method {name!r}"}
    try:
        return {**head, **_RUNNERS[name](frame, list(features), target, list(confounders), time, imputations, seed,
                                         **options)}
    except Exception as exc:  # noqa: BLE001 - a method that cannot run is reported, never fatal
        return {**head, "error": f"{type(exc).__name__}: {exc}"}


def headline(result: dict[str, Any]) -> str:
    """One line per method result, for the agents' digest, the notebook and the PDF."""
    name, g = result.get("method"), analyses._g
    if "error" in result:
        return f"{name}: not run ({result['error']})"
    rows = result.get("results") or []
    if name in {"partial_correlation", "survival_screen"}:
        return f"{name}: " + "; ".join(f"{r['feature']} {analyses._effect(r)}, q {g(r.get('q'))}" for r in rows[:5])
    if name == "subgroup_stratified":
        return f"{name} by {result.get('by')}: " + "; ".join(
            f"{r['feature']} heterogeneity p {g(r.get('heterogeneity_p'))}, I² {g(r.get('i2'))}" for r in rows[:5])
    if name == "robust_regression":
        return f"{name}: " + "; ".join(f"{r['feature']} {g(r['estimate'])} ({g(r['ci95'][0])} to {g(r['ci95'][1])}), "
                                       f"q {g(r.get('q'))}" for r in rows[:5])
    if name == "mutual_information":
        return f"{name}: " + "; ".join(f"{r['feature']} MI {g(r['mi'])}, p {g(r['p'])}" for r in rows[:5])
    if name == "mediation":
        return f"{name}: " + "; ".join(
            f"{r['feature']} via {r['mediator']}: indirect {g(r['indirect'])} ({g(r['indirect_ci95'][0])} to "
            f"{g(r['indirect_ci95'][1])}), {g(r['proportion_mediated'])} of total" for r in rows[:5])
    if name == "propensity_stratification":
        return f"{name} ({result.get('measure')}): " + "; ".join(
            f"{r['feature']} {g(r.get('effect'))} ({g(r['ci95'][0])} to {g(r['ci95'][1])})" for r in rows[:5])
    if result.get("pool") == "ml":
        top = ", ".join(r["feature"] for r in (result.get("importance") or [])[:3])
        return (f"{name}: held-out {result['metric']} {g(result['value'])} ({g(result['ci95'][0])} to "
                f"{g(result['ci95'][1])}), permutation p {g(result['null_p'])}; top features {top}")
    return f"{name}: done"


def record_headline(record: dict[str, Any]) -> str:
    """One line for any recorded analysis: a pool method (``method:<name>``) or an agent analysis tool."""
    tool, result = str(record.get("tool", "")), record.get("result") or {}
    return headline(result) if tool.startswith("method:") else analyses.headline(tool, result)


def _bounded_worker(conn: Any, args: tuple, options: dict[str, Any]) -> None:
    try:
        conn.send(run(*args, **options))
    finally:
        conn.close()


def run_bounded(name: str, frame: pd.DataFrame, features: list[str], target: str,
                confounders: list[str] | tuple[str, ...] = (), time: str | None = None, imputations: int = 0,
                seed: int = 17, budget_s: float = 0, **options: Any) -> dict[str, Any]:
    """``run`` with a wall-clock budget: with ``budget_s`` > 0 the method runs in its own process, which is
    stopped when the budget is spent, so one slow fit (a solver that converges slowly on a rare outcome) can
    never hold up the analysis; the record then says the method was stopped. ``budget_s`` <= 0 runs in place."""
    args = (name, frame, list(features), target, list(confounders), time, imputations, seed)
    if not budget_s or budget_s <= 0:
        return run(*args, **options)
    import multiprocessing

    context = multiprocessing.get_context("spawn")   # no fork: the parent holds threads and network clients
    receive, send = context.Pipe(duplex=False)
    # Not a daemon: a daemonic process may not start its own workers, and the tree models use every core.
    worker = context.Process(target=_bounded_worker, args=(send, args, options), daemon=False)
    worker.start()
    send.close()
    try:
        if receive.poll(budget_s):
            try:
                return receive.recv()
            except EOFError:
                pass
        stopped = worker.is_alive()
        head = {"method": name, "pool": pool_of(name), "features": list(features)}
        if stopped:
            return {**head, "error": f"stopped after {budget_s:.0f} s (the method's time budget)", "timed_out": True}
        return {**head, "error": f"the method's process ended without a result (exit code {worker.exitcode})"}
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(5)
        receive.close()
