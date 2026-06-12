"""Statistical utilities used by deterministic CoDaS runners."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


def finite_xy(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def is_constant(values: np.ndarray) -> bool:
    return len(values) == 0 or np.nanstd(values) < 1e-12


def lag1_autocorr(values: Iterable[float]) -> float:
    """Lag-1 autocorrelation of a series in its given (time) order. Returns 0.0 when undefined.

    Used to gauge temporal dependence: strongly autocorrelated autocorrelated streams (EDA, HR) violate
    the independence assumption of correlation screening, so the effective sample size is far below
    the row count (temporal pseudo-replication)."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 4:
        return 0.0
    a0, a1 = a[:-1], a[1:]
    if is_constant(a0) or is_constant(a1):
        return 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(a0, a1)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def correlation_pvalue(r: float, n: int) -> float:
    """Two-sided p-value for a correlation coefficient r at sample size n (t-distribution).

    Lets us recompute significance at an autocorrelation-corrected EFFECTIVE n, so a small
    correlation between two slowly-drifting series is not declared significant on the inflated
    raw row count."""
    if r is None or n is None or not np.isfinite(r) or n < 3:
        return 1.0
    r = max(min(float(r), 0.999999), -0.999999)
    df = int(n) - 2
    if df <= 0:
        return 1.0
    t = r * math.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))


def autocorr_effective_n(n: int, r1_x: float, r1_y: float) -> int:
    """Effective sample size for the correlation of two autocorrelated series (Pyper & Peterman 1998
    AR(1) approximation): n_eff = n * (1 - r1x*r1y) / (1 + r1x*r1y). Clamped to [4, n]. When either
    series is white noise (r1 ~ 0) the product ~ 0 and n_eff ~ n, so non-temporal data is unaffected."""
    if n is None or n < 4:
        return int(n or 0)
    prod = float(r1_x) * float(r1_y)
    prod = max(min(prod, 0.999), -0.999)
    factor = (1.0 - prod) / (1.0 + prod)
    n_eff = int(round(n * factor))
    return max(4, min(int(n), n_eff))


def intraclass_correlation(values: Iterable[float], groups: Iterable) -> float:
    """One-way random-effects ICC(1): fraction of variance that is BETWEEN clusters.
    ICC ~ 0 => the variable varies freely within a cluster (rows ~ independent for it).
    ICC ~ 1 => the variable is ~constant within a cluster (the rows are near-replicates).
    Used to size the design-effect deflation for clustered repeated measures."""
    v = np.asarray(list(values), dtype=float)
    g = np.asarray(list(groups))
    mask = np.isfinite(v)
    v, g = v[mask], g[mask]
    if v.size < 3:
        return 0.0
    uniq = np.unique(g)
    k = int(uniq.size)
    N = int(v.size)
    if k < 2 or k >= N:
        return 0.0
    grand = float(v.mean())
    ssb = 0.0; ssw = 0.0; sizes = []
    for u in uniq:
        gi = v[g == u]
        if gi.size == 0:
            continue
        sizes.append(gi.size)
        mi = float(gi.mean())
        ssb += gi.size * (mi - grand) ** 2
        ssw += float(((gi - mi) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (N - k) if (N - k) > 0 else 0.0
    sizes = np.asarray(sizes, dtype=float)
    m0 = (N - (sizes ** 2).sum() / N) / (k - 1)  # ANOVA mean cluster size (unbalanced)
    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return 0.0
    icc = (msb - msw) / denom
    return float(max(0.0, min(1.0, icc)))


def cluster_effective_n(n: int, mean_cluster_size: float, icc_x: float, icc_y: float, n_clusters: int) -> int:
    """Design-effect effective sample size for a correlation under clustered repeated measures
    (the between-cluster analog of the Pyper-Peterman AR(1) correction). DEFF = 1 + (m̄-1)*ρ_icc
    where ρ_icc is the geometric mean of the two variables' ICCs, so a feature that varies freely
    WITHIN a cluster (icc~0) is barely deflated, while a feature that is ~constant within a cluster
    (icc~1, a per-subject trait) is deflated toward the number of clusters. Clamped to [n_clusters, n]."""
    if n is None or n < 4 or n_clusters is None or n_clusters < 2:
        return int(n or 0)
    rho_icc = math.sqrt(max(float(icc_x), 0.0) * max(float(icc_y), 0.0))
    deff = 1.0 + (max(float(mean_cluster_size), 1.0) - 1.0) * rho_icc
    if deff <= 1.0:
        return int(n)
    n_eff = int(round(n / deff))
    return max(int(n_clusters), min(int(n), n_eff))


def within_subject_two_stage(feature: Iterable[float], target: Iterable[float], groups: Iterable,
                             time: "Iterable | None" = None,
                             min_pairs_per_subject: int = 5, min_subjects: int = 8,
                             min_effective_pairs: int = 5) -> dict:
    """Two-stage (summary-statistic) within-subject association — the rigorous way to test a
    LONGITUDINAL predictor without pseudo-replication. Stage 1: within each subject, compute that
    subject's own Spearman ρ between feature and outcome over their repeated measures (confound-free —
    every time-invariant between-subject confound is differenced out). Stage 2: test whether the
    per-subject ρ's are centered away from 0 across subjects (Wilcoxon signed-rank), with n = number of
    subjects (the honest independent-unit count). Returns median within-subject ρ, p, q-able p, and the
    fraction of subjects with a consistent-sign effect.

    This recovers a within-subject tracking signal (e.g. voice degrades AS a patient's UPDRS rises)
    that a cross-sectional between-subject screen at n≈N_subjects cannot see — while never scoring the
    repeated rows as independent.

    AR(1) temporal-autocorrelation correction (Pyper & Peterman 1998): a subject's repeated measures are
    serially autocorrelated, so its raw point count overstates how much INDEPENDENT information backs its
    ρ_i. For each subject we time-order the rows (by `time` if provided, else data order), estimate the
    lag-1 autocorrelation of the feature and the outcome, and derive the AR(1)-corrected effective pairs
    n_eff = n·(1−r1f·r1t)/(1+r1f·r1t). Subjects whose n_eff falls below `min_effective_pairs` are dropped
    (their ρ_i is autocorrelation-unreliable), and the median effective pairs / within-subject
    autocorrelation are reported so the two-stage result is read with the AR-aware sample size, not the
    raw point count. White-noise within-subject series (r1≈0) are unaffected (n_eff≈n)."""
    f = np.asarray(list(feature), dtype=float)
    t = np.asarray(list(target), dtype=float)
    g = np.asarray(list(groups))
    tm = np.asarray(list(time)) if time is not None else None
    rhos: list[float] = []
    eff_pairs: list[int] = []
    within_ac: list[float] = []
    n_excluded_autocorr = 0
    for u in np.unique(g):
        m = (g == u) & np.isfinite(f) & np.isfinite(t)
        if int(m.sum()) < min_pairs_per_subject:
            continue
        fi, ti = f[m], t[m]
        if tm is not None:                       # time-order this subject's series so lag-1 is meaningful
            order = np.argsort(tm[m], kind="stable")
            fi, ti = fi[order], ti[order]
        if is_constant(fi) or is_constant(ti):
            continue
        # AR(1) correction: discount this subject's pairs by within-subject temporal autocorrelation.
        r1f, r1t = lag1_autocorr(fi), lag1_autocorr(ti)
        n_eff = autocorr_effective_n(int(len(fi)), r1f, r1t)
        if n_eff < min_effective_pairs:          # too autocorrelated -> ρ_i unreliable, drop the subject
            n_excluded_autocorr += 1
            continue
        r = stats.spearmanr(fi, ti).statistic
        if np.isfinite(r):
            rhos.append(float(r))
            eff_pairs.append(int(n_eff))
            within_ac.append(float(abs(r1f) + abs(r1t)) / 2.0)
    n_subj = len(rhos)
    base = {"n_subjects": n_subj, "n_subjects_excluded_autocorr": n_excluded_autocorr,
            "median_effective_pairs": (float(np.median(eff_pairs)) if eff_pairs else math.nan),
            "median_within_autocorr": (float(np.median(within_ac)) if within_ac else math.nan)}
    if n_subj < min_subjects:
        return {"within_rho_median": math.nan, "p_value": 1.0, "frac_consistent_sign": math.nan, **base}
    arr = np.asarray(rhos, dtype=float)
    med = float(np.median(arr))
    # one-sample test that the per-subject correlations are centered away from 0
    try:
        if np.allclose(arr, 0.0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        p = 1.0
    frac = float(np.mean(np.sign(arr) == np.sign(med))) if med != 0 else 0.5
    return {"within_rho_median": med, "p_value": p, "frac_consistent_sign": frac, **base}


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    x_arr, y_arr = finite_xy(x, y)
    if len(x_arr) < 3 or is_constant(x_arr) or is_constant(y_arr):
        return math.nan, math.nan, int(len(x_arr))
    result = stats.spearmanr(x_arr, y_arr)
    return float(result.statistic), float(result.pvalue), int(len(x_arr))


def spearman_rho_fast(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """Spearman rho ONLY (no p-value), for hot resample loops (bootstrap/permutation).

    Spearman rho == Pearson r of the average-ranks, which is exactly what scipy.stats.spearmanr
    computes — so this returns a value bit-identical to safe_spearman(x, y)[0], but skips scipy's
    discarded p-value machinery (beta/permutation), which dominated discovery runtime on wide data.
    Inputs MUST already be finite, equal-length float arrays (callers pass dropna'd data). Returns
    nan for n<3 or a constant resample, matching safe_spearman's guards so callers filter the same."""
    n = len(x_arr)
    if n < 3:
        return math.nan
    rx = stats.rankdata(x_arr)
    ry = stats.rankdata(y_arr)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return math.nan
    return float(np.dot(rx - rx.mean(), ry - ry.mean()) / (n * sx * sy))


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    x_arr, y_arr = finite_xy(x, y)
    if len(x_arr) < 3 or is_constant(x_arr) or is_constant(y_arr):
        return math.nan, math.nan, int(len(x_arr))
    statistic, p_value = stats.pearsonr(x_arr, y_arr)
    return float(statistic), float(p_value), int(len(x_arr))


def safe_kendall(x: Iterable[float], y: Iterable[float]) -> tuple[float, float, int]:
    x_arr, y_arr = finite_xy(x, y)
    if len(x_arr) < 3 or is_constant(x_arr) or is_constant(y_arr):
        return math.nan, math.nan, int(len(x_arr))
    result = stats.kendalltau(x_arr, y_arr)
    return float(result.statistic), float(result.pvalue), int(len(x_arr))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray([1.0 if not np.isfinite(value) else value for value in p_values], dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running_min = min(running_min, ranked[i] * n / rank)
        adjusted[order[i]] = running_min
    return [float(min(1.0, value)) for value in adjusted]


def same_sign(a: float, b: float) -> bool:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12 or abs(b) < 1e-12:
        return False
    return np.sign(a) == np.sign(b)


def signed_direction(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return "flat"
    return "positive" if value > 0 else "negative"


def auc_against_target(y: Iterable[float], scores: Iterable[float]) -> tuple[float, int]:
    y_arr = np.asarray(y, dtype=float)
    score_arr = np.asarray(scores, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(score_arr)
    y_arr = y_arr[mask]
    score_arr = score_arr[mask]
    if len(y_arr) < 10 or is_constant(y_arr) or is_constant(score_arr):
        return math.nan, int(len(y_arr))

    unique = np.unique(y_arr)
    if len(unique) == 2:
        labels = (y_arr == unique.max()).astype(int)
    else:
        labels = (y_arr >= np.nanmedian(y_arr)).astype(int)
    if len(np.unique(labels)) < 2:
        return math.nan, int(len(y_arr))
    auc = float(roc_auc_score(labels, score_arr))
    return max(auc, 1.0 - auc), int(len(y_arr))


def partial_spearman(
    x: Iterable[float],
    y: Iterable[float],
    covariates: np.ndarray,
) -> tuple[float, float, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    cov = np.asarray(covariates, dtype=float)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.all(np.isfinite(cov), axis=1)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    cov = cov[mask]
    if len(x_arr) < max(10, cov.shape[1] + 5):
        return math.nan, math.nan, int(len(x_arr))

    ranked_x = stats.rankdata(x_arr).reshape(-1, 1)
    ranked_y = stats.rankdata(y_arr).reshape(-1, 1)
    ranked_cov = np.column_stack([stats.rankdata(cov[:, i]) for i in range(cov.shape[1])])

    model_x = LinearRegression().fit(ranked_cov, ranked_x)
    model_y = LinearRegression().fit(ranked_cov, ranked_y)
    resid_x = (ranked_x - model_x.predict(ranked_cov)).ravel()
    resid_y = (ranked_y - model_y.predict(ranked_cov)).ravel()
    return safe_spearman(resid_x, resid_y)
