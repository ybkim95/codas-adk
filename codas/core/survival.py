"""Cox proportional-hazards regression for time-to-event outcomes.

Newton-Raphson on the partial likelihood with Efron's handling of tied event times, the default in
most survival software. Standard errors come from the inverse observed information. Used by the
confirmation step, where a pre-registered exposure is tested on held-out data, and by the agents'
analysis tools (``codas.core.analyses``); the discovery screen does not depend on it.

For screens over many columns (``codas.aou.phenome``) there is also a vectorised fit under Breslow's ties
(``breslow_fit``) and the score test of each column against one such base fit (``score_tests``).

With ``entry`` times the model is fitted in counting-process form: a row is at risk from its entry to its time
(left truncation), so a person split into intervals with the exposure of each interval gives a time-varying
exposure; one person's unsplit row and the same follow-up split into intervals of equal covariates give the
same fit. The risk-set sums are then taken segment by segment, so memory stays linear in the rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class CoxResult:
    coef: np.ndarray          # log hazard ratios, one per column of X
    se: np.ndarray
    p: np.ndarray             # two-sided Wald p-values
    n: int                    # rows used (complete cases)
    events: int
    converged: bool
    cov: np.ndarray | None = None  # inverse observed information (covariance of coef)
    dropped: list[int] = field(default_factory=list)  # columns left out: constant or collinear with earlier ones

    def hazard_ratio(self, j: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
        """HR and its (1 - alpha) Wald confidence interval for column ``j``."""
        z = stats.norm.ppf(1 - alpha / 2)
        b, s = float(self.coef[j]), float(self.se[j])
        return math.exp(b), math.exp(b - z * s), math.exp(b + z * s)


def _efron(beta: np.ndarray, time: np.ndarray, event: np.ndarray, x: np.ndarray):
    """Log partial likelihood, gradient and Hessian (Efron ties). Rows are sorted by descending time.

    Risk-set sums are cumulative sums over the sorted rows, read at the end of each block of equal
    times, so only the distinct event times are looped over. A coefficient so large that a risk set's
    weights all underflow gives a log-likelihood of -inf, so the caller's step-halving rejects it."""
    k = x.shape[1]
    eta = x @ beta
    eta -= eta.max()  # the likelihood is invariant to a shift; this keeps exp() finite
    w = np.exp(eta)
    wx = w[:, None] * x
    risk0 = np.cumsum(w)
    risk1 = np.cumsum(wx, axis=0)
    risk2 = np.cumsum(wx[:, :, None] * x[:, None, :], axis=0)
    ends = np.flatnonzero(np.r_[time[1:] != time[:-1], True])  # last row of each block of equal times
    starts = np.r_[0, ends[:-1] + 1]
    loglik, grad, hess = 0.0, np.zeros(k), np.zeros((k, k))
    for start, end in zip(starts, ends):
        ev = np.flatnonzero(event[start:end + 1]) + start
        d = len(ev)
        if not d:
            continue
        tie0, tie1 = w[ev].sum(), wx[ev].sum(axis=0)
        tie2 = (wx[ev][:, :, None] * x[ev][:, None, :]).sum(axis=0)
        loglik += eta[ev].sum()
        grad += x[ev].sum(axis=0)
        for m in range(d):
            frac = m / d
            s0 = risk0[end] - frac * tie0
            s1 = risk1[end] - frac * tie1
            s2 = risk2[end] - frac * tie2
            if not s0 > 0:  # exp() underflowed over this risk set: a step far past the optimum
                return -math.inf, grad, hess
            loglik -= math.log(s0)
            mean = s1 / s0
            grad -= mean
            hess -= s2 / s0 - np.outer(mean, mean)
    return loglik, grad, hess


def _prefix_sums(w: np.ndarray, x: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sums of w, w x and w x x' over the first ``ends[j]`` rows, for each j (``ends`` ascending), taken segment
    by segment so no (rows, k, k) array is built."""
    k = x.shape[1]
    s0, s1, s2 = np.zeros(len(ends)), np.zeros((len(ends), k)), np.zeros((len(ends), k, k))
    a0, a1, a2, last = 0.0, np.zeros(k), np.zeros((k, k)), 0
    for j, end in enumerate(ends):
        if end > last:
            seg_w, seg_x = w[last:end], x[last:end]
            a0 += seg_w.sum()
            a1 = a1 + seg_w @ seg_x
            a2 = a2 + (seg_x * seg_w[:, None]).T @ seg_x
            last = end
        s0[j], s1[j], s2[j] = a0, a1, a2
    return s0, s1, s2


def _efron_counting(beta: np.ndarray, time: np.ndarray, event: np.ndarray, x: np.ndarray, entry: np.ndarray):
    """``_efron`` for rows at risk from ``entry`` to ``time`` (rows sorted by descending time): the risk set at t
    is every row with time >= t less every row that entered at or after t."""
    k = x.shape[1]
    eta = x @ beta
    eta -= eta.max()
    w = np.exp(eta)
    wx = w[:, None] * x
    ends = np.flatnonzero(np.r_[time[1:] != time[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    blocks = [(a, b) for a, b in zip(starts, ends) if event[a:b + 1].any()]
    if not blocks:
        return 0.0, np.zeros(k), np.zeros((k, k))
    at = np.asarray([b + 1 for _, b in blocks])                      # rows with time >= t: a prefix
    r0, r1, r2 = _prefix_sums(w, x, at)
    by_entry = np.argsort(-entry, kind="mergesort")
    entered = entry[by_entry]
    late = np.searchsorted(-entered, -time[[b for _, b in blocks]], side="right")  # rows with entry >= t
    order = np.argsort(late, kind="mergesort")
    l0, l1, l2 = (np.empty_like(v) for v in (r0, r1, r2))
    l0[order], l1[order], l2[order] = _prefix_sums(w[by_entry], x[by_entry], late[order])
    loglik, grad, hess = 0.0, np.zeros(k), np.zeros((k, k))
    for j, (start, end) in enumerate(blocks):
        ev = np.flatnonzero(event[start:end + 1]) + start
        d = len(ev)
        tie0, tie1 = w[ev].sum(), wx[ev].sum(axis=0)
        tie2 = (wx[ev][:, :, None] * x[ev][:, None, :]).sum(axis=0)
        loglik += eta[ev].sum()
        grad += x[ev].sum(axis=0)
        for m in range(d):
            frac = m / d
            s0 = r0[j] - l0[j] - frac * tie0
            s1 = r1[j] - l1[j] - frac * tie1
            s2 = r2[j] - l2[j] - frac * tie2
            if not s0 > 0:
                return -math.inf, grad, hess
            loglik -= math.log(s0)
            mean = s1 / s0
            grad -= mean
            hess -= s2 / s0 - np.outer(mean, mean)
    return loglik, grad, hess


def _independent_columns(x: np.ndarray) -> list[int]:
    """Columns that are neither constant nor a linear combination of earlier columns, in order: a
    covariate repeated under two names (two sources counting the same days) makes the information
    matrix singular, and the model is identified only without it."""
    sd = x.std(axis=0)
    z = np.where(sd > 0, (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    kept: list[int] = []
    for j in range(x.shape[1]):
        if sd[j] > 0 and np.linalg.matrix_rank(z[:, kept + [j]]) == len(kept) + 1:
            kept.append(j)
    return kept


def _newton(time: np.ndarray, event: np.ndarray, x: np.ndarray, max_iter: int, tol: float,
            entry: np.ndarray | None = None):
    k = x.shape[1]
    beta = np.zeros(k)
    _efron_ = _efron if entry is None else (lambda b, t, e, z: _efron_counting(b, t, e, z, entry))
    loglik, grad, hess = _efron_(beta, time, event, x)
    converged = False
    for _ in range(max_iter):
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            break
        new = beta + step
        new_loglik, new_grad, new_hess = _efron_(new, time, event, x)
        halvings = 0
        while new_loglik < loglik - 1e-12 and halvings < 20:  # step-halving keeps the ascent monotone
            step /= 2
            new = beta + step
            new_loglik, new_grad, new_hess = _efron_(new, time, event, x)
            halvings += 1
        if not np.isfinite(new_loglik):  # no finite step left: keep the last estimate, not converged
            break
        beta, grad, hess = new, new_grad, new_hess
        if abs(new_loglik - loglik) < tol * (abs(loglik) + tol):
            converged = True
            break
        loglik = new_loglik
    try:
        cov = np.linalg.inv(-hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
        se = np.full(k, np.nan)
    return beta, se, cov, converged


def cox_ph(time, event, x, max_iter: int = 50, tol: float = 1e-9, entry=None) -> CoxResult:
    """Fit a Cox model. ``x`` is an (n, k) array of covariates; rows with any missing value are dropped.
    Columns that are constant or collinear with earlier ones are left out (``dropped``, estimates NaN);
    column 0, the exposure of interest, is never displaced by a later column. With ``entry`` each row is at
    risk from its entry to its time (counting-process form; ``n`` then counts rows, not people)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    keep = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(x), axis=1) & (time > 0)
    if entry is not None:
        entry = np.asarray(entry, dtype=float)
        keep &= np.isfinite(entry) & (entry < time)
        entry = entry[keep]
    time, event, x = time[keep], event[keep].astype(bool), x[keep]
    k = x.shape[1]
    nan = np.full(k, np.nan)
    cols = _independent_columns(x) if len(time) else []
    if len(time) < len(cols) + 2 or event.sum() < 1 or 0 not in cols:
        return CoxResult(nan, nan, nan, int(len(time)), int(event.sum()), False, np.full((k, k), np.nan),
                         [j for j in range(k) if j not in cols])
    order = np.argsort(-time, kind="mergesort")
    time, event, x = time[order], event[order], x[order]
    b, s, c, converged = _newton(time, event, x[:, cols], max_iter, tol,
                                 None if entry is None else entry[order])
    beta, se, cov = nan.copy(), nan.copy(), np.full((k, k), np.nan)
    beta[cols], se[cols] = b, s
    cov[np.ix_(cols, cols)] = c
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.norm.sf(np.abs(beta / se))
    return CoxResult(beta, se, p, int(len(time)), int(event.sum()), converged, cov,
                     [j for j in range(k) if j not in cols])


def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value for a risk ratio (a hazard ratio approximates it for a rare outcome):
    the minimum strength of association an unmeasured confounder would need with both exposure and
    outcome to explain the estimate away."""
    if not np.isfinite(rr) or rr <= 0:
        return math.nan
    rr = rr if rr >= 1 else 1 / rr
    return rr + math.sqrt(rr * (rr - 1))


# --- screening: one base fit per outcome, then a score test per column ------------------------------
# Under Breslow's ties the information sums over event times collapse to weighted sums over rows: a row's
# weight c is the sum of d_b / S0_b over the event blocks whose risk set holds it, so sum_b d_b S2_b / S0_b
# = z' diag(c w) z, one matrix product, and no (rows x columns x columns) array is ever formed.

def _tie_blocks(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows sorted by descending time: the last row of each block of equal times that holds an event
    (its risk set is every row up to it) and the number of events in the block."""
    ends = np.flatnonzero(np.r_[time[1:] != time[:-1], True]) if len(time) else np.zeros(0, dtype=int)
    d = np.add.reduceat(event.astype(float), np.r_[0, ends[:-1] + 1]) if len(ends) else np.zeros(0)
    keep = d > 0
    return ends[keep], d[keep]


def _row_weights(ends: np.ndarray, d: np.ndarray, s0: np.ndarray, n: int) -> np.ndarray:
    """c_r = sum of d_b / S0_b over the blocks b whose risk set (rows 0..end_b) holds row r."""
    tail = np.cumsum((d / s0)[::-1])[::-1]
    first = np.searchsorted(ends, np.arange(n))  # the first block whose end is at or after the row
    return np.where(first < len(ends), tail[np.minimum(first, len(ends) - 1)], 0.0)


def breslow_fit(time: Any, event: Any, z: Any, max_iter: int = 50, tol: float = 1e-9) -> np.ndarray:
    """Cox coefficients of ``z`` under Breslow's handling of ties, by Newton-Raphson with step-halving.
    Rows must be complete and sorted by descending time. Risk-set sums are cumulative sums read at the end of
    each block of tied times, with no loop over event times: a screen fits one such base model per outcome
    and tests every column against it (``score_tests``)."""
    time, event, z = np.asarray(time, dtype=float), np.asarray(event, dtype=bool), np.asarray(z, dtype=float)
    ends, d = _tie_blocks(time, event)
    beta = np.zeros(z.shape[1])
    if not len(ends) or not z.shape[1]:
        return beta
    zsum = z[event].sum(axis=0)

    def likelihood(b: np.ndarray):
        eta = z @ b
        shift = eta.max()
        w = np.exp(eta - shift)
        s0 = np.cumsum(w)[ends]
        if not np.all(s0 > 0):
            return -math.inf, None, None
        mean = np.cumsum(w[:, None] * z, axis=0)[ends] / s0[:, None]
        c = _row_weights(ends, d, s0, len(time))
        loglik = float(eta[event].sum() - (d * (np.log(s0) + shift)).sum())
        grad = zsum - d @ mean
        hess = -(z.T @ ((c * w)[:, None] * z) - (mean * d[:, None]).T @ mean)
        return loglik, grad, hess

    loglik, grad, hess = likelihood(beta)
    for _ in range(max_iter):
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            break
        new = beta + step
        new_loglik, new_grad, new_hess = likelihood(new)
        halvings = 0
        while not new_loglik >= loglik - 1e-12 and halvings < 20:  # also rejects a step with -inf
            step /= 2
            new = beta + step
            new_loglik, new_grad, new_hess = likelihood(new)
            halvings += 1
        if not np.isfinite(new_loglik):
            break
        done = abs(new_loglik - loglik) < tol * (abs(loglik) + tol)
        beta, loglik, grad, hess = new, new_loglik, new_grad, new_hess
        if done:
            break
    return beta


def score_tests(time: Any, event: Any, z: Any, x: Any, beta: Any) -> list[dict[str, Any]]:
    """For each column of ``x``: the score test of adding it, standardised, to the Cox model of ``z`` fitted
    at ``beta`` (``breslow_fit``). Rows sorted by descending time; ``z`` complete; ``x`` may be missing.

    Neyman's C(alpha) form: the score for the column is taken net of its regression on the score for ``z``,
    so a ``beta`` fitted on every row stays valid on the rows where the column is observed. Returns per
    column the one-step log hazard ratio per SD (a Newton step from zero, close to the maximum-likelihood
    estimate for the modest effects a screen ranks), its z statistic, the two-sided p, and the rows and
    events used. Columns missing on the same rows are tested together."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    z, x = np.asarray(z, dtype=float), np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    eta = z @ np.asarray(beta, dtype=float)
    w_all = np.exp(eta - (eta.max() if len(eta) else 0.0))
    observed_all = np.isfinite(x)
    groups: dict[bytes, list[int]] = {}
    for j in range(x.shape[1]):
        groups.setdefault(np.packbits(observed_all[:, j]).tobytes(), []).append(j)
    out: list[dict[str, Any]] = [{} for _ in range(x.shape[1])]
    for cols in groups.values():
        observed = observed_all[:, cols[0]]
        t, e, zz, w = time[observed], event[observed], z[observed], w_all[observed]
        nan = {"log_hr": math.nan, "z": math.nan, "p": math.nan, "n": int(observed.sum()), "events": int(e.sum())}
        ends, d = _tie_blocks(t, e)
        v = x[observed][:, cols]
        sd = v.std(axis=0) if len(v) else np.zeros(len(cols))
        if len(ends) < 2:
            for j in cols:
                out[j] = dict(nan)
            continue
        v = (v - v.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        s0 = np.cumsum(w)[ends]
        cw = _row_weights(ends, d, s0, len(t)) * w
        mz = np.cumsum(w[:, None] * zz, axis=0)[ends] / s0[:, None]
        izz = zz.T @ (cw[:, None] * zz) - (mz * d[:, None]).T @ mz
        uz = zz[e].sum(axis=0) - d @ mz
        mx = np.cumsum(w[:, None] * v, axis=0)[ends] / s0[:, None]
        ux = v[e].sum(axis=0) - d @ mx
        ixx = (cw[:, None] * v * v).sum(axis=0) - d @ (mx * mx)
        ixz = v.T @ (cw[:, None] * zz) - (mx * d[:, None]).T @ mz
        a = ixz @ np.linalg.pinv(izz)
        score = ux - a @ uz
        info = ixx - (a * ixz).sum(axis=1)
        for k, j in enumerate(cols):
            if not (sd[k] > 0 and info[k] > 0):
                out[j] = dict(nan)
                continue
            stat = float(score[k] / math.sqrt(info[k]))
            out[j] = {**nan, "log_hr": float(score[k] / info[k]), "z": stat, "p": float(2 * stats.norm.sf(abs(stat)))}
    return out
