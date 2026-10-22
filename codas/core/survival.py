"""Cox proportional-hazards regression for time-to-event outcomes.

Newton-Raphson on the partial likelihood with Efron's handling of tied event times, the default in
most survival software. Standard errors come from the inverse observed information. Used by the
confirmation step, where a pre-registered exposure is tested on held-out data, and by the agents'
analysis tools (``codas.core.analyses``); the discovery screen does not depend on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def cox_ph(time, event, x, max_iter: int = 50, tol: float = 1e-9) -> CoxResult:
    """Fit a Cox model. ``x`` is an (n, k) array of covariates; rows with any missing value are dropped."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    keep = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(x), axis=1) & (time > 0)
    time, event, x = time[keep], event[keep].astype(bool), x[keep]
    k = x.shape[1]
    if len(time) < k + 2 or event.sum() < 1:
        nan = np.full(k, np.nan)
        return CoxResult(nan, nan, nan, int(len(time)), int(event.sum()), False, np.full((k, k), np.nan))
    order = np.argsort(-time, kind="mergesort")
    time, event, x = time[order], event[order], x[order]
    beta = np.zeros(k)
    loglik, grad, hess = _efron(beta, time, event, x)
    converged = False
    for _ in range(max_iter):
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            break
        new = beta + step
        new_loglik, new_grad, new_hess = _efron(new, time, event, x)
        halvings = 0
        while new_loglik < loglik - 1e-12 and halvings < 20:  # step-halving keeps the ascent monotone
            step /= 2
            new = beta + step
            new_loglik, new_grad, new_hess = _efron(new, time, event, x)
            halvings += 1
        if not np.isfinite(new_loglik):  # no finite step left: keep the last estimate, not converged
            break
        beta, grad, hess = new, new_grad, new_hess
        if abs(new_loglik - loglik) < tol * (abs(loglik) + tol):
            loglik = new_loglik
            converged = True
            break
        loglik = new_loglik
    try:
        cov = np.linalg.inv(-hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
        se = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.norm.sf(np.abs(beta / se))
    return CoxResult(beta, se, p, int(len(time)), int(event.sum()), converged, cov)


def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value for a risk ratio (a hazard ratio approximates it for a rare outcome):
    the minimum strength of association an unmeasured confounder would need with both exposure and
    outcome to explain the estimate away."""
    if not np.isfinite(rr) or rr <= 0:
        return math.nan
    rr = rr if rr >= 1 else 1 / rr
    return rr + math.sqrt(rr * (rr - 1))
