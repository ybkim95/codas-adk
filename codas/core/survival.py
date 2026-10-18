"""Cox proportional-hazards regression for time-to-event outcomes.

Newton-Raphson on the partial likelihood with Efron's handling of tied event times, the default in
most survival software. Standard errors come from the inverse observed information. Used by the
confirmation step, where a pre-registered exposure is tested on held-out data; no other part of the
engine depends on it.
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

    def hazard_ratio(self, j: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
        """HR and its (1 - alpha) Wald confidence interval for column ``j``."""
        z = stats.norm.ppf(1 - alpha / 2)
        b, s = float(self.coef[j]), float(self.se[j])
        return math.exp(b), math.exp(b - z * s), math.exp(b + z * s)


def _efron(beta: np.ndarray, time: np.ndarray, event: np.ndarray, x: np.ndarray):
    """Log partial likelihood, gradient and Hessian (Efron ties). Rows are sorted by descending time."""
    k = x.shape[1]
    eta = x @ beta
    eta -= eta.max()  # the likelihood is invariant to a shift; this keeps exp() finite
    w = np.exp(eta)
    loglik, grad, hess = 0.0, np.zeros(k), np.zeros((k, k))
    risk0, risk1, risk2 = 0.0, np.zeros(k), np.zeros((k, k))
    i, n = 0, len(time)
    while i < n:
        t = time[i]
        j = i
        tie0, tie1, tie2 = 0.0, np.zeros(k), np.zeros((k, k))
        d, xsum = 0, np.zeros(k)
        while j < n and time[j] == t:  # everyone with this time enters the risk set together
            wx = w[j] * x[j]
            risk0 += w[j]
            risk1 += wx
            risk2 += np.outer(wx, x[j])
            if event[j]:
                d += 1
                xsum += x[j]
                loglik += eta[j]
                tie0 += w[j]
                tie1 += wx
                tie2 += np.outer(wx, x[j])
            j += 1
        for m in range(d):
            frac = m / d
            s0 = risk0 - frac * tie0
            s1 = risk1 - frac * tie1
            s2 = risk2 - frac * tie2
            loglik -= math.log(s0)
            mean = s1 / s0
            grad -= mean
            hess -= s2 / s0 - np.outer(mean, mean)
        grad += xsum
        i = j
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
        return CoxResult(nan, nan, nan, int(len(time)), int(event.sum()), False)
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
        se = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.norm.sf(np.abs(beta / se))
    return CoxResult(beta, se, p, int(len(time)), int(event.sum()), converged)


def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value for a risk ratio (a hazard ratio approximates it for a rare outcome):
    the minimum strength of association an unmeasured confounder would need with both exposure and
    outcome to explain the estimate away."""
    if not np.isfinite(rr) or rr <= 0:
        return math.nan
    rr = rr if rr >= 1 else 1 / rr
    return rr + math.sqrt(rr * (rr - 1))
