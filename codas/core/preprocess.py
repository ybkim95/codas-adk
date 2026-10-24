"""Feature scaling for parametric models, fitted once on the discovery share and frozen.

A hazard ratio "per SD" means something only if the SD is the same on both shares: estimated on the holdout,
it would rescale every effect by the holdout's own spread. So each feature's transform is fitted on the
discovery share, written into the pre-registration, and applied unchanged to the holdout:

1. clip at the discovery 0.5th and 99.5th percentiles (a few extreme values cannot drive a linear or Cox fit);
2. take the log of a strictly positive feature whose clipped skew exceeds ``SKEW_FOR_LOG`` (a right-skewed
   level such as a step count acts multiplicatively);
3. centre and scale by the discovery mean and SD of the result.

Rank-based statistics need none of this and are computed on the raw values. Every transform is described in
words for the report, and the unclipped estimate stays available as a sensitivity analysis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

CLIP = (0.005, 0.995)
SKEW_FOR_LOG = 2.0
MIN_VALUES = 40  # fewer values than this: centre and scale only


def fit(values: pd.Series) -> dict[str, Any]:
    """The transform of one feature, from its discovery values."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return {"lo": None, "hi": None, "log": False, "center": 0.0, "scale": 1.0}
    lo, hi = (float(np.quantile(x, CLIP[0])), float(np.quantile(x, CLIP[1]))) if len(x) >= MIN_VALUES else (None, None)
    clipped = np.clip(x, lo, hi) if lo is not None else x
    log = bool(lo is not None and lo > 0 and np.std(clipped) > 0 and stats.skew(clipped) > SKEW_FOR_LOG)
    scaled = np.log(clipped) if log else clipped
    sd = float(np.std(scaled, ddof=1))
    return {"lo": lo, "hi": hi, "log": log, "center": float(np.mean(scaled)), "scale": sd if sd > 0 else 1.0}


def fit_all(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    return {c: fit(frame[c]) for c in columns if c in frame.columns}


def apply(values: pd.Series, transform: dict[str, Any] | None, *, clip: bool = True) -> pd.Series:
    """``values`` on the frozen scale (standardised units of the discovery share). Without a transform the
    values are standardised by their own mean and SD (the behaviour before frozen transforms)."""
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if not transform:
        sd = x.std(ddof=0)
        return (x - x.mean()) / (sd if sd and math.isfinite(sd) else 1.0)
    if clip and transform.get("lo") is not None:
        x = x.clip(transform["lo"], transform["hi"])
    if transform.get("log"):
        x = np.log(x.where(x > 0))
    return (x - transform["center"]) / transform["scale"]


def describe(transform: dict[str, Any]) -> str:
    """The transform in words, for the pre-registration and the report."""
    steps = []
    if transform.get("lo") is not None:
        steps.append(f"clipped to [{transform['lo']:.4g}, {transform['hi']:.4g}] (discovery 0.5th-99.5th percentiles)")
    if transform.get("log"):
        steps.append("log-transformed (skew above %g)" % SKEW_FOR_LOG)
    steps.append(f"one unit = {transform['scale']:.4g} (the discovery SD{' of the log' if transform.get('log') else ''})")
    return "; ".join(steps)
