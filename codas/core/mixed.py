"""Within-person mixed models for repeated outcome measurements.

One row per measurement occasion: the outcome measured on that occasion and the exposure summarised
over the period just before it, with a person index. The exposure is split into the person's own
mean (between-person part) and the occasion's deviation from it (within-person part), and a linear
mixed model with a random intercept per person is fitted:

    y_ij = b0 + b_within * (x_ij - mean_i(x)) + b_between * mean_i(x) + covariates + u_i + e_ij

The within-person effect compares a person with themselves, so stable differences between people
(adherence, body size, socioeconomic position, genetics) cannot produce it: a feature that only tracks
such a trait shows a between-person effect and a within-person effect near zero. The decomposition is
the hybrid (within-between) model of Allison (2009) and Bell & Jones (2015). Fitted with statsmodels
``MixedLM`` (REML). Results are aggregate.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

_Z = 1.959964


def _term(fit: Any, k: int) -> dict[str, Any]:
    est, se = float(fit.params.iloc[k]), float(fit.bse.iloc[k])
    p = float(fit.pvalues.iloc[k]) if np.isfinite(fit.pvalues.iloc[k]) else math.nan
    return {"estimate": est, "se": se, "p": p, "ci95": [est - _Z * se, est + _Z * se]}


def mixed_model(panel: pd.DataFrame, feature: str, target: str, covariates: list[str] | tuple[str, ...] = (),
                person: str = "person_index", min_repeats: int = 2) -> dict[str, Any]:
    """Within- and between-person effects of ``feature`` on ``target`` (per SD of the feature's
    within-person variation and per SD of its person means), random intercept per person."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    cols = [person, feature, target, *covariates]
    data = panel[cols].apply(pd.to_numeric, errors="coerce").dropna()
    counts = data.groupby(person)[target].transform("size")
    repeats = int(data.loc[counts >= min_repeats, person].nunique())
    if repeats < 30:
        return {"feature": feature, "error": f"too few people with {min_repeats}+ measurements for a "
                                             f"within-person model ({repeats})"}
    mean = data.groupby(person)[feature].transform("mean")
    within, between = data[feature] - mean, mean
    sd_w, sd_b = float(within.std(ddof=0)), float(between.std(ddof=0))
    if sd_w == 0 or sd_b == 0:
        return {"feature": feature, "error": "the feature does not vary within or between people"}
    exog = pd.DataFrame({"const": 1.0, "within": within / sd_w, "between": (between - between.mean()) / sd_b})
    for c in covariates:
        exog[c] = data[c]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter; non-convergence is reported below
        fit = MixedLM(data[target], exog, groups=data[person]).fit(reml=True)
    var_u, var_e = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    return {"feature": feature, "model": "linear mixed model, random intercept per person (REML)",
            "within": _term(fit, 1), "between": _term(fit, 2), "icc": var_u / (var_u + var_e) if var_u + var_e > 0 else math.nan,
            "persons": int(data[person].nunique()), "rows": int(len(data)), "persons_with_repeats": repeats,
            "converged": bool(getattr(fit, "converged", True)),
            "measure": "outcome units per SD of within-person / between-person variation in the feature"}
