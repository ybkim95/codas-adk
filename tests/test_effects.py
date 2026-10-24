"""Effect sizes beyond a p-value (codas.core.effects): checked against hand calculations and planted effects."""

from __future__ import annotations

import math
from bisect import bisect_left, insort

import numpy as np
import pandas as pd

from codas.core import effects


def reference_concordance(time, event, risk) -> float:
    """Harrell's C the plain way (the implementation the fast one replaced): walk the times from the latest,
    keeping everyone later sorted by risk."""
    t, e, r = (np.asarray(v, dtype=float) for v in (time, event, risk))
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[ok], e[ok], r[ok]
    order = np.argsort(-t, kind="mergesort")
    seen: list[float] = []
    concordant = comparable = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and t[order[j]] == t[order[i]]:
            j += 1
        for k in order[i:j]:
            if e[k] and seen:
                below = bisect_left(seen, r[k])
                concordant += below + 0.5 * (bisect_left(seen, np.nextafter(r[k], np.inf)) - below)
                comparable += len(seen)
        for k in order[i:j]:
            insort(seen, r[k])
        i = j
    return concordant / comparable if comparable else math.nan


def test_the_fast_concordance_equals_the_reference_with_ties_and_gaps():
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(2, 90))
        t = rng.integers(0, 12, n).astype(float)
        e = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(float)
        r = rng.integers(0, 5, n).astype(float)
        t[rng.random(n) < 0.05] = np.nan
        a, b = reference_concordance(t, e, r), effects.concordance(t, e, r)
        assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12, trial
    # By hand: (time, event, risk) = (1, 1, 0.9), (2, 1, 0.5), (3, 0, 0.7). Pairs led by an event: (1 vs 2)
    # concordant, (1 vs 3) concordant, (2 vs 3) discordant: C = 2/3.
    assert effects.concordance([1, 2, 3], [1, 1, 0], [0.9, 0.5, 0.7]) == 2 / 3


def test_auc_r_squared_and_cumulative_incidence_by_hand():
    assert effects.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert effects.r_squared([1, 2, 3], [1, 2, 3]) == 1.0 and effects.r_squared([1, 2, 3], [2, 2, 2]) == 0.0
    # Kaplan-Meier: four people, events at 1 and 2 (one censored at 2): S(2) = 3/4 * 2/3 = 1/2.
    assert effects.cumulative_incidence([1, 2, 2, 3], [1, 1, 0, 1], 2) == 0.5
    assert effects.cumulative_incidence([5, 6], [1, 1], 2) == 0.0


def test_magnitude_classes_and_the_e_value_of_the_bound():
    assert [effects.magnitude("hr_per_sd", v) for v in (1.02, 0.9, 1.3, 0.6)] == ["negligible", "small", "moderate", "large"]
    assert [effects.magnitude("rho", v) for v in (0.05, -0.15, 0.25, 0.4)] == ["negligible", "small", "moderate", "large"]
    assert effects.magnitude("delta_c", 0.003) == "negligible" and effects.magnitude("rho", float("nan")) is None
    assert effects.e_value_bound(0.9, 1.1) == 1.0  # the interval includes the null
    assert abs(effects.e_value_bound(1.2, 1.6) - (1.2 + math.sqrt(1.2 * 0.2))) < 1e-12


def test_value_effects_are_in_the_outcomes_units_on_frozen_cutpoints():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(50, 10, n)
    age = rng.normal(size=n)
    y = 6.0 + 0.02 * x + 0.3 * age + rng.normal(0, 0.5, n)   # 0.02 outcome units per unit of x
    cov = pd.DataFrame({"age": age})
    cut = list(np.quantile(x, [0.25, 0.5, 0.75]))
    out = effects.value_effects(y, x, cov, sd=10.0, unit=5.0, cutpoints=cut)
    direct = np.linalg.lstsq(np.column_stack([np.ones(n), x, age]), y, rcond=None)[0][1]  # per unit of x
    assert abs(out["per_sd"] - 10 * direct) < 1e-3 and abs(out["per_sd"] - 0.2) < 0.03
    assert out["per_sd_ci95"][0] < 10 * direct < out["per_sd_ci95"][1]
    assert abs(out["per_unit"] - 5 * direct) < 1e-3          # per 5 units of x
    # Top vs bottom quarter: the mean x differs by about 2 * 1.27 SD = 25 units, so about 0.5 outcome units.
    assert 0.4 < out["top_vs_bottom_quarter"] < 0.6
    assert out["partial_delta_r2"] > 0.05 and out["magnitude"] in {"moderate", "large"}


def test_survival_effects_give_quarters_trend_and_absolute_risk():
    rng = np.random.default_rng(2)
    n = 8000
    x = rng.normal(size=n)
    t = rng.exponential(1 / (0.2 * np.exp(0.4 * x)))
    c = rng.exponential(5, n)
    time, event = np.minimum(t, c) * 365, (t <= c).astype(float)
    cut = list(np.quantile(x, [0.25, 0.5, 0.75]))
    out = effects.survival_effects(time, event, x, None, hr_per_sd=1.49, ci95=[1.42, 1.56], sd=1.0, unit=0.5,
                                   cutpoints=cut, horizon=3 * 365, draws=50)
    assert abs(out["hr_per_unit"] - math.sqrt(1.49)) < 1e-3 and out["magnitude"] == "moderate"
    assert out["quarter_hr"][0] == 1.0 and out["quarter_hr"][1] < out["quarter_hr"][2] < out["quarter_hr"][3]
    assert out["p_trend"] < 1e-10
    ci = out["cumulative_incidence"]
    assert ci == sorted(ci) and out["risk_difference"] == round(ci[3] - ci[0], 4) > 0.1
    assert out["risk_difference_ci95"][0] < out["risk_difference"] < out["risk_difference_ci95"][1]
    assert out["e_value_ci"] > 1
    # A quarter with 1-20 events gets no absolute risk (it would reveal the count), nor does the difference.
    few = event.copy()
    lowest = np.flatnonzero(effects.quarter(x, cut) == 0)
    few[lowest] = 0
    few[lowest[:5]] = 1
    masked = effects.survival_effects(time, few, x, None, hr_per_sd=1.49, ci95=[1.42, 1.56], sd=1.0, unit=None,
                                      cutpoints=cut, horizon=3 * 365, draws=10, min_events=20)
    assert masked["cumulative_incidence"][0] is None and masked["cumulative_incidence"][3] is not None
    assert "risk_difference" not in masked and "hr_per_unit" not in masked


def _survival_shares(seed, n=6000):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"age": rng.normal(size=n), "x": rng.normal(size=n), "inc": rng.normal(size=n)})
    frame["dup"] = frame.inc + 0.2 * rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.5 * frame.age + 0.4 * frame.x + 0.5 * frame.inc))
    c = rng.exponential(1.5, n)
    frame["days"], frame["y"] = np.ceil(np.minimum(t, c) * 365), (t <= c).astype(int)
    frame.loc[frame.index[:300], "age"] = np.nan          # filled with the discovery median, never the outcome
    return frame


def test_frozen_models_score_the_gain_and_the_head_to_head_on_the_holdout():
    discovery, holdout = _survival_shares(3), _survival_shares(4)
    frozen = effects.fit_frozen(discovery, target="y", time="days", covariates=["age"], feature="x", incumbent="inc")
    assert set(frozen["models"]) == {"base", "feature", "incumbent", "both"}
    out = effects.evaluate_frozen(frozen, holdout, target="y", time="days", draws=30)
    assert out["metric"] == "C" and out["feature"] > out["base"]
    assert out["gain_ci95"][0] > 0 and out["gain_beyond_incumbent"] > 0      # x adds beyond the incumbent
    # The incumbent measured again adds nothing beyond it, and loses nothing head to head.
    dup = effects.evaluate_frozen(effects.fit_frozen(discovery, target="y", time="days", covariates=["age"],
                                                     feature="dup", incumbent="inc"), holdout, target="y", time="days",
                                  draws=30)
    assert abs(dup["gain_beyond_incumbent"]) < 0.005 and dup["head_to_head"] < 0.005
    assert effects.evaluate_frozen(frozen, holdout, target="y", time="days", draws=0)["gain_ci95"] is None


def test_frozen_models_for_a_value_outcome_use_r_squared():
    rng = np.random.default_rng(5)
    frames = []
    for _ in range(2):
        f = pd.DataFrame({"age": rng.normal(size=3000), "x": rng.normal(size=3000)})
        f["y"] = 0.5 * f.age + 0.3 * f.x + rng.normal(size=3000)
        frames.append(f)
    frozen = effects.fit_frozen(frames[0], target="y", time=None, covariates=["age"], feature="x")
    out = effects.evaluate_frozen(frozen, frames[1], target="y", time=None, draws=30)
    assert out["metric"] == "R2" and 0.05 < out["gain"] < 0.12 and out["gain_ci95"][0] > 0


def test_detection_rate_by_hand():
    case = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([9, 8, 2, 1, 7, 6, 5, 4, 3, 2.5, 1.5, 1.2, 0.5, 0.1])
    # The 90th percentile of the ten non-cases' scores is 6.1: two of the four cases score above it.
    assert effects.detection_rate(case, score, 0.10) == 0.5
    assert math.isnan(effects.detection_rate(np.zeros(5), np.arange(5), 0.1))


def test_the_strong_baseline_and_detection_rates_on_the_holdout():
    """With the incumbent among the established markers, a feature that only repeats it adds nothing beyond the
    strong baseline, while a feature with its own effect still does; detection rates come at the horizon."""
    discovery, holdout = _survival_shares(3), _survival_shares(4)
    frozen = effects.fit_frozen(discovery, target="y", time="days", covariates=["age"], feature="x", baseline=["inc"])
    assert {"strong", "strong_feature"} <= set(frozen["models"]) and frozen["baseline"] == ["inc"]
    out = effects.evaluate_frozen(frozen, holdout, target="y", time="days", draws=30, horizon=365, min_count=20)
    assert out["baseline"] == ["inc"] and out["gain_beyond_strong_ci95"][0] > 0
    rates = out["detection"]["rates"]
    assert out["detection"]["false_positive_rates"] == [0.05, 0.10]
    assert rates["strong_feature"][1] > rates["strong"][1] and out["detection_gain_beyond_strong"] > 0
    assert rates["feature"][1] >= rates["feature"][0]           # more cases found at more false positives
    dup = effects.evaluate_frozen(effects.fit_frozen(discovery, target="y", time="days", covariates=["age"],
                                                     feature="dup", baseline=["inc"]),
                                  holdout, target="y", time="days", draws=30, horizon=365)
    assert abs(dup["gain_beyond_strong"]) < 0.005
    # No marker beyond the covariates: no strong baseline; no horizon: no detection rates.
    plain = effects.fit_frozen(discovery, target="y", time="days", covariates=["age"], feature="x", baseline=["age"])
    assert "strong" not in plain["models"]
    assert "detection" not in effects.evaluate_frozen(plain, holdout, target="y", time="days", draws=0)


def test_a_detection_rate_that_would_reveal_a_small_count_is_withheld():
    discovery, holdout = _survival_shares(3), _survival_shares(4)
    frozen = effects.fit_frozen(discovery, target="y", time="days", covariates=["age"], feature="x", baseline=["inc"])
    few = holdout.copy()
    few.loc[few.index[150:], "y"] = 0                        # a few dozen cases: some rate finds or misses 1-20
    out = effects.evaluate_frozen(frozen, few, target="y", time="days", draws=10, horizon=365, min_count=20)
    cases = int(((few.y > 0) & (few.days <= 365)).sum())
    assert 20 < cases < 100
    for values in out["detection"]["rates"].values():
        for v in values:
            assert v is None or not (0 < round(v * cases) <= 20 or 0 < cases - round(v * cases) <= 20)
    assert any(v is None for values in out["detection"]["rates"].values() for v in values)
    shown = out["detection"]["rates"]
    for gain, (a, b) in (("detection_gain", ("feature", "base")), ("detection_gain_beyond_strong",
                                                                    ("strong_feature", "strong"))):
        if gain in out:
            assert shown[a][-1] is not None and shown[b][-1] is not None
    # With 20 or fewer cases no rate is given at all.
    fewer = holdout.copy()
    fewer.loc[fewer.index[15:], "y"] = 0
    assert "detection" not in effects.evaluate_frozen(frozen, fewer, target="y", time="days", draws=0, horizon=365,
                                                      min_count=20)
