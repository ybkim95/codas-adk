"""The phenotype language over a person-day table (codas.core.phenotype_dsl), hand-checked."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from codas.core import phenotype_dsl as dsl


def _days() -> pd.DataFrame:
    """Two people. Person 0: 20 consecutive days, steps rising by 100 a day from 5000, sleep alternating 300 and
    420 minutes, resting heart rate 60 + 0.05 * (420 - the previous night's sleep). Person 1: only 10 days."""
    rows = []
    for d in range(20):
        sleep = 300 if d % 2 == 0 else 420
        rows.append({"row": 0, "day": -20 + d, "weekday": (d % 7) + 1, "steps": 5000 + 100 * d, "sleep": sleep})
    for r in rows[1:]:
        r["rhr"] = 60 + 0.05 * (420 - next(p["sleep"] for p in rows if p["day"] == r["day"] - 1))
    rows[0]["rhr"] = 60.0
    for d in range(10):
        rows.append({"row": 1, "day": -10 + d, "weekday": (d % 7) + 1, "steps": 8000.0, "sleep": 400, "rhr": 55.0})
    return pd.DataFrame(rows)


def test_each_summary_by_hand():
    days = _days()
    steps = 5000 + 100 * np.arange(20)

    def one(**definition):
        return dsl.compute(days, {"name": "steps_x", "signal": "steps", "min_days": 5, **definition})

    assert one(summary="mean").loc[0] == steps.mean() and one(summary="mean").loc[1] == 8000
    assert one(summary="sd").loc[0] == pytest.approx(steps.std(ddof=1))
    assert one(summary="cv").loc[0] == pytest.approx(steps.std(ddof=1) / steps.mean())
    assert one(summary="slope_per_30d").loc[0] == pytest.approx(3000)          # 100 a day
    assert one(summary="mean_abs_change").loc[0] == pytest.approx(100)
    weekend = days[(days.row == 0) & days.weekday.isin([1, 7])].steps.mean()
    weekday = days[(days.row == 0) & ~days.weekday.isin([1, 7])].steps.mean()
    assert one(summary="weekend_minus_weekday").loc[0] == pytest.approx(weekend - weekday)
    assert one(summary="fraction", fraction={"op": ">=", "value": 6000}).loc[0] == pytest.approx(10 / 20)
    # Too few days: missing.
    assert math.isnan(dsl.compute(days, {"name": "steps_x", "signal": "steps", "summary": "mean", "min_days": 14}).loc[1])


def test_previous_day_conditions_and_coupling():
    days = _days()
    # Resting heart rate after a short night (< 360 min) against after a long one.
    short = dsl.compute(days, {"name": "rhr_after_short", "signal": "rhr", "summary": "mean", "min_days": 5,
                               "where": [{"column": "prev:sleep", "op": "<", "value": 360}]})
    long = dsl.compute(days, {"name": "rhr_after_long", "signal": "rhr", "summary": "mean", "min_days": 5,
                              "where": [{"column": "prev:sleep", "op": ">=", "value": 360}]})
    assert short.loc[0] == pytest.approx(66.0) and long.loc[0] == pytest.approx(60.0)
    # Reactivity: the within-person slope of resting heart rate on the previous night's sleep.
    slope = dsl.compute(days, {"name": "rhr_on_sleep", "signal": "rhr", "summary": "coupling", "min_days": 5,
                               "couple": {"with": "sleep", "lag": 1}})
    assert slope.loc[0] == pytest.approx(-0.05) and math.isnan(slope.loc[1])   # person 1 never varies its sleep
    # A day whose previous calendar day is missing has no previous value.
    gap = _days().query("not (row == 0 and day == -15)")
    used = dsl.compute(gap, {"name": "rhr_n", "signal": "rhr", "summary": "mean", "min_days": 3,
                             "where": [{"column": "prev:sleep", "op": ">", "value": 0}]})
    assert used.loc[0] == pytest.approx(gap[(gap.row == 0) & (gap.day > -20) & (gap.day != -14)].rhr.mean())


def test_ratios_validation_identity_and_words():
    days = _days()
    ratio = dsl.compute(days, {"name": "steps_per_sleep", "signal": {"op": "ratio", "a": "steps", "b": "sleep"},
                               "summary": "mean", "min_days": 5})
    assert ratio.loc[1] == pytest.approx(20.0)
    allowed = set(dsl.signal_columns(days))
    assert allowed == {"steps", "sleep", "rhr"}
    bad = dsl.validate({"name": "Bad name", "signal": "row", "summary": "median", "min_days": 1,
                        "where": [{"column": "nowhere", "op": "<", "value": 1}]}, allowed)
    assert len(bad) == 5
    assert dsl.validate({"name": "cc", "signal": "rhr", "summary": "coupling", "couple": {"with": "x", "lag": 2}},
                        allowed) and dsl.validate({"name": "ff", "signal": "rhr", "summary": "fraction"}, allowed)
    a = {"name": "one", "signal": "steps", "summary": "sd", "min_days": 14}
    assert dsl.digest(a) == dsl.digest({**a, "name": "two"}) != dsl.digest({**a, "summary": "mean"})
    assert dsl.column_name(a) == f"dsl_one_{dsl.digest(a)}"
    assert dsl.describe({"name": "rr", "signal": "rhr", "summary": "coupling", "couple": {"with": "sleep", "lag": 1},
                         "where": [{"column": "prev:sleep", "op": "<", "value": 360}]}) == (
        "within-person slope of rhr on sleep the day before, over days with previous day sleep < 360 "
        "(at least 14 days)")
    with pytest.raises(ValueError):
        dsl.compute(days, {"name": "steps_x", "signal": "nowhere", "summary": "mean"})
