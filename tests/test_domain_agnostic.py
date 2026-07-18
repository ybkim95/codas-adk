"""The engine must decide feature construction and confounder handling from data, never from column
names. These tests pin that contract so a future change cannot quietly reintroduce dataset-specific
or disease-specific behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.core.data import _clock_hour_feature
from codas.core.discovery import DiscoveryRequest, run_discovery
from codas.core.validation import _infer_cyclic_period


def test_clock_hour_is_decided_from_values_not_names():
    # An arbitrarily named column of timestamps still yields a clock-hour feature.
    stamps = pd.Series(pd.date_range("2020-01-01 06:30", periods=60, freq="83min").astype(str), name="q7")
    feature = _clock_hour_feature(stamps)
    assert feature is not None and feature.dropna().nunique() > 1

    # A column whose name screams "sleep_time" but holds no timestamps yields nothing.
    assert _clock_hour_feature(pd.Series(["red", "green", "blue"] * 20, name="sleep_time")) is None
    # Numeric and date-only columns carry no time-of-day signal.
    assert _clock_hour_feature(pd.Series(np.arange(60.0))) is None
    assert _clock_hour_feature(pd.Series(pd.date_range("2020-01-01", periods=60, freq="D").astype(str))) is None


def test_cyclic_period_is_inferred_from_values_not_names():
    assert _infer_cyclic_period(pd.Series(np.tile(np.arange(24), 6))) == 24.0     # hour of day
    assert _infer_cyclic_period(pd.Series(np.tile(np.arange(7), 20))) == 7.0       # weekday
    assert _infer_cyclic_period(pd.Series(np.tile(np.arange(1, 13), 10))) == 12.0  # month
    # A continuous covariate and an unbounded integer (age) are not cyclic.
    assert _infer_cyclic_period(pd.Series(np.random.default_rng(0).normal(size=400))) is None
    assert _infer_cyclic_period(pd.Series(np.random.default_rng(0).integers(18, 91, 400))) is None


def test_subgroup_consistency_does_not_degenerate_on_a_binary_target():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)  # genuine classification signal
    df = pd.DataFrame({"feat": x, "noise": rng.normal(size=n), "y": y})
    report = run_discovery(df, DiscoveryRequest(target_column="y", validation_resamples=200))
    feat = next(c for c in report.candidates if c.feature == "feat")
    subgroup = next(t for t in feat.tests if t.name == "subgroup_consistency")
    # The check must actually run (a target-median split would collapse to a constant class) and hold.
    assert subgroup.applicable and subgroup.passed
