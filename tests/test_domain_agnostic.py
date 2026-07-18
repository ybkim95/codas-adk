"""The engine must decide feature construction and confounder handling from data, never from column
names. These tests pin that contract so a future change cannot quietly reintroduce dataset-specific
or disease-specific behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.core.data import _clock_hour_feature
from codas.core.discovery import DiscoveryRequest, run_discovery
from codas.core.validation import _confounder_covariate_matrix


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


def test_bounded_integer_confounder_is_not_cyclic_encoded():
    # A count / likert confounder (0..6) must be adjusted as an ordinary covariate, never guessed to be
    # a cyclic hour-of-day-like variable and expanded into sin/cos terms. Value ranges alone cannot tell
    # a weekday from a child count, so the engine must not infer cyclicity at all.
    n = 140
    frame = pd.DataFrame({"likert": np.tile(np.arange(7), n // 7 + 1)[:n]})
    cov = _confounder_covariate_matrix(frame, ["likert"], frame.index)
    assert cov.shape[1] == 6  # one-hot of 7 integer levels (k-1), not a 2-column sin/cos Fourier pair


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
