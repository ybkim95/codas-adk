"""Columns the analysis design takes out of the candidate pool stay out of every engineered feature,
and the held-out model's permutation null has enough draws to support a 0.05 decision.

Dataset-agnostic: arbitrary column names on synthetic data.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.agents.tools import propose_feature, set_target
from codas.core.data import build_analysis_frame
from codas.core.discovery import DiscoveryRequest, run_discovery


def _frame(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n)
    return pd.DataFrame({
        "y": y,
        "signal_mean": 0.5 * y + rng.normal(0, 1, n),
        "other_mean": rng.normal(0, 1, n),
        "third_mean": rng.normal(0, 1, n),
        "leak_mean": y + rng.normal(0, 0.05, n),      # encodes the target; excluded by the caller
        "conf_mean": 0.3 * y + rng.normal(0, 1, n),   # declared confounder
        "ctx_mean": 0.3 * y + rng.normal(0, 1, n),    # adjustment-only context column
    })


def _names(report) -> list[str]:
    return [c.feature for c in report.candidates]


def test_blocked_columns_never_enter_an_engineered_feature():
    analysis = build_analysis_frame(_frame(), "y", excluded_columns=["leak_mean"], confounder_columns=["conf_mean"],
                                    context_columns=["ctx_mean"])
    engineered = analysis.feature_columns
    assert any("_x_" in f for f in engineered) and any("_over_" in f for f in engineered)
    for blocked in ("leak_mean", "conf_mean", "ctx_mean", "y"):
        assert not any(blocked in f.split("_x_") or blocked in f.split("_over_") for f in engineered), (blocked, engineered)
    for f in engineered:  # every component of every feature is an admissible column
        assert not set(analysis.feature_components.get(f, [])) & {"leak_mean", "conf_mean", "ctx_mean", "y"}, f


def test_proposed_feature_on_an_excluded_column_is_not_evaluated():
    report = run_discovery(_frame(), DiscoveryRequest(
        target_column="y", excluded_columns=["leak_mean"], validation_resamples=50,
        proposed_features=[{"op": "product", "a": "signal_mean", "b": "leak_mean", "name": "sneaky"}]))
    assert "sneaky" not in _names(report)
    assert any("sneaky" in w and "excluded" in w for w in report.warnings)


def test_propose_feature_refuses_target_excluded_and_context_columns(tmp_path):
    csv = tmp_path / "d.csv"
    _frame().to_csv(csv, index=False)
    ctx = SimpleNamespace(state={"csv_path": str(csv), "context_columns": ["ctx_mean"]},
                          actions=SimpleNamespace(escalate=False))
    set_target("y", ctx, excluded_columns_csv="leak_mean")
    for b in ("leak_mean", "ctx_mean", "y"):
        assert "error" in propose_feature("ratio", "signal_mean", b, ctx), b
    assert not ctx.state.get("proposed_features")
    assert "registered" in propose_feature("ratio", "signal_mean", "other_mean", ctx)


def test_large_sample_permutation_null_supports_a_005_decision():
    rng = np.random.default_rng(0)
    n = 3500  # above the size at which the held-out model repeats CV only a few times
    x = rng.normal(0, 1, (n, 3))
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df["y"] = 0.4 * df["a"] + rng.normal(0, 1, n)
    fs = run_discovery(df, DiscoveryRequest(target_column="y", validation_resamples=50)).fact_sheet
    assert fs["ml_null_draws"] >= 39
    assert fs["ml_above_chance"] is True and fs["ml_metric_vs_null_p"] < 0.05
