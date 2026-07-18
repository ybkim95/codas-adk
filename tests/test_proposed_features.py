"""Tests for the generative-interpreter feature-proposal loop (paper Section 2.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.core.data import build_analysis_frame
from codas.core.discovery import DiscoveryRequest, _materialize_proposed_features, run_discovery


def test_materialize_supports_the_four_safe_ops():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [2.0, 4.0, 5.0]})
    w: list[str] = []
    out = _materialize_proposed_features(df, [
        {"op": "ratio", "a": "a", "b": "b", "name": "r"},
        {"op": "product", "a": "a", "b": "b", "name": "p"},
        {"op": "difference", "a": "a", "b": "b", "name": "d"},
        {"op": "sum", "a": "a", "b": "b", "name": "s"},
    ], w)
    assert list(out["r"]) == [5.0, 5.0, 6.0]
    assert list(out["p"]) == [20.0, 80.0, 150.0]
    assert list(out["d"]) == [8.0, 16.0, 25.0]
    assert list(out["s"]) == [12.0, 24.0, 35.0]


def test_materialize_skips_invalid_proposals_with_warning():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    w: list[str] = []
    out = _materialize_proposed_features(df, [
        {"op": "power", "a": "a", "b": "b", "name": "bad_op"},
        {"op": "ratio", "a": "a", "b": "missing", "name": "bad_col"},
    ], w)
    assert "bad_op" not in out.columns and "bad_col" not in out.columns
    assert any("unsupported operation" in m for m in w)
    assert any("not found" in m for m in w)


def test_proposed_ratio_is_evaluated_by_the_full_battery():
    # y is driven by the RATIO a/b (with enough noise to stay below the construct-validity gate),
    # which neither a nor b captures alone. The proposal must be screened AND run through the battery.
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(1, 10, n)
    b = rng.uniform(1, 10, n)
    y = (a / b) + rng.normal(0, 1.5, n)
    df = pd.DataFrame({"a": a, "b": b, "y": y})
    rep = run_discovery(df, DiscoveryRequest(
        target_column="y",
        proposed_features=[{"op": "ratio", "a": "a", "b": "b", "name": "a_over_b_proposed"}],
        validation_resamples=200,
    ))
    cand = next((c for c in rep.candidates if c.feature == "a_over_b_proposed"), None)
    assert cand is not None                     # the proposal entered the pipeline
    assert len(cand.tests) >= 11                # it was evaluated by the full validation battery
    assert abs(cand.rho) > 0.3                  # the ratio signal was measured
    assert cand.verdict in {"validated", "conditional", "collinear_redundant", "rejected"}


def test_strong_proposed_leak_is_rejected_by_the_battery():
    # A proposal that near-perfectly determines the target is leakage-like: the construct-validity hard
    # gate must reject it, proving proposals are subject to the same gates as any feature.
    rng = np.random.default_rng(2)
    n = 400
    a = rng.uniform(1, 10, n)
    b = rng.uniform(1, 10, n)
    y = (a / b) + rng.normal(0, 0.05, n)  # rho ~ 0.99
    df = pd.DataFrame({"a": a, "b": b, "y": y})
    rep = run_discovery(df, DiscoveryRequest(
        target_column="y",
        proposed_features=[{"op": "ratio", "a": "a", "b": "b", "name": "leak_ratio"}],
        validation_resamples=150,
    ))
    cand = next((c for c in rep.candidates if c.feature == "leak_ratio"), None)
    assert cand is not None and cand.verdict == "rejected"


def test_identifier_guard_keeps_float_index_but_drops_integer_id():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "fitness_index": rng.uniform(1, 5, n),            # near-unique FLOAT named *_index -> keep
        "record_id": np.arange(n).astype(float),          # consecutive integer id -> drop
        "y": rng.normal(size=n),
    })
    an = build_analysis_frame(df, target_column="y")
    assert "fitness_index" in an.feature_columns
    assert "record_id" not in an.feature_columns


class _Ctx:
    def __init__(self):
        self.state: dict = {}


def test_propose_feature_tool_registers_and_validates_op():
    from codas.agents.tools import propose_feature
    ctx = _Ctx()
    bad = propose_feature("power", "a", "b", ctx)
    assert "error" in bad and ctx.state.get("proposed_features") in (None, [])

    ok = propose_feature("ratio", "steps", "rhr", ctx, name="fit")
    assert ok["registered"] == "fit"
    assert ctx.state["proposed_features"] == [{"op": "ratio", "a": "steps", "b": "rhr", "name": "fit"}]
    # idempotent: re-proposing the same name does not duplicate
    propose_feature("ratio", "steps", "rhr", ctx, name="fit")
    assert len(ctx.state["proposed_features"]) == 1
