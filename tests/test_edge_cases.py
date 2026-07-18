"""Rigorous robustness / edge-case tests.

Contract: the engine must NEVER crash on a malformed or degenerate input. It either returns a valid
DiscoveryReport, or raises InsufficientDataError with a clear message. Adversarial inputs (leakage,
duplicate names, non-numeric targets, inf, p>n, unicode) are covered explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from codas.core.data import InsufficientDataError
from codas.core.discovery import DiscoveryRequest, run_discovery

RNG = np.random.default_rng(7)


def _df(n: int = 200) -> pd.DataFrame:
    x1 = RNG.normal(0, 1, n)
    x2 = RNG.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "noise": RNG.normal(0, 1, n),
                         "y": 0.6 * x1 - 0.4 * x2 + RNG.normal(0, 1, n)})


def _run(df: pd.DataFrame, **kw) -> dict:
    kw.setdefault("validation_resamples", 120)
    kw.setdefault("top_k", 6)
    return run_discovery(df, DiscoveryRequest(**kw)).to_dict()


# --- inputs that MUST raise a clean boundary error (never a crash) ---

@pytest.mark.parametrize("make", [
    lambda: pd.DataFrame({"y": [], "x": []}),                     # empty
    lambda: _df(1),                                              # single row
    lambda: _df().assign(y=np.nan),                              # target all-NaN
    lambda: _df().assign(y=3.0),                                 # target constant
    lambda: _df()[["y"]].copy(),                                 # no candidate features
    lambda: pd.DataFrame({"y": _df()["y"], "a": 1.0, "b": 2.0}),  # all features constant
    lambda: pd.DataFrame({"y": _df()["y"], "a": np.nan}),        # all features NaN
])
def test_degenerate_inputs_raise_boundary(make):
    with pytest.raises(InsufficientDataError):
        _run(make(), target_column="y")


def test_missing_target_raises_clear_error():
    with pytest.raises(InsufficientDataError) as exc:
        _run(_df(), target_column="does_not_exist")
    assert "not present" in str(exc.value).lower()


def test_multiclass_text_target_raises():
    df = _df()
    df["y"] = (["a", "b", "c"] * len(df))[: len(df)]
    with pytest.raises(InsufficientDataError):
        _run(df, target_column="y")


# --- inputs that MUST run without crashing ---

def test_duplicate_column_names_do_not_crash():
    df = _df()
    df.columns = ["x1", "x1", "noise", "y"]  # duplicate name would make df["x1"] a DataFrame
    rep = _run(df, target_column="y")
    assert rep["fact_sheet"]["target_column"] == "y"


def test_binary_string_target_is_encoded_and_runs():
    df = _df()
    df["y"] = np.where(df["x1"] > 0, "high", "low")
    rep = _run(df, target_column="y")
    assert rep["fact_sheet"]["rows"] > 0  # ran (no crash, no silent empty)


def test_numeric_strings_are_coerced():
    df = _df()
    df["y"] = df["y"].round().astype(int).astype(str)
    rep = _run(df, target_column="y")
    assert rep["fact_sheet"]["target_column"] == "y"


def test_inf_values_handled():
    df = _df()
    df.loc[df.index[:5], "x1"] = np.inf
    _run(df, target_column="y")  # must not raise


def test_wide_p_greater_than_n():
    cols = {f"f{i}": RNG.normal(0, 1, 60) for i in range(200)}
    df = pd.DataFrame({**cols, "y": cols["f0"] * 0.5 + RNG.normal(0, 1, 60)})
    _run(df, target_column="y")  # 200 features, 60 rows: must not crash


def test_unicode_column_names():
    df = _df().rename(columns={"x1": "féatüre ⚡", "x2": "列二"})
    _run(df, target_column="y")  # must not raise


def test_partial_missing_target_rows_handled():
    df = _df()
    df.loc[df.index[:50], "y"] = np.nan  # 25% of the target missing
    rep = _run(df, target_column="y")
    # Ran without crashing, and surfaced the missing-target caveat rather than silently analyzing
    # only the observed subgroup.
    assert any("missing" in w.lower() for w in rep["warnings"])


# --- adversarial: leakage must NOT be reported as a clean validated predictor ---

def test_perfect_leakage_feature_not_validated():
    df = _df()
    df["leak"] = df["y"]  # feature identical to target
    rep = _run(df, target_column="y")
    leak = [c for c in rep["candidates"] if c["feature"] == "leak"]
    assert leak and leak[0]["verdict"] != "validated"


def test_near_perfect_proxy_flagged():
    df = _df()
    df["proxy"] = df["y"] + RNG.normal(0, 0.05, len(df))  # |rho| ~ 0.999
    rep = _run(df, target_column="y")
    proxy = [c for c in rep["candidates"] if c["feature"] == "proxy"]
    assert proxy and proxy[0]["verdict"] != "validated"


# --- determinism on an adversarial frame (inf + duplicate-content columns) ---

def test_deterministic_on_messy_frame():
    df = _df()
    df.loc[df.index[:10], "x1"] = np.inf
    df["dupe"] = df["x2"]
    req = DiscoveryRequest(target_column="y", validation_resamples=120, top_k=6, random_state=5)
    a = run_discovery(df.copy(), req).to_dict()["candidates"]
    b = run_discovery(df.copy(), req).to_dict()["candidates"]
    assert [c["feature"] for c in a] == [c["feature"] for c in b]
