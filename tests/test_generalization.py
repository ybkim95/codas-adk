"""Generalization tests.

The engine must work on ANY csv with arbitrary column names and make no assumption about the
problem domain or target. These datasets are deliberately non-clinical (housing, abstract) and use
column names the engine has never been told anything about.
"""

import numpy as np
import pandas as pd

from codas.core.discovery import DiscoveryRequest, run_discovery


def _housing_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    area = rng.uniform(50, 250, n)
    rooms = rng.integers(1, 6, n).astype(float)
    age = rng.uniform(0, 80, n)
    price = 3.0 * area + 20.0 * rooms - 0.5 * age + rng.normal(0, 15, n)
    return pd.DataFrame({
        "area_sqm": area,
        "rooms": rooms,
        "building_age": age,
        "random_noise": rng.normal(0, 1, n),
        "price_k": price,
    })


def _abstract_df() -> pd.DataFrame:
    # Meaningless names: target "y", features x1..x6, real signal only in x1 and x2.
    rng = np.random.default_rng(1)
    n = 350
    cols = {f"x{i}": rng.normal(0, 1, n) for i in range(1, 7)}
    y = 2.0 * cols["x1"] - 1.5 * cols["x2"] + rng.normal(0, 1, n)
    return pd.DataFrame({**cols, "y": y})


def test_runs_on_housing_with_arbitrary_target():
    report = run_discovery(
        _housing_df(),
        DiscoveryRequest(target_column="price_k", top_k=6, validation_resamples=120),
    ).to_dict()
    assert report["fact_sheet"]["rows"] == 400
    features = {c["feature"] for c in report["candidates"]}
    # area is the dominant driver of price; an area-derived feature must surface.
    assert any("area" in f for f in features), features


def test_runs_on_abstract_column_names():
    report = run_discovery(
        _abstract_df(),
        DiscoveryRequest(target_column="y", top_k=6, validation_resamples=120),
    ).to_dict()
    assert report["candidates"], "expected at least one candidate"
    features = {c["feature"] for c in report["candidates"]}
    assert any(("x1" in f) or ("x2" in f) for f in features), features


def test_handles_constant_and_missing_columns_without_crashing():
    df = _abstract_df()
    df["all_missing"] = np.nan
    df["constant"] = 7.0
    df.loc[df.index[:60], "x3"] = np.nan
    report = run_discovery(
        df,
        DiscoveryRequest(target_column="y", top_k=6, validation_resamples=100),
    ).to_dict()
    assert report["fact_sheet"]["target_column"] == "y"


def test_deterministic_on_arbitrary_data():
    df = _abstract_df()
    req = DiscoveryRequest(target_column="y", top_k=6, validation_resamples=120, random_state=3)
    a = run_discovery(df, req).to_dict()["candidates"]
    b = run_discovery(df, req).to_dict()["candidates"]
    assert [c["feature"] for c in a] == [c["feature"] for c in b]
    assert [round(c["rho"], 10) for c in a] == [round(c["rho"], 10) for c in b]
