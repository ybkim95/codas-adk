"""define_phenotype: a phenotype defined on the day-by-day baseline becomes a column of the cohort table, stays
outcome-blind until registration closes, is pre-registered by its definition and computed the same way on
the sealed share."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.agents import tools
from codas.aou.confirm import confirm, preregister
from codas.core import phenotype_dsl as dsl

TREND = {"name": "sig_trend", "signal": "sig", "summary": "slope_per_30d", "min_days": 10}


def _share(rng, n=300):
    """Each person's signal drifts at their own rate; the outcome follows the drift."""
    drift = rng.normal(size=n)
    days = pd.DataFrame([{"row": r, "day": d, "weekday": (d % 7) + 1, "sig": 50 + drift[r] * (d + 30) / 30
                          + rng.normal(scale=0.5)} for r in range(n) for d in range(-30, 0)])
    table = pd.DataFrame({"y": 2 * drift + rng.normal(size=n), "age": rng.normal(size=n)})
    return table, days


def _ctx(tmp_path, **state):
    table, days = _share(np.random.default_rng(0))
    table.to_csv(tmp_path / "c.csv", index=False)
    days.to_csv(tmp_path / "c.days.csv.gz", index=False, compression="gzip")
    return SimpleNamespace(state={"csv_path": str(tmp_path / "c.csv"), "target_column": "y",
                                  "days_path": str(tmp_path / "c.days.csv.gz"), "eda": {"features": {}}, **state})


def test_a_defined_phenotype_becomes_a_column_that_stays_outcome_blind_until_registration(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    out = tools.define_phenotype(json.dumps(TREND), "drift of the signal over the window", ctx)
    column = dsl.column_name(TREND)
    assert out["column"] == column and "withheld" in out["note"] and "exploratory_association" not in out
    table = pd.read_csv(tmp_path / "c.csv")
    days = pd.read_csv(tmp_path / "c.days.csv.gz")
    assert np.allclose(table[column], dsl.compute(days, TREND).to_numpy())
    (record,) = ctx.state["defined_phenotypes"]
    assert record["definition"] == {"name": "sig_trend", **dsl.canonical(TREND)} and record["column"] == column
    assert column in ctx.state["eda"]["features"] and out["distribution"]["n"] == 300
    assert tools.define_phenotype(json.dumps(TREND), "", ctx)["note"] == "already defined"
    bad = tools.define_phenotype(json.dumps({"name": "nope_x", "signal": "nope", "summary": "mean"}), "", ctx)
    assert bad["problems"] and bad["day_signals"] == ["sig"]
    # Once registration has closed, the association with the outcome is shown, as for any candidate.
    ctx.state["rounds"] = [{"round": 0}]
    later = tools.define_phenotype(json.dumps({**TREND, "name": "sig_change", "summary": "mean_abs_change"}), "", ctx)
    assert "spearman_rho" in later["exploratory_association"]
    monkeypatch.setattr(tools, "_PHENOTYPE_BUDGET", 2)
    over = tools.define_phenotype(json.dumps({**TREND, "name": "sig_mean", "summary": "mean"}), "", ctx)
    assert "At most 2 phenotypes" in over["error"]


def test_without_a_day_table_the_tool_says_so(tmp_path):
    ctx = _ctx(tmp_path, days_path="")
    assert "no day-by-day baseline" in tools.define_phenotype(json.dumps(TREND), "", ctx)["error"]


def test_the_definition_is_pre_registered_and_computed_on_the_sealed_share(tmp_path):
    ctx = _ctx(tmp_path)
    tools.define_phenotype(json.dumps(TREND), "drift", ctx)
    column = dsl.column_name(TREND)
    state = {**ctx.state, "confounder_columns": ["age"], "excluded_columns": [],
             "latest_report": {"candidates": [{"feature": column, "rho": 0.5, "q_value": 1e-6, "verdict": "validated",
                                               "components": [column]}]}}
    prereg = preregister(state, tmp_path, "c", discovery_csv=tmp_path / "c.csv")
    assert prereg["defined_phenotypes"] == [{"name": "sig_trend", **dsl.canonical(TREND)}]
    assert prereg["hypotheses"][0]["definition"] == {"phenotype": {"name": "sig_trend", **dsl.canonical(TREND)}}
    # The sealed share: its own people and days; the column is computed by the registered definition.
    table, days = _share(np.random.default_rng(1))
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    table.to_csv(sealed / "c.holdout.csv", index=False)
    dsl.add_columns(prereg["defined_phenotypes"], days, sealed / "c.holdout.csv")
    result = confirm(prereg, sealed / "c.holdout.csv", tmp_path)
    (row,) = result["hypotheses"]
    assert row["feature"] == column and row["replicated"] and row["rho"] > 0.5
