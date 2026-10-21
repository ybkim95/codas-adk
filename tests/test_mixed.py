"""Within-person mixed models (codas.core.mixed): a random-intercept model with the exposure split into
its within-person and between-person parts separates a real within-person effect from a stable trait
that only differs between people."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from codas.core.mixed import mixed_model  # noqa: E402


def _panel(people=400, visits=5, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(people), visits)
    trait = np.repeat(rng.normal(size=people), visits)        # stable between-person level
    state = rng.normal(size=people * visits)                   # visit-to-visit variation
    intercept = np.repeat(rng.normal(scale=2, size=people), visits)
    frame = pd.DataFrame({"person_index": pid, "x": trait + state,                       # real within effect
                          "proxy": trait + 0.3 * rng.normal(size=people * visits),     # trait only
                          "age": np.repeat(rng.normal(50, 10, people), visits)})
    frame["y"] = intercept + 2.0 * trait + 0.5 * state + rng.normal(size=len(frame))
    return frame


def test_within_and_between_effects_are_separated():
    f = _panel()
    fit = mixed_model(f, "x", "y", covariates=["age"])
    sd_within = (f.x - f.groupby("person_index").x.transform("mean")).std(ddof=0)
    truth = 0.5 * sd_within  # effects are per SD of the within-person variation
    assert fit["within"]["ci95"][0] < truth < fit["within"]["ci95"][1] and fit["within"]["p"] < 1e-10
    assert fit["between"]["estimate"] > 1.5
    assert fit["persons"] == 400 and fit["rows"] == 2000 and fit["persons_with_repeats"] == 400
    assert 0 < fit["icc"] < 1
    proxy = mixed_model(f, "proxy", "y", covariates=["age"])
    assert proxy["within"]["ci95"][0] < 0 < proxy["within"]["ci95"][1] or abs(proxy["within"]["estimate"]) < 0.3
    assert proxy["between"]["p"] < 1e-10


def test_too_few_repeats_is_reported_not_fitted():
    f = _panel(people=50, visits=1)
    assert "error" in mixed_model(f, "x", "y")


duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")


def _micro():
    from datetime import date, timedelta

    con = duckdb.connect()
    days = [date(2023, 1, 1) + timedelta(days=i) for i in range(120)]
    act = [(1, d, 1000.0 + 10 * i) for i, d in enumerate(days)] + [(1, date(2022, 12, 31), 50.0)]  # one invalid day
    act += [(2, d, 3000.0) for d in days[:20]]  # P2: too few days before its measurement
    tables = {
        "person": pd.DataFrame({"person_id": [1, 2], "year_of_birth": [1980, 1990], "sex_at_birth_concept_id": [10, 10]}),
        "concept": pd.DataFrame({"concept_id": [10, 600], "concept_name": ["Female", "Lab"], "standard_concept": [None, "S"],
                                 "domain_id": ["Gender", "Measurement"]}),
        "concept_ancestor": pd.DataFrame({"ancestor_concept_id": [600], "descendant_concept_id": [600]}),
        "measurement": pd.DataFrame({
            "measurement_id": [1, 2, 3, 4, 5], "person_id": [1, 1, 1, 1, 2], "measurement_concept_id": [600] * 5,
            "measurement_date": [date(2023, 1, 20), date(2023, 3, 15), date(2023, 4, 20), date(2023, 4, 20),
                                 date(2023, 3, 1)],
            "value_as_number": [9.0, 6.0, 5.0, 7.0, 4.0]}),
        "activity_summary": pd.DataFrame(act, columns=["person_id", "date", "steps"]),
    }
    for name, frame in tables.items():
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    return con


def test_panel_sql_pairs_each_measurement_with_the_period_before_it():
    from codas.aou.cohort import CohortSpec, build_panel_sql, panel_columns
    from codas.aou.synthetic import SYNTHETIC_CDR, duckdb_query

    spec = CohortSpec.from_dict({
        "name": "pn", "design": "incident", "window_days": 30, "min_wear_days": 5,
        "valid_day": {"column": "steps", "min": 100}, "panel": {"period_days": 30, "min_days": 10},
        "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]},
        "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]}]})
    out = duckdb_query(_micro())(None, *build_panel_sql(spec, SYNTHETIC_CDR))
    assert list(out.columns) == panel_columns(spec)
    out = out.sort_values("days_from_start").reset_index(drop=True)
    # P1's Jan 20 value falls inside the first period after the first valid wear day, so it is left out.
    # P2 wore the device Jan 1-20 only, and nothing in the 30 days before its Mar 1 value: dropped.
    assert len(out) == 2 and out.person_index.nunique() == 1
    first, second = out.iloc[0], out.iloc[1]
    # Mar 15: period [Feb 13, Mar 15) = days 43..72 of 2023 -> steps 1000 + 10*i for i in 43..72
    assert first.lab == 6.0 and first.days_from_start == 73 and first.act_days == 30
    assert first.act_steps_mean == pytest.approx(1000 + 10 * np.mean(range(43, 73)))
    assert second.lab == 6.0  # two values on Apr 20 averaged
    assert second.act_steps_mean == pytest.approx(1000 + 10 * np.mean(range(79, 109)))
    assert first.age == 43


def test_panel_on_the_synthetic_cdr_recovers_the_planted_within_person_effect(monkeypatch):
    from codas.aou import cohort
    from codas.aou.synthetic import REPEATED_LAB, SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    spec = cohort.CohortSpec.from_dict({
        "name": "rep", "design": "incident", "window_days": 60, "min_wear_days": 20, "lag_days": 0,
        "holdout_fraction": 0.5, "panel": {"period_days": 90, "min_days": 20},
        "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [REPEATED_LAB]},
        "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]}]})
    out_dir = cohort.ROOT / ".codas_runs" / "panel_test"
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    try:
        csv = cohort.extract_cohort(spec, SYNTHETIC_CDR, out_dir, client=object(), check_schema=False)
        panel = pd.read_csv(cohort.panel_path(csv))
        sealed = pd.read_csv(cohort.holdout_path(cohort.panel_path(csv)))
        assert list(panel.columns) == cohort.panel_columns(spec) and len(panel) > 300 and len(sealed) > 300
        fit = mixed_model(panel, "act_steps_mean", "lab", covariates=["age"])
        assert fit["within"]["estimate"] > 0 and fit["within"]["p"] < 1e-4
        assert fit["between"]["estimate"] > 0
    finally:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
