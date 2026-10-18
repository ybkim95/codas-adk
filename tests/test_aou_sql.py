"""The extractor's BigQuery SQL, executed: a hand-checked micro-CDR and the synthetic CDR in DuckDB.

Skipped when duckdb/sqlglot are not installed (they are dev-only; the Workbench runs BigQuery).
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from codas.aou import cohort, feasibility as feas  # noqa: E402
from codas.aou import runner  # noqa: E402
from codas.aou.cohort import CohortSpec, build_cohort_sql, output_columns  # noqa: E402
from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"

def _window_spec() -> CohortSpec:
    """The example spec with the outcome-anchored design these tests characterise (no split, no lag)."""
    from dataclasses import replace

    return replace(CohortSpec.from_json(SPEC), design="window", lag_days=0, horizon_days=None, valid_day=None,
                   holdout_fraction=0.0, min_wear_days=30, negative_controls=[])

CATALOG = ROOT / "scripts" / "aou" / "outcomes_catalog.json"

MICRO_SPEC = CohortSpec.from_dict({
    "name": "micro",
    "outcome": {"kind": "measurement", "name": "hba1c", "concept_ids": [3004410]},
    "window_days": 180, "min_wear_days": 30,
    "wearables": [
        {"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]},
        {"table": "sleep_daily_summary", "date_column": "sleep_date", "prefix": "sleep",
         "columns": ["minute_asleep"], "filter_column": "is_main_sleep", "filter_value": True},
    ],
})


def _days(first: date, n: int) -> list[date]:
    return [first + timedelta(days=i) for i in range(n)]


@pytest.fixture
def micro():
    """Three people whose expected rows can be worked out by hand."""
    con = duckdb.connect()
    act, sleep = [], []
    # P1: 30 days of 1,000 steps in June; 99,999 on the index day and 50,000 a year earlier (both out).
    act += [(1, d, 1000.0) for d in _days(date(2023, 6, 1), 30)]
    act += [(1, date(2023, 7, 1), 99999.0), (1, date(2022, 12, 1), 50000.0)]
    sleep += [(1, d, True, 400.0) for d in _days(date(2023, 6, 1), 30)]
    sleep += [(1, d, False, 40.0) for d in _days(date(2023, 6, 1), 30)]  # naps: filtered out
    # P2: first HbA1c (March) has no wear before it, so the August one becomes the index.
    act += [(2, d, 2000.0) for d in _days(date(2023, 7, 1), 31)]
    # P3: only 10 days of wear: below min_wear_days, excluded.
    act += [(3, d, 3000.0) for d in _days(date(2023, 6, 1), 10)]
    tables = {
        "person": pd.DataFrame({"person_id": [1, 2, 3], "year_of_birth": [1980, 1990, 1970],
                                "sex_at_birth_concept_id": [10, 11, 10]}),
        "concept": pd.DataFrame({"concept_id": [10, 11, 3004410], "concept_name": ["Female", "Male", "HbA1c"],
                                 "standard_concept": [None, None, "S"],
                                 "domain_id": ["Gender", "Gender", "Measurement"]}),
        "concept_ancestor": pd.DataFrame({"ancestor_concept_id": [3004410], "descendant_concept_id": [3004410]}),
        "measurement": pd.DataFrame({
            "measurement_id": [1, 2, 3, 4, 5], "person_id": [1, 1, 2, 2, 3],
            "measurement_concept_id": [3004410] * 5,
            "measurement_date": [date(2023, 7, 1), date(2023, 9, 1), date(2023, 3, 1), date(2023, 8, 1),
                                 date(2023, 6, 20)],
            "value_as_number": [6.0, 7.0, 9.9, 5.5, 6.6]}),
        "activity_summary": pd.DataFrame(act, columns=["person_id", "date", "steps"]),
        "sleep_daily_summary": pd.DataFrame(sleep, columns=["person_id", "sleep_date", "is_main_sleep",
                                                            "minute_asleep"]),
    }
    for name, frame in tables.items():
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    return con


def test_micro_cdr_matches_hand_computed_rows(micro):
    out = duckdb_query(micro)(None, *build_cohort_sql(MICRO_SPEC, SYNTHETIC_CDR))
    out = out.sort_values("hba1c").reset_index(drop=True)
    assert list(out.columns) == output_columns(MICRO_SPEC)
    assert len(out) == 2  # P3 excluded
    p2, p1 = out.iloc[0], out.iloc[1]
    assert p1.hba1c == 6.0 and p1.act_days == 30 and p1.act_steps_mean == 1000.0  # index day + old day excluded
    assert p1.sleep_minute_asleep_mean == 400.0 and p1.sleep_days == 30           # naps excluded
    assert p1.age == 43 and p1.female == 1
    assert p2.hba1c == 5.5 and p2.act_days == 31 and p2.act_steps_mean == 2000.0  # second measurement indexed
    assert p2.female == 0 and pd.isna(p2.sleep_days)                               # no sleep data -> NULL


COVARIATES = [
    {"name": "weight", "kind": "measurement", "concept_ids": [900], "lookback_days": 365, "value_max": 300.0},
    {"name": "htn", "kind": "condition", "concept_ids": [901], "lookback_days": None},
    {"name": "rx", "kind": "drug", "concept_ids": [903], "lookback_days": 30},
]


def test_covariates_match_hand_computed_values(micro):
    """P1 is indexed on 2023-07-01, P2 on 2023-08-01 (see the micro fixture)."""
    micro.execute("""INSERT INTO measurement VALUES
        (10, 1, 900, DATE '2023-07-01', 70.0),   -- P1 on the index day: counts, and is the latest
        (11, 1, 900, DATE '2023-06-01', 68.0),   -- P1 earlier in the lookback
        (12, 1, 900, DATE '2022-01-01', 99.0),   -- P1 outside the 365-day lookback
        (13, 1, 900, DATE '2023-07-02', 200.0),  -- P1 after the index: never used
        (14, 2, 900, DATE '2023-05-01', 999.0),  -- P2 above value_max: dropped, so P2 has no value
        (15, 2, 900, DATE '2023-08-15', 80.0)""")  # P2 after the index
    micro.execute("INSERT INTO concept_ancestor VALUES (900, 900), (901, 901), (901, 902), (903, 903)")
    micro.execute("""CREATE TABLE condition_occurrence AS SELECT * FROM (VALUES
        (1, 902, DATE '2020-01-01'),   -- P1: a descendant concept, long before (no lookback limit)
        (2, 901, DATE '2023-08-01')    -- P2: on the index date, so not prior history
        ) AS t(person_id, condition_concept_id, condition_start_date)""")
    micro.execute("""CREATE TABLE drug_exposure AS SELECT * FROM (VALUES
        (1, 903, DATE '2023-05-01'),   -- P1: 61 days before the index, outside the 30-day lookback
        (2, 903, DATE '2023-07-15')    -- P2: 17 days before the index
        ) AS t(person_id, drug_concept_id, drug_exposure_start_date)""")
    spec = CohortSpec.from_dict({**_micro_dict(), "covariates": COVARIATES})
    out = duckdb_query(micro)(None, *build_cohort_sql(spec, SYNTHETIC_CDR)).sort_values("hba1c")
    assert list(out.columns) == output_columns(spec)
    p2, p1 = out.iloc[0], out.iloc[1]
    assert p1.weight == 70.0 and p1.htn == 1 and p1.rx == 0
    assert pd.isna(p2.weight) and p2.htn == 0 and p2.rx == 1
    # the covariates leave the wearable features and the cohort untouched
    base = duckdb_query(micro)(None, *build_cohort_sql(MICRO_SPEC, SYNTHETIC_CDR)).sort_values("hba1c")
    pd.testing.assert_frame_equal(out[output_columns(MICRO_SPEC)].reset_index(drop=True), base.reset_index(drop=True))


def test_covariate_that_is_the_outcome_is_refused_or_left_out():
    same = {"name": "a1c_again", "kind": "measurement", "concept_ids": [3004410]}
    with pytest.raises(ValueError, match="outcome"):
        CohortSpec.from_dict({**_micro_dict(), "covariates": [same]})
    base = CohortSpec.from_dict({**_micro_dict(), "covariates": [COVARIATES[0]]})
    base.covariates.append(cohort.CovariateSpec(**same))  # as a base spec reused for many outcomes
    o = feas.CatalogOutcome(name="hba1c", label="HbA1c", kind="measurement", concept_ids=[3004410],
                            expect_name="a1c")
    assert [c.name for c in feas.spec_for(o, base).covariates] == ["weight"]


def _micro_dict() -> dict:
    return {
        "name": "micro",
        "outcome": {"kind": "measurement", "name": "hba1c", "concept_ids": [3004410]},
        "window_days": 180, "min_wear_days": 30,
        "wearables": [
            {"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]},
            {"table": "sleep_daily_summary", "date_column": "sleep_date", "prefix": "sleep",
             "columns": ["minute_asleep"], "filter_column": "is_main_sleep", "filter_value": True},
        ],
    }


def test_explore_skips_covariates_that_fail_the_concept_check(synthetic, monkeypatch):
    monkeypatch.setattr(cohort, "_query", duckdb_query(synthetic))
    base = _window_spec()
    base.covariates.append(cohort.CovariateSpec(name="unknown_cov", kind="drug", concept_ids=[123456789]))
    kept, failed = feas.verify_covariates(base, SYNTHETIC_CDR, object())
    assert [c.name for c in kept.covariates] == ["bmi_cov"] and "not in the CDR" in failed["unknown_cov"]


@pytest.fixture(scope="module")
def synthetic():
    con = duckdb.connect()
    build_synthetic_cdr(con)
    return con


def test_synthetic_planted_signal_survives_extraction(synthetic):
    spec = _window_spec()
    df = duckdb_query(synthetic)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    assert len(df) > 300
    assert df["hba1c"].corr(df["act_steps_mean"], method="spearman") < -0.4


def test_catalog_verification_flags_unknown_concepts(synthetic, monkeypatch):
    monkeypatch.setattr(cohort, "_query", duckdb_query(synthetic))
    outcomes = feas.load_catalog(CATALOG)
    problems = feas.verify_catalog(outcomes, SYNTHETIC_CDR, client=None)
    assert problems["hba1c"] is None and problems["t2d"] is None and problems["depression"] is None
    assert "not in the CDR" in problems["ldl"]  # the synthetic vocabulary has no LDL concept


def test_concept_check_rejects_wrong_domain_and_name():
    o = feas.CatalogOutcome(name="x", label="X", kind="condition", concept_ids=[1], expect_name="diabetes")
    assert "domain" in feas.check_concepts([o], [{"concept_id": 1, "concept_name": "Diabetes", "domain_id": "Drug"}])["x"]
    assert "expected" in feas.check_concepts([o], [{"concept_id": 1, "concept_name": "Asthma", "domain_id": "Condition"}])["x"]


def test_feasibility_counts_match_extraction(synthetic, monkeypatch):
    monkeypatch.setattr(cohort, "_query", duckdb_query(synthetic))
    base = _window_spec()
    outcomes = {o.name: o for o in feas.load_catalog(CATALOG)}
    rows = {r["name"]: r for r in feas.feasibility([outcomes["hba1c"], outcomes["t2d"]], base, SYNTHETIC_CDR, None)}
    full = duckdb_query(synthetic)(None, *build_cohort_sql(feas.spec_for(outcomes["t2d"], base), SYNTHETIC_CDR))
    assert rows["t2d"]["n_participants"] == len(full) and rows["t2d"]["n_cases"] == int(full["t2d"].sum())
    assert rows["hba1c"]["n_participants"] > 300


def test_rank_is_deterministic_and_respects_minimums():
    rows = [
        {"name": "a", "kind": "measurement", "n_participants": 5000},
        {"name": "b", "kind": "measurement", "n_participants": 900},
        {"name": "c", "kind": "condition", "n_participants": 6000, "n_cases": 300, "n_controls": 5700},
        {"name": "d", "kind": "condition", "n_participants": 6000, "n_cases": 150, "n_controls": 5850},
        {"name": "e", "kind": "measurement", "n_participants": 5000},
    ]
    assert [r["name"] for r in feas.rank_outcomes(rows, top_k=3)] == ["a", "e", "c"]


def test_feasibility_table_masks_small_counts():
    rows = [{"name": "x", "label": "X", "kind": "condition", "n_participants": 500, "n_cases": 12, "n_controls": 488}]
    table = feas.feasibility_table(rows, {"y": "concept 1 not in the CDR"})
    assert "| 12 |" not in table and "<=20" in table and "skipped" in table


def test_explore_screen_on_synthetic_cdr(synthetic, monkeypatch):
    monkeypatch.setattr(cohort, "_query", duckdb_query(synthetic))
    out_dir = ROOT / ".codas_runs" / "aou_explore_test"
    try:
        result = runner.explore(CATALOG, _window_spec(), SYNTHETIC_CDR, client=object(), top_k=2,
                                min_measurement_n=100, min_cases=30, out_dir=out_dir, screen_only=True,
                                log=lambda *_: None)
        # by default the top K is taken for each outcome kind, so conditions run alongside measurements
        assert set(result["chosen"]) == {"hba1c", "hdl", "t2d", "depression"}
        overall = runner.explore(CATALOG, _window_spec(), SYNTHETIC_CDR, client=object(), top_k=2,
                                 per_kind=False, min_measurement_n=100, min_cases=30, out_dir=out_dir,
                                 screen_only=True, log=lambda *_: None)
        # across kinds, the measurements' participant counts outrank the conditions' case counts
        assert set(overall["chosen"]) == {"hba1c", "hdl"}
        summary = Path(result["summary_path"]).read_text(encoding="utf-8")
        assert "Feasibility screen" in summary and "skipped" in summary
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.parametrize("main,nap", [("true", "false"), (True, False)])
def test_sleep_filter_matches_string_and_bool_flags(main, nap):
    """The Registered Tier v9 stores is_main_sleep as STRING; earlier releases used BOOL."""
    con = duckdb.connect()
    days = _days(date(2023, 1, 1), 10)
    tables = {
        "concept_ancestor": pd.DataFrame({"ancestor_concept_id": [1], "descendant_concept_id": [1]}),
        "measurement": pd.DataFrame({"measurement_id": [1], "person_id": [1], "measurement_concept_id": [1],
                                     "measurement_date": [date(2023, 1, 20)], "value_as_number": [5.0]}),
        "activity_summary": pd.DataFrame({"person_id": 1, "date": days, "steps": 1000.0}),
        "sleep_daily_summary": pd.DataFrame({"person_id": 1, "sleep_date": days * 2,
                                             "is_main_sleep": [main] * 10 + [nap] * 10,
                                             "minute_asleep": [400.0] * 10 + [40.0] * 10}),
    }
    for name, frame in tables.items():
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    spec = CohortSpec.from_dict({
        "name": "flags", "outcome": {"kind": "measurement", "name": "y", "concept_ids": [1]},
        "window_days": 60, "min_wear_days": 5, "include_demographics": False,
        "wearables": [
            {"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]},
            {"table": "sleep_daily_summary", "date_column": "sleep_date", "prefix": "sleep",
             "columns": ["minute_asleep"], "filter_column": "is_main_sleep", "filter_value": True},
        ],
    })
    out = duckdb_query(con)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    assert out.iloc[0].sleep_minute_asleep_mean == 400.0 and out.iloc[0].sleep_days == 10


def test_reciprocal_ratios_are_not_double_counted(synthetic):
    """a/b and b/a are rank-identical; only one of each pair may count as a validated finding."""
    from codas.core.discovery import DiscoveryRequest, run_discovery

    df = duckdb_query(synthetic)(None, *build_cohort_sql(_window_spec(), SYNTHETIC_CDR))
    report = run_discovery(df, DiscoveryRequest(target_column="hba1c", confounder_columns=["age", "female"],
                                                validation_resamples=60, max_ratio_features=12))
    passing = {c.feature for c in report.candidates if c.verdict in {"validated", "conditional"}}
    for feature in passing:
        if "_over_" in feature:
            numerator, denominator = feature.split("_over_", 1)
            assert f"{denominator}_over_{numerator}" not in passing, feature


def test_explore_runs_named_outcomes_instead_of_the_ranking(synthetic, monkeypatch):
    monkeypatch.setattr(cohort, "_query", duckdb_query(synthetic))
    out_dir = ROOT / ".codas_runs" / "aou_named_test"
    try:
        result = runner.explore(CATALOG, _window_spec(), SYNTHETIC_CDR, client=object(),
                                outcome_names=["depression", "ldl"], out_dir=out_dir, screen_only=True,
                                log=lambda *_: None)
        assert result["chosen"] == ["depression"]   # ldl fails the concept check in the synthetic vocabulary
        with pytest.raises(ValueError):
            runner.explore(CATALOG, _window_spec(), SYNTHETIC_CDR, client=object(),
                           outcome_names=["not_in_catalog"], out_dir=out_dir, screen_only=True, log=lambda *_: None)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


INCIDENT = {
    "name": "inc",
    "design": "incident", "window_days": 30, "lag_days": 10, "min_wear_days": 5,
    "valid_day": {"column": "steps", "min": 100},
    "outcome": {"kind": "condition", "name": "dx", "concept_ids": [500]},
    "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]}],
}


@pytest.fixture
def incident_cdr():
    """Six people; every expected value below is worked out by hand."""
    con = duckdb.connect()
    act = []
    # P1: Jan 1 has 50 steps (invalid), then 1,000 a day to Feb 9. First valid day Jan 2 -> index Feb 1,
    #     outcome counted from Feb 11; baseline [Jan 2, Feb 1) holds 30 valid days.
    act += [(1, date(2023, 1, 1), 50.0)] + [(1, d, 1000.0) for d in _days(date(2023, 1, 2), 39)]
    act += [(2, d, 2000.0) for d in _days(date(2023, 3, 1), 46)]     # P2: index Mar 31, from Apr 10
    act += [(3, d, 3000.0) for d in _days(date(2023, 1, 1), 40)]     # P3: diagnosed before (prevalent)
    act += [(4, d, 4000.0) for d in _days(date(2023, 1, 1), 40)]     # P4: diagnosed inside the lag
    act += [(5, d, 5000.0) for d in _days(date(2023, 1, 1), 40)]     # P5: no EHR visit after the lag
    act += [(6, d, 6000.0) for d in _days(date(2023, 1, 1), 3)]      # P6: 3 baseline days < min_wear_days
    tables = {
        "person": pd.DataFrame({"person_id": [1, 2, 3, 4, 5, 6], "year_of_birth": [1980] * 6,
                                "sex_at_birth_concept_id": [10] * 6}),
        "concept": pd.DataFrame({"concept_id": [10, 500], "concept_name": ["Female", "Some condition"],
                                 "standard_concept": [None, "S"], "domain_id": ["Gender", "Condition"]}),
        "concept_ancestor": pd.DataFrame({"ancestor_concept_id": [500], "descendant_concept_id": [500]}),
        "condition_occurrence": pd.DataFrame({
            "person_id": [1, 3, 4], "condition_concept_id": [500, 500, 500],
            "condition_start_date": [date(2023, 6, 1), date(2022, 5, 1), date(2023, 2, 5)]}),
        "visit_occurrence": pd.DataFrame({
            "person_id": [1, 1, 2, 3, 4, 5, 6],
            "visit_start_date": [date(2023, 6, 1), date(2023, 12, 1), date(2024, 1, 10), date(2024, 1, 1),
                                 date(2024, 1, 1), date(2023, 1, 15), date(2024, 1, 1)]}),
        "measurement": pd.DataFrame({
            "measurement_id": [1, 2, 3, 4], "person_id": [1, 1, 1, 2], "measurement_concept_id": [600] * 4,
            "measurement_date": [date(2023, 2, 5), date(2023, 3, 1), date(2023, 9, 1), date(2023, 5, 1)],
            "value_as_number": [9.0, 7.0, 8.0, 6.0]}),
        "activity_summary": pd.DataFrame(act, columns=["person_id", "date", "steps"]),
    }
    for name, frame in tables.items():
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    return con


def test_incident_condition_cohort_matches_hand_computed_rows(incident_cdr):
    spec = CohortSpec.from_dict(INCIDENT)
    out = duckdb_query(incident_cdr)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    assert list(out.columns) == output_columns(spec) and "followup_days" in out.columns
    out = out.sort_values("act_steps_mean").reset_index(drop=True)
    assert out["act_steps_mean"].tolist() == [1000.0, 2000.0]   # P3-P6 excluded, each for its own reason
    p1, p2 = out.iloc[0], out.iloc[1]
    assert p1.dx == 1 and p1.followup_days == 110 and p1.act_days == 30   # Feb 11 -> Jun 1; invalid Jan 1 dropped
    assert p2.dx == 0 and p2.followup_days == 275 and p2.act_days == 30   # Apr 10 -> last visit 2024-01-10


def test_incident_measurement_takes_the_first_value_after_the_lag(incident_cdr):
    spec = CohortSpec.from_dict({**INCIDENT, "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]}})
    out = duckdb_query(incident_cdr)(None, *build_cohort_sql(spec, SYNTHETIC_CDR)).sort_values("act_steps_mean")
    # P1: Feb 5 is inside baseline+lag (before Feb 11), so Mar 1's 7.0 is the outcome; P2: May 1's 6.0.
    assert out["lab"].tolist() == [7.0, 6.0] and "followup_days" not in out.columns
    capped = CohortSpec.from_dict({**INCIDENT, "horizon_days": 19,
                                   "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]}})
    out = duckdb_query(incident_cdr)(None, *build_cohort_sql(capped, SYNTHETIC_CDR))
    # 19 days after the lag: P1's window is [Feb 11, Mar 2), which holds Mar 1; P2's is [Apr 10, Apr 29),
    # which misses May 1, so P2 drops out.
    assert out["lab"].tolist() == [7.0]


def test_holdout_rows_are_sealed_in_a_separate_table(incident_cdr, monkeypatch, tmp_path):
    spec = CohortSpec.from_dict({**INCIDENT, "holdout_fraction": 0.5, "split_salt": 7})
    monkeypatch.setattr(cohort, "_query", duckdb_query(incident_cdr))
    out_dir = ROOT / ".codas_runs" / "holdout_test"
    try:
        path = cohort.extract_cohort(spec, SYNTHETIC_CDR, out_dir, client=object(), check_schema=False)
        discovery, sealed = pd.read_csv(path), pd.read_csv(cohort.holdout_path(path))
        assert list(discovery.columns) == list(sealed.columns) == output_columns(spec)  # no split column, no ids
        expected_sealed = {pid for pid in (1, 2) if cohort.holdout_bucket(pid, 7) < 500}
        sealed_steps = {1000.0 * pid for pid in expected_sealed}
        assert set(sealed["act_steps_mean"]) == sealed_steps
        assert set(discovery["act_steps_mean"]) == {1000.0, 2000.0} - sealed_steps
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_incident_design_allows_a_baseline_value_of_the_outcome_as_covariate():
    same = {"name": "lab_baseline", "kind": "measurement", "concept_ids": [600]}
    spec = CohortSpec.from_dict({**INCIDENT, "covariates": [same],
                                 "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]}})
    assert [c.name for c in spec.covariates] == ["lab_baseline"]
    with pytest.raises(ValueError, match="outcome"):
        CohortSpec.from_dict({**INCIDENT, "design": "window", "covariates": [same],
                              "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]}})


def test_incident_holdout_sql_selects_no_identifier():
    spec = CohortSpec.from_dict({**INCIDENT, "holdout_fraction": 0.5})
    sql, _ = build_cohort_sql(spec, SYNTHETIC_CDR)
    selected = sql[sql.rindex("\nSELECT"):sql.index("\nFROM index_events ie")]
    assert "person_id" not in selected and "_holdout" in selected


def test_negative_control_outcomes_reach_only_the_sealed_table(incident_cdr, monkeypatch):
    incident_cdr.execute("INSERT INTO condition_occurrence VALUES (1, 700, DATE '2023-03-01'), "
                         "(2, 700, DATE '2023-04-01')")
    incident_cdr.execute("INSERT INTO concept_ancestor VALUES (700, 700), (701, 701)")
    controls = [{"name": "ctl_a", "concept_ids": [700]}, {"name": "ctl_b", "concept_ids": [701]}]
    for outcome in (INCIDENT["outcome"], {"kind": "measurement", "name": "lab", "concept_ids": [600]}):
        spec = CohortSpec.from_dict({**INCIDENT, "outcome": outcome, "negative_controls": controls})
        out = duckdb_query(incident_cdr)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
        assert list(out.columns) == cohort.holdout_columns(spec)
        p1, p2 = (out.sort_values("act_steps_mean").iloc[i] for i in (0, 1))
        # Follow-up starts at index + lag: P1 on Feb 11, P2 on Apr 10.
        assert p1.nc_ctl_a == 1 and p1.nc_ctl_a_days == (date(2023, 3, 1) - date(2023, 2, 11)).days
        assert pd.isna(p2.nc_ctl_a) and pd.isna(p2.nc_ctl_a_days)  # diagnosed before follow-up: left empty
        assert p1.nc_ctl_b == 0 and p1.nc_ctl_b_days == (date(2023, 12, 1) - date(2023, 2, 11)).days
        assert p2.nc_ctl_b == 0 and p2.nc_ctl_b_days == (date(2024, 1, 10) - date(2023, 4, 10)).days
    spec = CohortSpec.from_dict({**INCIDENT, "negative_controls": controls, "holdout_fraction": 0.5, "split_salt": 7})
    monkeypatch.setattr(cohort, "_query", duckdb_query(incident_cdr))
    out_dir = ROOT / ".codas_runs" / "nc_test"
    try:
        path = cohort.extract_cohort(spec, SYNTHETIC_CDR, out_dir, client=object(), check_schema=False)
        assert list(pd.read_csv(path).columns) == output_columns(spec)  # the agents never see the controls
        assert list(pd.read_csv(cohort.holdout_path(path)).columns) == cohort.holdout_columns(spec)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    present = [("activity_summary", c) for c in ("person_id", "date", "steps")]
    assert "condition_occurrence.condition_concept_id" in cohort.missing_columns(spec, present)


def test_negative_controls_are_validated_and_dropped_for_their_own_outcome():
    with pytest.raises(ValueError, match="incident"):
        CohortSpec.from_dict({**INCIDENT, "design": "window", "negative_controls": [{"name": "c", "concept_ids": [9]}]})
    with pytest.raises(ValueError, match="outcome"):
        CohortSpec.from_dict({**INCIDENT, "negative_controls": [{"name": "c", "concept_ids": [500]}]})
    with pytest.raises(ValueError, match="positive control"):
        CohortSpec.from_dict({**INCIDENT, "positive_controls": ["no_such_column"]})
    base = CohortSpec.from_dict({**INCIDENT, "negative_controls": [{"name": "c", "concept_ids": [9]}],
                                 "positive_controls": ["age"]})
    same = feas.CatalogOutcome(name="nine", label="Nine", kind="condition", concept_ids=[9], expect_name="")
    assert feas.spec_for(same, base).negative_controls == []
    other = feas.CatalogOutcome(name="ten", label="Ten", kind="condition", concept_ids=[10], expect_name="")
    spec = feas.spec_for(other, base)
    assert [c.name for c in spec.negative_controls] == ["c"] and spec.positive_controls == ["age"]
    slim, _ = feas.feasibility_sql(spec, SYNTHETIC_CDR)
    assert "n0_concept_ids" not in slim  # the feasibility count does not compute the controls


AOU_SIZED_ID = 200_001_000_000  # All of Us person ids have 12 digits


def test_holdout_split_handles_all_of_us_sized_person_ids(incident_cdr, monkeypatch):
    """A 12-digit person id times a 32-bit hash constant overflows INT64 (BigQuery and DuckDB both raise)."""
    for table in ("person", "condition_occurrence", "visit_occurrence", "measurement", "activity_summary"):
        incident_cdr.execute(f"UPDATE {table} SET person_id = person_id + {AOU_SIZED_ID}")
    spec = CohortSpec.from_dict({**INCIDENT, "holdout_fraction": 0.5, "split_salt": 3})
    out = duckdb_query(incident_cdr)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    sealed = {int(round(steps / 1000)) for steps, flag in zip(out["act_steps_mean"], out["_holdout"]) if flag}
    assert sealed == {pid for pid in (1, 2) if cohort.holdout_bucket(pid + AOU_SIZED_ID, 3) < 500}


def test_holdout_bucket_is_uniform_and_the_salt_reshuffles_it():
    ids = range(AOU_SIZED_ID, AOU_SIZED_ID + 100_000)
    first = {i for i in ids if cohort.holdout_bucket(i, 0) < 500}
    assert abs(len(first) / 100_000 - 0.5) < 0.01
    again = {i for i in ids if cohort.holdout_bucket(i, 1) < 500}
    assert abs(len(first & again) / len(first) - 0.5) < 0.02  # a new salt is a new split, not a shifted one
    assert all(0 <= cohort.holdout_bucket(i, 0) < 1000 for i in (1, 2**62, AOU_SIZED_ID))
