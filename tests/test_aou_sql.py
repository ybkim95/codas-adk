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
    base = CohortSpec.from_json(SPEC)
    base.covariates.append(cohort.CovariateSpec(name="unknown_cov", kind="drug", concept_ids=[123456789]))
    kept, failed = feas.verify_covariates(base, SYNTHETIC_CDR, object())
    assert [c.name for c in kept.covariates] == ["bmi_cov"] and "not in the CDR" in failed["unknown_cov"]


@pytest.fixture(scope="module")
def synthetic():
    con = duckdb.connect()
    build_synthetic_cdr(con)
    return con


def test_synthetic_planted_signal_survives_extraction(synthetic):
    spec = CohortSpec.from_json(SPEC)
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
    base = CohortSpec.from_json(SPEC)
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
        result = runner.explore(CATALOG, CohortSpec.from_json(SPEC), SYNTHETIC_CDR, client=object(), top_k=2,
                                min_measurement_n=100, min_cases=30, out_dir=out_dir, screen_only=True,
                                log=lambda *_: None)
        # by default the top K is taken for each outcome kind, so conditions run alongside measurements
        assert set(result["chosen"]) == {"hba1c", "hdl", "t2d", "depression"}
        overall = runner.explore(CATALOG, CohortSpec.from_json(SPEC), SYNTHETIC_CDR, client=object(), top_k=2,
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

    df = duckdb_query(synthetic)(None, *build_cohort_sql(CohortSpec.from_json(SPEC), SYNTHETIC_CDR))
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
        result = runner.explore(CATALOG, CohortSpec.from_json(SPEC), SYNTHETIC_CDR, client=object(),
                                outcome_names=["depression", "ldl"], out_dir=out_dir, screen_only=True,
                                log=lambda *_: None)
        assert result["chosen"] == ["depression"]   # ldl fails the concept check in the synthetic vocabulary
        with pytest.raises(ValueError):
            runner.explore(CATALOG, CohortSpec.from_json(SPEC), SYNTHETIC_CDR, client=object(),
                           outcome_names=["not_in_catalog"], out_dir=out_dir, screen_only=True, log=lambda *_: None)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
