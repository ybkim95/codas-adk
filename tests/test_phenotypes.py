"""Intraday digital phenotypes: the BigQuery SQL executed in DuckDB on a hand-built micro-CDR and compared
with an independent numpy implementation of each published definition; column resolution; and, on the
synthetic CDR, that each phenotype tracks the latent that generated it."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from codas.aou import cohort  # noqa: E402
from codas.aou.cohort import CohortSpec, build_cohort_sql, output_columns  # noqa: E402
from codas.aou.phenotypes import PhenotypeSource, resolve  # noqa: E402
from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query, intraday_latents  # noqa: E402

SPEC = {
    "name": "ph", "design": "incident", "window_days": 7, "lag_days": 0, "min_wear_days": 3,
    "valid_day": {"column": "steps", "min": 100},
    "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]},
    "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]}],
    "phenotypes": [
        {"kind": "minute_activity", "table": "steps_intraday", "prefix": "mi",
         "columns": {"time": ["datetime"], "value": ["steps"]}},
        {"kind": "sleep_stages", "table": "sleep_level", "prefix": "ss",
         "columns": {"date": "sleep_date", "start": "start_datetime", "level": "level",
                     "duration": "duration_in_min", "main": "is_main_sleep"}},
    ],
}

# P1's minutes, by day: (hour, minute, steps). Jan 1 is an invalid primary day and Jan 9 the index day,
# so both are left out; Jan 5 has no intraday rows (a valid day without minute data).
MINUTES = {
    date(2023, 1, 2): [(7, 0, 30), (7, 1, 40), (7, 2, 10), (12, 30, 90), (12, 31, 100), (21, 10, 5)],
    date(2023, 1, 3): [(8, 15, 60), (8, 16, 0), (8, 17, 70), (13, 0, 120), (22, 0, 10)],
    date(2023, 1, 4): [(6, 59, 20), (7, 0, 25), (14, 45, 80), (14, 46, 85), (14, 47, 90), (19, 5, 15)],
    date(2023, 1, 6): [(9, 0, 50), (12, 10, 110), (12, 11, 105), (12, 12, 95), (23, 59, 3)],
    date(2023, 1, 7): [(10, 0, 40), (15, 30, 70), (15, 31, 75)],
    date(2023, 1, 8): [(11, 20, 35), (11, 21, 45), (16, 0, 60)],
    date(2023, 1, 1): [(12, 0, 999)],
    date(2023, 1, 9): [(12, 0, 999)],
}

# P1's sleep segments: (sleep_date, start, [(level, minutes), ...]).
NIGHTS = [
    (date(2023, 1, 3), datetime(2023, 1, 2, 23, 0), [("light", 60), ("deep", 30), ("rem", 40), ("wake", 10), ("light", 100)]),
    (date(2023, 1, 7), datetime(2023, 1, 7, 1, 0), [("light", 120), ("rem", 60)]),           # Saturday: free night
    (date(2023, 1, 8), datetime(2023, 1, 8, 0, 30), [("light", 90), ("wake", 15), ("rem", 45)]),  # Sunday: free
    (date(2023, 1, 9), datetime(2023, 1, 9, 0, 0), [("light", 300)]),                       # index day: out
]


@pytest.fixture
def micro():
    con = duckdb.connect()
    act = [(1, date(2023, 1, 1), 50.0)] + [(1, date(2023, 1, 2) + timedelta(days=i), 1000.0) for i in range(10)]
    minutes = [(1, datetime(d.year, d.month, d.day, h, m), float(v)) for d, rows in MINUTES.items() for h, m, v in rows]
    stages = []
    for d, start, segments in NIGHTS:
        t = start
        for level, mins in segments:
            stages.append((1, d, "true", level, t, float(mins)))
            t += timedelta(minutes=mins)
    stages.append((1, date(2023, 1, 3), "false", "light", datetime(2023, 1, 3, 14, 0), 40.0))  # a nap
    tables = {
        "person": pd.DataFrame({"person_id": [1], "year_of_birth": [1980], "sex_at_birth_concept_id": [10]}),
        "concept": pd.DataFrame({"concept_id": [10, 600], "concept_name": ["Female", "Lab"],
                                 "standard_concept": [None, "S"], "domain_id": ["Gender", "Measurement"]}),
        "concept_ancestor": pd.DataFrame({"ancestor_concept_id": [600], "descendant_concept_id": [600]}),
        "measurement": pd.DataFrame({"measurement_id": [1], "person_id": [1], "measurement_concept_id": [600],
                                     "measurement_date": [date(2023, 2, 1)], "value_as_number": [5.0]}),
        "activity_summary": pd.DataFrame(act, columns=["person_id", "date", "steps"]),
        "steps_intraday": pd.DataFrame(minutes, columns=["person_id", "datetime", "steps"]),
        "sleep_level": pd.DataFrame(stages, columns=["person_id", "sleep_date", "is_main_sleep", "level",
                                                     "start_datetime", "duration_in_min"]),
    }
    for name, frame in tables.items():
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    return con


def _expected_activity() -> dict[str, float]:
    """The definitions, written independently of the SQL."""
    days = [d for d in MINUTES if date(2023, 1, 2) <= d < date(2023, 1, 9)]
    grid = np.zeros((len(days), 24))
    for i, d in enumerate(days):
        for h, _m, v in MINUTES[d]:
            grid[i, h] += v
    profile = grid.mean(axis=0)
    var_all = grid.var()
    diffs = np.diff(grid, axis=1).ravel()
    hours = np.arange(24)
    a = 2 * np.mean(profile * np.cos(2 * np.pi * hours / 24))
    b = 2 * np.mean(profile * np.sin(2 * np.pi * hours / 24))
    windows = lambda L: np.array([profile[(s + np.arange(L)) % 24].mean() for s in range(24)])  # noqa: E731
    m10, l5 = windows(10), windows(5)
    l5_onset = int(np.argmin(l5))  # lowest start hour on ties
    peak1, peak30, bouts, active = [], [], 0, 0
    for d in days:
        vals = sorted((v for _h, _m, v in MINUTES[d]), reverse=True)
        peak1.append(vals[0])
        peak30.append(sum(vals[:30]) / 30)
        act_min = sorted(h * 60 + m for h, m, v in MINUTES[d] if v > 0)
        active += len(act_min)
        bouts += sum(1 for k, x in enumerate(act_min) if k == 0 or x - act_min[k - 1] > 1)
    return {"mi_days": len(days), "mi_rel_amplitude": (m10.max() - l5.min()) / (m10.max() + l5.min()),
            "mi_interdaily_stability": profile.var() / var_all, "mi_intradaily_variability": np.mean(diffs ** 2) / var_all,
            "mi_cosinor_rel_amplitude": np.hypot(a, b) / profile.mean(),
            "mi_acrophase_hour": (np.arctan2(b, a) * 24 / (2 * np.pi)) % 24,
            "mi_l5_onset_hour": (l5_onset + 12) % 24 + 12,
            "mi_peak1_mean": np.mean(peak1), "mi_peak30_mean": np.mean(peak30), "mi_fragmentation": bouts / active}


def test_minute_activity_phenotypes_match_their_definitions(micro):
    spec = CohortSpec.from_dict(SPEC)
    out = duckdb_query(micro)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    assert list(out.columns) == output_columns(spec) and len(out) == 1
    row = out.iloc[0]
    for column, value in _expected_activity().items():
        assert row[column] == pytest.approx(value, rel=1e-9, abs=1e-9), column


def test_sleep_stage_phenotypes_match_their_definitions(micro):
    row = duckdb_query(micro)(None, *build_cohort_sql(CohortSpec.from_dict(SPEC), SYNTHETIC_CDR)).iloc[0]
    midpoints, onsets = [25.0, 26.5, 25.75], [23.0, 25.0, 24.5]  # hours after noon
    assert row.ss_days == 3                                       # the index-day night and the nap are out
    assert row.ss_midpoint_hour == pytest.approx(np.mean(midpoints))
    assert row.ss_midpoint_sd == pytest.approx(np.std(midpoints, ddof=1))
    assert row.ss_onset_sd == pytest.approx(np.std(onsets, ddof=1))
    assert row.ss_social_jetlag == pytest.approx(np.mean([26.5, 25.75]) - 25.0)
    assert row.ss_wake_bouts == pytest.approx(2 / 3)
    assert row.ss_transitions_per_hour == pytest.approx(np.mean([5 / (230 / 60), 2 / 3, 3 / (135 / 60)]))
    assert row.ss_rem_latency == pytest.approx(105.0)


def test_roles_take_the_first_column_the_table_has_and_optional_sources_are_skipped():
    source = PhenotypeSource(kind="minute_activity", table="t", prefix="p",
                             columns={"time": ["datetime", "ts"], "value": ["steps"]})
    fixed, reason = resolve(source, {("t", "person_id"), ("t", "ts"), ("t", "steps")})
    assert reason is None and fixed.columns == {"time": "ts", "value": "steps"}
    gone, reason = resolve(source, {("t", "person_id"), ("t", "steps")})
    assert gone is None and "time" in reason
    spec = CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    present = [(w.table, c) for w in spec.wearables[:2]
               for c in [w.date_column, "person_id", *w.columns, *([w.filter_column] if w.filter_column else [])]]
    present += [("steps_intraday", c) for c in ("person_id", "datetime", "steps")]
    fitted, notes = cohort.resolve_sources(spec, present)
    assert [w.prefix for w in fitted.wearables] == ["act", "sleep"]             # optional HR sources dropped
    assert [p.prefix for p in fitted.phenotypes] == ["stepsday"]                # sleep stages not in this CDR
    assert any("heart_rate_summary" in n for n in notes) and any("sleep_level" in n for n in notes)
    wearable_tables = {"activity_summary", "sleep_daily_summary", "steps_intraday"}
    assert not [m for m in cohort.missing_columns(fitted, present) if m.split(".")[0] in wearable_tables]
    assert "stepsday_days" in cohort.availability_columns(fitted)


def test_phenotypes_track_the_latents_that_generated_them():
    from dataclasses import replace

    con = duckdb.connect()
    build_synthetic_cdr(con)
    spec = CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    spec = replace(spec, wearables=spec.wearables[:2], holdout_fraction=0.0, negative_controls=[])
    sql, params = build_cohort_sql(spec, SYNTHETIC_CDR)
    sql = sql.replace("SELECT\n  ie.outcome_value", "SELECT\n  ie.person_id, ie.outcome_value", 1)  # to match latents
    out = duckdb_query(con)(None, sql, params)
    latents = intraday_latents(600)
    idx = (out.person_id - 200_000_000_001).to_numpy()
    rho = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")  # noqa: E731
    assert len(out) >= 50
    assert rho(out.stepsday_acrophase_hour, latents["peak_hour"][idx]) > 0.9
    assert rho(out.stepsday_cosinor_rel_amplitude, latents["rhythm"][idx]) > 0.8
    assert rho(out.sleepstage_onset_sd, latents["onset_sd"][idx]) > 0.8
