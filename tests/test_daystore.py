"""The day store (codas.aou.daystore) on the synthetic CDR, checked against the raw tables by hand, and each
outcome's shares keyed by the rows of its cohort tables."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codas.aou import cohort, daystore
from codas.aou.cohort import CohortSpec
from codas.core import phenotype_dsl as dsl

SPEC = Path(__file__).resolve().parents[1] / "scripts" / "aou" / "hba1c_fitbit.example.json"
DAY_STORE = {"from": ["hr10h", "sleepseg", "stepsday"]}


@pytest.fixture(scope="module")
def synthetic():
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    spec = replace(CohortSpec.from_json(SPEC), window_days=60, lag_days=7, min_wear_days=20, day_store=DAY_STORE)
    present = duckdb_query(con)(None, *cohort.schema_check_sql(spec, SYNTHETIC_CDR))
    spec, _notes = cohort.resolve_sources(spec, list(zip(present.table_name, present.column_name)))
    days = duckdb_query(con)(None, *daystore.build_daystore_sql(spec, SYNTHETIC_CDR))
    return con, spec, days


def test_every_valid_day_in_the_window_with_its_daily_values(synthetic):
    con, spec, days = synthetic
    assert list(days.columns) == ["person_id", "day", "weekday", *daystore.day_columns(spec)]
    assert days["day"].between(-60, -1).all() and set(days["weekday"]) == set(range(1, 8))
    act = con.execute("SELECT * FROM activity_summary").df()
    act["d"] = pd.to_datetime(act["date"]).dt.date
    valid = act[(act.steps >= 100) & (act.steps <= 45000)]
    first = valid.groupby("person_id").d.min()
    for pid in days.person_id.drop_duplicates().sample(5, random_state=0):
        index = first[pid] + timedelta(days=60)
        mine = valid[(valid.person_id == pid) & (valid.d >= index - timedelta(days=60)) & (valid.d < index)]
        one = days[days.person_id == pid].set_index("day").sort_index()
        assert list(one.index) == sorted((d - index).days for d in mine.d)
        steps = mine.assign(day=[(d - index).days for d in mine.d]).groupby("day").steps.mean()
        assert np.allclose(one["act_steps"], steps.loc[one.index])
        # Sunday is 1, Saturday 7.
        weekday = [((index + timedelta(days=int(k))).isoweekday() % 7) + 1 for k in one.index]
        assert list(one["weekday"]) == weekday


def _index_date(con, person_id):
    """The person's index date: the first valid wear day plus the 60-day window."""
    first = con.execute(f"SELECT MIN(CAST(date AS DATE)) FROM activity_summary WHERE person_id = {int(person_id)} "
                        "AND steps BETWEEN 100 AND 45000").fetchone()[0]
    return pd.to_datetime(first).date() + timedelta(days=60)


def test_minute_level_signals_by_hand(synthetic):
    con, spec, days = synthetic
    hr_source = next(p for p in spec.phenotypes if p.prefix == "hr10h")
    hr = con.execute(f"SELECT person_id, {hr_source.columns['time']} AS t, {hr_source.columns['value']} AS v "
                     f"FROM {hr_source.table}").df()
    hr = hr[(hr.v >= 30) & (hr.v <= 220)]
    hr["d"] = hr.t.dt.date
    row = days.dropna(subset=["hr10h_day_rest"]).iloc[0]
    day = _index_date(con, row.person_id) + timedelta(days=int(row.day))
    minutes = hr[(hr.person_id == row.person_id) & (hr.d == day)].v.to_numpy()
    assert row.hr10h_day_minutes == len(minutes) >= 600
    assert row.hr10h_day_rest == pytest.approx(np.sort(minutes)[:30].mean())
    assert row.hr10h_day_mean == pytest.approx(minutes.mean())
    steps = con.execute("SELECT person_id, datetime AS t, steps AS v FROM steps_intraday "
                        f"WHERE person_id = {int(row.person_id)}").df()
    steps["d"] = steps.t.dt.date
    s = steps[steps.d == day]
    s = s[s.v.notna()]
    mine = days[(days.person_id == row.person_id) & (days.day == row.day)].iloc[0]
    assert mine.stepsday_active_min == (s.v > 0).sum() and mine.stepsday_brisk_min == (s.v >= 100).sum()
    top = np.sort(s.v.to_numpy())[::-1][:30]
    assert mine.stepsday_peak30 == pytest.approx(top.sum() / 30)
    minute_of_day = s.t.dt.hour * 60 + s.t.dt.minute
    assert mine.stepsday_first_active == pytest.approx(minute_of_day[s.v > 0].min() / 60)
    assert mine.stepsday_last_active == pytest.approx(minute_of_day[s.v > 0].max() / 60)


def test_each_counted_night_by_hand(synthetic):
    con, spec, days = synthetic
    src = next(p for p in spec.phenotypes if p.prefix == "sleepseg")
    col = {r: src.columns[r] for r in ("date", "start", "level", "duration")}
    seg = con.execute(f"SELECT person_id, {col['date']} AS d, {col['start']} AS st, {col['level']} AS lvl, "
                      f"{col['duration']} AS dur FROM {src.table}"
                      + (f" WHERE LOWER(CAST({src.columns['main']} AS VARCHAR)) = 'true'" if src.columns.get('main') else "")).df()
    row = days.dropna(subset=["sleepseg_onset"]).iloc[3]
    night = _index_date(con, row.person_id) + timedelta(days=int(row.day))
    mine = seg[(seg.person_id == row.person_id) & (pd.to_datetime(seg.d).dt.date == night)]
    onset = pd.to_datetime(mine.st).min()
    asleep = mine[~mine.lvl.str.lower().isin(["wake", "awake"])].dur.sum()
    assert row.sleepseg_asleep == pytest.approx(asleep) and 180 <= asleep <= 960
    assert row.sleepseg_onset == pytest.approx((onset.hour + 12) % 24 + onset.minute / 60 + 12)
    assert row.sleepseg_wake_bouts == mine.lvl.str.lower().isin(["wake", "awake"]).sum()


def test_each_share_is_keyed_by_its_rows_and_phenotypes_land_in_the_table(tmp_path):
    days = pd.DataFrame({"person_id": [30] * 20 + [10] * 20 + [40] * 20, "day": list(range(-20, 0)) * 3,
                         "weekday": [(d % 7) + 1 for d in range(20)] * 3,
                         "sig": list(np.arange(20.0)) + [5.0] * 20 + list(2 * np.arange(20.0))})
    csv = tmp_path / "x.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)   # rows: persons 10, 30, 20 (no days)
    daystore.split(days, pd.Series([10, 30, 20]), pd.Series([40]), csv)
    disc = pd.read_csv(daystore.days_path(csv))
    assert "person_id" not in disc.columns and sorted(disc.row.unique()) == [0, 1]
    assert (disc[disc.row == 0].sig == 5.0).all() and disc[disc.row == 1].sig.max() == 19
    sealed = pd.read_csv(daystore.holdout_days_path(csv))
    assert daystore.holdout_days_path(csv).parent.name == "sealed" and (sealed.row == 0).all()
    definition = {"name": "sig_trend", "signal": "sig", "summary": "slope_per_30d", "min_days": 10}
    (column,) = dsl.add_columns([definition], disc, csv)
    table = pd.read_csv(csv)
    assert column == dsl.column_name(definition) and list(table.a) == [1.0, 2.0, 3.0]
    assert table[column].iloc[0] == pytest.approx(0.0) and table[column].iloc[1] == pytest.approx(30.0)
    assert np.isnan(table[column].iloc[2])                                  # person 20 has no days


def test_the_extraction_writes_each_outcomes_day_tables(tmp_path, synthetic, monkeypatch):
    con, spec, _days = synthetic
    from codas.aou.synthetic import SYNTHETIC_CDR, duckdb_query

    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    out = cohort.ROOT / ".codas_runs" / "daystore_test"
    try:
        csv = cohort.extract_cohort(replace(spec, holdout_fraction=0.5), SYNTHETIC_CDR, out, client=object(),
                                    check_schema=False)
        table, days = pd.read_csv(csv), pd.read_csv(daystore.days_path(csv))
        assert "person_id" not in table.columns and "person_id" not in days.columns
        assert daystore.holdout_days_path(csv).exists()
        # Row k of the day table is row k of the cohort table: its mean daily steps match.
        means = days.groupby("row").act_steps.mean()
        assert np.allclose(table.loc[means.index, "act_steps_mean"], means, rtol=1e-9)
    finally:
        import shutil

        shutil.rmtree(out, ignore_errors=True)


def test_a_day_store_needs_the_incident_design_and_minute_sources():
    base = CohortSpec.from_json(SPEC)
    with pytest.raises(ValueError, match="incident design"):
        daystore.validate(replace(base, design="window", negative_controls=[], day_store=DAY_STORE))
    with pytest.raises(ValueError, match="not a minute heart-rate"):
        daystore.validate(replace(base, day_store={"from": ["act"]}))
    with pytest.raises(ValueError, match="day_store must be"):
        replace(base, day_store={"from": "hr10h"}).validate()
    # A derived spec without the sources (a feasibility count, a CDR that lacks them) is still a valid spec.
    replace(base, phenotypes=[]).validate()
