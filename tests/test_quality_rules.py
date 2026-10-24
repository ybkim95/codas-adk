"""Quality rules of the extraction, each executed in DuckDB on a hand-built micro-CDR and compared with a
hand computation: heart-rate wear days, heart rate during the main-sleep period, night-length bounds,
contributing days per phenotype, value bounds on daily rows, the two-date case rule, EHR activity before
the index date, visit-count and table-membership covariates, and the outcome-unit summary."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from codas.aou import cohort  # noqa: E402
from codas.aou.cohort import CohortSpec, build_cohort_sql, quality_rules, unit_mix  # noqa: E402
from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query  # noqa: E402

PEOPLE = [1, 2, 3, 4, 5]
FIRST = date(2023, 1, 2)            # first valid wear day; with a 7-day window the index date is Jan 9
INDEX = FIRST + timedelta(days=7)

BASE = {
    "name": "qc", "design": "incident", "window_days": 7, "lag_days": 0, "min_wear_days": 2,
    "valid_day": {"column": "steps", "min": 100},
    "outcome": {"kind": "measurement", "name": "lab", "concept_ids": [600]},
    "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]}],
}
HR = {"kind": "minute_heart_rate", "table": "heart_rate_minute_level", "prefix": "hr",
      "columns": {"time": "datetime", "value": "heart_rate_value"}}
SEG = {"kind": "sleep_stages", "table": "sleep_level", "prefix": "seg",
       "columns": {"date": "sleep_date", "start": "start_datetime", "level": "level",
                   "duration": "duration_in_min", "main": "is_main_sleep"}}


def _cdr(**extra: pd.DataFrame):
    """Five people with ten valid wear days from Jan 2, a lab value after the window, visits through Dec."""
    tables = {
        "person": pd.DataFrame({"person_id": PEOPLE, "year_of_birth": 1980, "sex_at_birth_concept_id": 10}),
        "concept": pd.DataFrame({"concept_id": [10, 600, 8554, 9579], "concept_name": ["Female", "Lab", "percent",
                                                                                     "millimole per mole"],
                                 "standard_concept": [None, "S", "S", "S"],
                                 "domain_id": ["Gender", "Measurement", "Unit", "Unit"]}),
        "concept_ancestor": pd.DataFrame({"ancestor_concept_id": [600, 700, 701], "descendant_concept_id": [600, 700, 701]}),
        "measurement": pd.DataFrame({"measurement_id": PEOPLE, "person_id": PEOPLE, "measurement_concept_id": 600,
                                     "measurement_date": date(2023, 2, 1), "value_as_number": [5.0, 6.0, 7.0, 8.0, 9.0],
                                     "unit_concept_id": 8554}),
        "activity_summary": pd.DataFrame([(p, FIRST + timedelta(days=i), 1000.0) for p in PEOPLE for i in range(10)],
                                         columns=["person_id", "date", "steps"]),
        "visit_occurrence": pd.DataFrame([(p, date(2023, 12, 1)) for p in PEOPLE], columns=["person_id", "visit_start_date"]),
        "condition_occurrence": pd.DataFrame(columns=["person_id", "condition_concept_id", "condition_start_date"]),
    }
    tables.update(extra)
    con = duckdb.connect()
    for name, frame in tables.items():
        frame = frame.copy()
        for column in frame.columns:
            if column.endswith("date") and not column.endswith("datetime"):
                frame[column] = pd.to_datetime(frame[column]).dt.date
        con.register("_t", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    return con


def _run(con, spec: dict, **kw) -> pd.DataFrame:
    sql, params = build_cohort_sql(CohortSpec.from_dict(spec), SYNTHETIC_CDR, **kw)
    sql = sql.replace("SELECT\n  ie.outcome_value", "SELECT\n  ie.person_id, ie.outcome_value", 1)
    return duckdb_query(con)(None, sql, params).set_index("person_id").sort_index()


def _minutes(day: date, start_minute: int, count: int, value) -> list[tuple]:
    base = datetime(day.year, day.month, day.day)
    return [(1, base + timedelta(minutes=start_minute + k), float(value(k) if callable(value) else value))
            for k in range(count)]


# --- heart rate: wear days, sleep periods, contributing days ---------------------------------------------

def _wear_day_minutes() -> pd.DataFrame:
    rows = (_minutes(FIRST, 0, 700, lambda k: 60 + k % 20)                       # 700 minutes from 00:00
            + _minutes(FIRST + timedelta(days=1), 0, 650, lambda k: 62 + k % 15)  # 650 minutes
            + _minutes(FIRST + timedelta(days=2), 360, 100, 90)                 # 100 minutes from 06:00
            + _minutes(INDEX, 0, 700, 99))                                       # the index day: outside the window
    return pd.DataFrame(rows, columns=["person_id", "datetime", "heart_rate_value"])


def test_heart_rate_counts_only_days_with_enough_minutes():
    minutes = _wear_day_minutes()
    con = _cdr(heart_rate_minute_level=minutes)
    worn = _run(con, {**BASE, "phenotypes": [{**HR, "min_minutes_per_day": 600}]}).loc[1]
    every = _run(con, {**BASE, "phenotypes": [HR]}).loc[1]
    m = minutes[(minutes.datetime < datetime(2023, 1, 4))].copy()               # Jan 2 and 3: 600+ minutes
    m["d"], m["h"] = m.datetime.dt.date, m.datetime.dt.hour
    resting = m.groupby("d").heart_rate_value.apply(lambda v: np.sort(v.to_numpy())[:30].mean()).mean()
    night, day = m[m.h < 6].heart_rate_value.mean(), m[(m.h >= 10) & (m.h < 20)].heart_rate_value.mean()
    assert worn.hr_days == 2 and every.hr_days == 3                               # the 100-minute day is out
    assert worn.hr_resting_hr == pytest.approx(resting) and worn.hr_night_hr == pytest.approx(night)
    assert worn.hr_day_hr == pytest.approx(day) and worn.hr_night_dip == pytest.approx(1 - night / day)
    assert every.hr_resting_hr > worn.hr_resting_hr                               # the 90-bpm hour pulled it up


IN_BED = [(datetime(2023, 1, 2, 23), datetime(2023, 1, 3, 4)),        # long main sleep, 270 min asleep
          (datetime(2023, 1, 3, 23), datetime(2023, 1, 4, 0, 40))]      # short main sleep, 100 min asleep
WALK = (datetime(2023, 1, 2, 12), datetime(2023, 1, 2, 12, 30))


def _sleep_tables() -> dict[str, pd.DataFrame]:
    """Heart rate every minute from Jan 2 08:00 to Jan 4 17:59: 50 bpm in the long main sleep, 40 in the short
    one, 70 in a daytime nap, 120 in a 30-minute walk (with steps), 80 otherwise."""
    long_night, short_night = IN_BED
    nap = (datetime(2023, 1, 3, 14), datetime(2023, 1, 3, 14, 40))
    stamps = pd.date_range(datetime(2023, 1, 2, 8), datetime(2023, 1, 4, 17, 59), freq="min")

    def bpm(t):
        for (lo, hi), value in ((long_night, 50.0), (short_night, 40.0), (nap, 70.0), (WALK, 120.0)):
            if lo <= t < hi:
                return value
        return 80.0

    minutes = pd.DataFrame({"person_id": 1, "datetime": stamps, "heart_rate_value": [bpm(t) for t in stamps]})
    stages = []
    for sleep_date, start, segments, main in ((date(2023, 1, 3), long_night[0], [("light", 120), ("wake", 30), ("deep", 150)], "true"),
                                              (date(2023, 1, 4), short_night[0], [("light", 100)], "true"),
                                              (date(2023, 1, 3), nap[0], [("light", 40)], "false")):
        t = start
        for level, length in segments:
            stages.append((1, sleep_date, main, level, t, float(length)))
            t += timedelta(minutes=length)
    sleep = pd.DataFrame(stages, columns=["person_id", "sleep_date", "is_main_sleep", "level", "start_datetime",
                                          "duration_in_min"])
    steps = pd.DataFrame({"person_id": 1, "datetime": pd.date_range(WALK[0], WALK[1], freq="min", inclusive="left"),
                          "steps": 100.0})
    return {"heart_rate_minute_level": minutes, "sleep_level": sleep, "steps_intraday": steps}


STEPS = {"kind": "minute_activity", "table": "steps_intraday", "prefix": "st",
         "columns": {"time": "datetime", "value": "steps"}}


def test_sleep_heart_rate_follows_the_main_sleep_period():
    tables = _sleep_tables()
    con = _cdr(**tables)
    bounded = {**SEG, "night_minutes": [180, 960]}
    linked = {**HR, "sleep_from": "seg", "steps_from": "st"}
    row = _run(con, {**BASE, "phenotypes": [bounded, STEPS, linked]}).loc[1]
    loose = _run(con, {**BASE, "phenotypes": [SEG, STEPS, linked]}).loc[1]
    m = tables["heart_rate_minute_level"]
    day = m[(m.datetime.dt.hour >= 10) & (m.datetime.dt.hour < 20)].heart_rate_value.mean()
    awake = m[~m.datetime.apply(lambda t: any(lo <= t < hi for lo, hi in IN_BED + [WALK]))]
    assert row.seg_days == 1 and loose.seg_days == 2           # the 100-minute night is out of 3-16 h
    assert row.hr_sleep_hr == pytest.approx(50.0)              # the long night only; the nap never counts
    assert row.hr_sleep_dip == pytest.approx(1 - 50.0 / day)
    assert loose.hr_sleep_hr == pytest.approx((300 * 50.0 + 100 * 40.0) / 400)  # both main sleeps, by minute
    # Awake and inactive: neither night (counted or not) nor the walk; the nap is not main sleep, so it stays.
    assert row.hr_sedentary_hr == pytest.approx(awake.heart_rate_value.mean())
    assert loose.hr_sedentary_hr == pytest.approx(row.hr_sedentary_hr)
    plain = _run(con, {**BASE, "phenotypes": [SEG, HR]})
    assert "hr_sleep_hr" not in plain.columns                  # no sleep source named: no sleep columns
    assert "hr_sedentary_hr" not in _run(con, {**BASE, "phenotypes": [SEG, {**HR, "sleep_from": "seg"}]}).columns


def test_phenotypes_are_missing_with_too_few_contributing_days():
    con = _cdr(heart_rate_minute_level=_wear_day_minutes())
    row = _run(con, {**BASE, "phenotypes": [{**HR, "min_days": 4}]}).loc[1]
    assert row.hr_days == 3 and pd.isna(row.hr_resting_hr) and pd.isna(row.hr_night_dip)
    assert not pd.isna(_run(con, {**BASE, "phenotypes": [{**HR, "min_days": 3}]}).loc[1, "hr_resting_hr"])


# --- daily rows, cases, EHR activity, covariates ------------------------------------------------------------

def test_daily_rows_outside_their_bounds_are_left_out():
    nights = pd.DataFrame({"person_id": 1, "sleep_date": [FIRST + timedelta(days=i) for i in range(4)],
                           "minute_asleep": [100.0, 400.0, 500.0, 1000.0]})
    con = _cdr(sleep_daily_summary=nights)
    source = {"table": "sleep_daily_summary", "date_column": "sleep_date", "prefix": "sl", "columns": ["minute_asleep"]}
    bounded = _run(con, {**BASE, "wearables": BASE["wearables"] + [{**source, "value_bounds": {"minute_asleep": [180, 960]}}]})
    raw = _run(con, {**BASE, "wearables": BASE["wearables"] + [source]})
    assert bounded.loc[1, "sl_days"] == 2 and bounded.loc[1, "sl_minute_asleep_mean"] == pytest.approx(450.0)
    assert bounded.loc[1, "sl_minute_asleep_sd"] == pytest.approx(np.std([400, 500], ddof=1))
    assert raw.loc[1, "sl_days"] == 4 and raw.loc[1, "sl_minute_asleep_mean"] == pytest.approx(500.0)


DIAGNOSES = pd.DataFrame([
    (1, 700, date(2023, 2, 1)),                               # one date after follow-up starts
    (2, 700, date(2023, 2, 1)), (2, 700, date(2023, 3, 1)),   # two dates: a case from Feb 1
    (3, 700, date(2023, 1, 5)), (3, 700, date(2023, 2, 1)),   # diagnosed before follow-up: prevalent
    (5, 700, date(2023, 2, 1)), (5, 700, date(2023, 2, 1)),   # two records on one date: one date
    (1, 701, date(2023, 2, 1)), (2, 701, date(2023, 4, 1)), (2, 701, date(2023, 5, 1)), (3, 701, date(2023, 1, 3)),
], columns=["person_id", "condition_concept_id", "condition_start_date"])


def test_a_case_needs_diagnoses_on_enough_distinct_dates():
    con = _cdr(condition_occurrence=DIAGNOSES)
    outcome = {"kind": "condition", "name": "dx", "concept_ids": [700]}
    two = _run(con, {**BASE, "outcome": {**outcome, "min_occurrences": 2}})
    one = _run(con, {**BASE, "outcome": outcome})
    assert list(two.index) == [2, 4] and two.dx.tolist() == [1, 0]         # 1 and 5: one date; 3: prevalent
    assert two.loc[2, "followup_days"] == (date(2023, 2, 1) - INDEX).days     # the first of the two dates
    assert list(one.index) == [1, 2, 4, 5] and one.dx.tolist() == [1, 1, 0, 1]
    # Outcome-anchored: a case's window ends at its first diagnosis, so a longer window keeps wear in it.
    window = _run(con, {**BASE, "design": "window", "window_days": 60, "outcome": {**outcome, "min_occurrences": 2}})
    assert list(window.index) == [2, 3, 4] and window.dx.tolist() == [1, 1, 0]  # no prevalence rule here
    controls = {**BASE, "negative_controls": [{"name": "ctl", "concept_ids": [701], "min_occurrences": 2}]}
    nc = _run(con, controls)
    assert pd.isna(nc.loc[1, "nc_ctl"]) and nc.loc[2, "nc_ctl"] == 1 and pd.isna(nc.loc[3, "nc_ctl"])
    assert nc.loc[4, "nc_ctl"] == 0 and nc.loc[2, "nc_ctl_days"] == (date(2023, 4, 1) - INDEX).days


def test_ehr_activity_before_the_index_date_and_visit_and_membership_covariates():
    visits = pd.DataFrame([(1, date(2022, 12, 20)), (1, date(2023, 1, 1)), (1, date(2023, 1, 1)),
                           (2, date(2022, 6, 1)), (3, INDEX)] + [(p, date(2023, 12, 1)) for p in PEOPLE],
                          columns=["person_id", "visit_start_date"])
    con = _cdr(visit_occurrence=visits, wear_study=pd.DataFrame({"person_id": [1, 4]}))
    covariates = [{"name": "visits", "kind": "visits", "lookback_days": 30},
                  {"name": "enrolled", "kind": "in_table", "table": "wear_study"}]
    out = _run(con, {**BASE, "covariates": covariates})
    assert out.visits.tolist() == [2, 0, 0, 0, 0]              # distinct dates in the 30 days before Jan 9
    assert out.enrolled.tolist() == [1, 0, 0, 1, 0]
    active = _run(con, {**BASE, "ehr_lookback_days": 30})
    assert list(active.index) == [1]                            # 2: too long ago; 3: on the index date itself
    assert list(_run(con, {**BASE, "ehr_lookback_days": 365}).index) == [1, 2]


# --- the extraction's quality summary: outcome units and split-half reliability --------------------------------

def test_each_half_of_the_window_is_summarised_for_split_half_reliability():
    steps = pd.DataFrame([(p, FIRST + timedelta(days=i), float(100 * (i + 1))) for p in PEOPLE for i in range(10)],
                         columns=["person_id", "date", "steps"])
    out = _run(_cdr(activity_summary=steps), BASE, qc=True).loc[1]
    # Window Jan 2-8; the first half is the days before Jan 6 (index minus half of the 7 days).
    assert out._act_steps_mean_h1 == pytest.approx(250.0) and out._act_steps_mean_h2 == pytest.approx(600.0)
    assert out._act_steps_sd_h1 == pytest.approx(np.std([100, 200, 300, 400], ddof=1))
    assert out._act_steps_sd_h2 == pytest.approx(np.std([500, 600, 700], ddof=1))
    assert "_act_steps_mean_h1" not in _run(_cdr(activity_summary=steps), BASE).columns


def test_split_half_reliability_needs_enough_people():
    from codas.aou.cohort import split_half

    spec = CohortSpec.from_dict(BASE)
    rng = np.random.default_rng(3)
    trait = rng.normal(0, 1, 60)
    frame = pd.DataFrame({"_act_steps_mean_h1": trait + rng.normal(0, 0.3, 60),
                          "_act_steps_mean_h2": trait + rng.normal(0, 0.3, 60),
                          "_act_steps_sd_h1": rng.normal(0, 1, 60), "_act_steps_sd_h2": rng.normal(0, 1, 60)})
    result = split_half(frame, spec)
    rho = frame["_act_steps_mean_h1"].rank().corr(frame["_act_steps_mean_h2"].rank())
    assert result["act_steps_mean"]["reliability"] == pytest.approx(2 * rho / (1 + rho), abs=1e-3)
    assert result["act_steps_mean"]["reliability"] > 0.8 and result["act_steps_sd"]["reliability"] < 0.5
    assert split_half(frame.iloc[:39], spec) == {}              # fewer than 2 x 20 people: not estimated


def test_unit_summary_masks_small_groups_and_flags_mixed_scales():
    frame = pd.DataFrame({"lab": np.r_[np.linspace(5, 7, 100), np.linspace(40, 60, 30), [9.0] * 5],
                          cohort.UNIT_COLUMN: [8554] * 100 + [9579] * 30 + [np.nan] * 5,
                          cohort.UNIT_NAME_COLUMN: ["percent"] * 100 + ["millimole per mole"] * 30 + [None] * 5})
    mix = unit_mix(frame, "lab")
    assert mix["mixed"] and [u["unit"] for u in mix["units"]] == ["percent", "millimole per mole", "not recorded"]
    percent, mmol, unknown = mix["units"]
    assert percent["share"] == "74%" and percent["median"] == pytest.approx(6.0)   # 100 people: quartiles shown
    assert mmol["share"] == "22%" and "median" not in mmol                        # 30 people: under 2 x 20
    assert unknown["share"] == "20 people or fewer" and unknown["unit_concept_id"] is None
    assert not unit_mix(frame.iloc[:100], "lab")["mixed"]
    assert unit_mix(frame.drop(columns=[cohort.UNIT_COLUMN]), "lab") == {"units": [], "mixed": False}


def test_unit_summary_comes_from_the_discovery_share_and_an_allowlist_keeps_one_unit(monkeypatch):
    con = duckdb.connect()
    build_synthetic_cdr(con)
    spec = CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    from dataclasses import replace

    # Outcome-anchored, so most people with a lab value are in (the prospective lab window is short here).
    spec = replace(spec, design="window", lag_days=0, horizon_days=None, phenotypes=[], wearables=spec.wearables[:1],
                   covariates=[], negative_controls=[], positive_controls=["age"], panel=None, ehr_lookback_days=None)
    sealed = [int(p) for (p,) in con.execute("SELECT person_id FROM person").fetchall()
              if cohort.holdout_bucket(int(p), spec.split_salt) < 500]
    con.execute("UPDATE measurement SET unit_concept_id = 9579 WHERE person_id IN (" + ",".join(map(str, sealed)) + ")")
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    out_dir = cohort.ROOT / ".codas_runs" / "units_test"
    try:
        path = cohort.extract_cohort(spec, SYNTHETIC_CDR, out_dir, client=object(), check_schema=False)
        qc = json.loads(cohort.qc_path(path).read_text())
        mix, reliability = qc["units"], qc["reliability"]
        assert [u["unit"] for u in mix["units"]] == ["percent"] and mix["units"][0]["share"] == "100%"
        assert not mix["mixed"]                                 # the sealed share's unit never shows
        # Synthetic steps are a trait of the person; fairly active minutes are day-to-day noise.
        assert reliability["act_steps_mean"]["reliability"] > 0.9
        assert abs(reliability["act_fairly_active_minutes_mean"]["reliability"]) < 0.4
        for table in (path, cohort.holdout_path(path)):         # no internal column reaches a table
            assert not [c for c in pd.read_csv(table).columns if c.startswith("_")]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    every = duckdb_query(con)(None, *build_cohort_sql(replace(spec, holdout_fraction=0.0), SYNTHETIC_CDR))
    kept = duckdb_query(con)(None, *build_cohort_sql(replace(spec, holdout_fraction=0.0, outcome=replace(
        spec.outcome, unit_concept_ids=[8554])), SYNTHETIC_CDR))
    assert 0 < len(kept) < len(every)                           # the sealed people's values were in mmol/mol


# --- rules in words, resolution, validation ------------------------------------------------------------------

def test_quality_rules_are_stated_in_words_and_reach_the_goal():
    from codas.aou.runner import goal_for

    spec = CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    rules = " | ".join(quality_rules(spec))
    for phrase in ("activity_summary.steps 100-45000", "minute_asleep outside 180-960", "600 heart-rate minutes",
                   "180-960 min asleep", "at least 14 contributing days", "EHR visit in the 365 days"):
        assert phrase in rules, phrase
    assert "negative control" not in rules                      # sealed-table columns the agents never see
    assert "negative control impacted_cerumen" in " | ".join(quality_rules(spec, negative_controls=True))
    assert "600 heart-rate minutes" in goal_for(spec)
    condition = CohortSpec.from_dict({**BASE, "outcome": {"kind": "condition", "name": "dx", "concept_ids": [700],
                                                         "min_occurrences": 2}})
    assert any("2 distinct dates" in r for r in quality_rules(condition))


def test_resolution_drops_what_this_cdr_lacks():
    spec = CohortSpec.from_dict({**BASE, "phenotypes": [SEG, {**HR, "sleep_from": "seg", "min_minutes_per_day": 600,
                                                              "min_days": 3, "columns": {"time": ["ts", "datetime"],
                                                                                         "value": "heart_rate_value"}}],
                                 "covariates": [{"name": "enrolled", "kind": "in_table", "table": "wear_study"}]})
    present = [("activity_summary", c) for c in ("person_id", "date", "steps")]
    present += [("heart_rate_minute_level", c) for c in ("person_id", "datetime", "heart_rate_value")]
    fitted, notes = cohort.resolve_sources(spec, present)
    (hr,) = fitted.phenotypes
    assert hr.prefix == "hr" and not hr.sleep_from and hr.columns["time"] == "datetime"
    assert (hr.min_minutes_per_day, hr.min_days) == (600, 3)       # fitting a source keeps all its settings
    assert "hr_sleep_hr" not in cohort.output_columns(fitted) and not fitted.covariates
    assert any("sleep_hr" in n for n in notes) and any("wear_study" in n for n in notes)


@pytest.mark.parametrize("change", [
    {"wearables": [{**BASE["wearables"][0], "value_bounds": {"steps": [5, 1]}}]},
    {"outcome": {**BASE["outcome"], "min_occurrences": 2}},
    {"outcome": {"kind": "condition", "name": "dx", "concept_ids": [700], "unit_concept_ids": [8554]}},
    {"phenotypes": [{**SEG, "min_minutes_per_day": 600}]},
    {"phenotypes": [{**HR, "night_minutes": [180, 960]}]},
    {"phenotypes": [{**HR, "min_days": 0}]},
    {"covariates": [{"name": "member", "kind": "in_table"}]},
    {"covariates": [{"name": "visits", "kind": "visits", "concept_ids": [9202]}]},
    {"ehr_lookback_days": 0},
])
def test_invalid_quality_settings_are_refused(change):
    with pytest.raises(ValueError):
        CohortSpec.from_dict({**BASE, **change})


def test_catalog_defaults_fill_what_an_entry_leaves_out(tmp_path):
    from codas.aou.feasibility import load_catalog

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"defaults": {"condition": {"min_occurrences": 2}}, "outcomes": [
        {"name": "a", "label": "A", "kind": "condition", "concept_ids": [1], "expect_name": "a"},
        {"name": "b", "label": "B", "kind": "condition", "concept_ids": [2], "expect_name": "b", "min_occurrences": 1},
        {"name": "c", "label": "C", "kind": "measurement", "concept_ids": [3], "expect_name": "c"}]}))
    a, b, c = load_catalog(path)
    assert (a.outcome_spec().min_occurrences, b.outcome_spec().min_occurrences, c.outcome_spec().min_occurrences) == (2, 1, 1)
    real = {o.kind: o.min_occurrences for o in load_catalog(cohort.ROOT / "scripts" / "aou" / "outcomes_catalog.json")}
    assert real == {"measurement": 1, "condition": 2}


def test_the_dossier_names_the_outcome_units_and_unreliable_features():
    from codas.core.eda import digest

    one = {"units": [{"unit": "percent", "share": "100%"}], "mixed": False}
    two = {"units": [{"unit": "percent", "share": "80%"}, {"unit": "millimole per mole", "share": "20%"}], "mixed": True}
    assert "Outcome units: percent (100%)." in digest({"features": {}, "outcome_units": one})
    assert "more than one unit" in digest({"features": {}, "outcome_units": two})
    text = digest({"features": {}, "reliability": {"a_mean": {"reliability": 0.95}, "b_sd": {"reliability": 0.31}}})
    assert "Unreliable over the window" in text and "b_sd (0.31)" in text and "a_mean" not in text
