"""A synthetic, All of Us–shaped CDR in DuckDB, for testing and demos outside the Workbench.

No row here describes a real person. The tables carry the OMOP and Fitbit columns the extractor
reads, with planted associations (more steps -> lower HbA1c and less type 2 diabetes; shorter,
more variable sleep -> more depression) so an end-to-end run has something to find. A separate latent
body size raises calories burned, BMI and HbA1c and lowers HDL, so calories look like a biomarker
until BMI is adjusted for. One negative-control condition (impacted cerumen) is diagnosed at random,
unrelated to anything else, so a confirmation step should find no association with it. Lab values carry
their unit; a third of the people are listed in ``wear_study`` (a sub-study enrolment table); minute heart
rate covers 08:00-17:59 on most days, so the usual 10-hours-of-data rule keeps most days and drops some;
most diagnoses are recorded on a second date, and most people have a routine visit in the first half-year.

``duckdb_query`` runs the extractor's BigQuery SQL against this database: it inlines the bound
parameters and transpiles BigQuery to DuckDB with sqlglot. Install ``duckdb`` and ``sqlglot`` to use
it; neither is needed in the Workbench.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

SYNTHETIC_CDR = "memory.main"

# Concept rows the catalog refers to (ids as in the OMOP vocabulary; names abbreviated).
CONCEPTS = [
    (3004410, "Hemoglobin A1c/Hemoglobin.total in Blood", "S", "Measurement"),
    (3007070, "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "S", "Measurement"),
    (201826, "Type 2 diabetes mellitus", "S", "Condition"),
    (440383, "Depressive disorder", "S", "Condition"),
    (4193704, "Type 2 diabetes mellitus without complication", "S", "Condition"),
    (3038553, "Body mass index (BMI) [Ratio]", "S", "Measurement"),
    (1503297, "metformin", "S", "Drug"),
    (374375, "Impacted cerumen", "S", "Condition"),
    (380706, "Regular astigmatism", "S", "Condition"),
    (140641, "Verruca vulgaris", "S", "Condition"),
    (1585847, "Female", None, "Gender"),
    (1585846, "Male", None, "Gender"),
    (990001, "Synthetic repeated laboratory value", "S", "Measurement"),
    (21600712, "DRUGS USED IN DIABETES", "C", "Drug"),
    (1585857, "Smoking: 100 Cigs Lifetime", None, "Observation"),
    (1586000, "Smoking: Smoking Frequency", None, "Observation"),
    (1585858, "100 Cigs Lifetime: Yes", None, "Answer"),
    (1585859, "100 Cigs Lifetime: No", None, "Answer"),
    (903096, "PMI: Skip", None, "Answer"),
    # Units (ids as in the OMOP vocabulary to the best of our knowledge; only the names are read here).
    (8554, "percent", "S", "Unit"),
    (8840, "milligram per deciliter", "S", "Unit"),
    (9531, "kilogram per square meter", "S", "Unit"),
]
# The unit each synthetic lab is recorded in.
MEASUREMENT_UNITS = {3004410: 8554, 3007070: 8840, 3038553: 9531, 990001: 8840}
# Vocabulary and code of the concepts a spec can name by public code instead of by id.
CODES = {201826: ("SNOMED", "44054006"), 3004410: ("LOINC", "4548-4"), 3038553: ("LOINC", "39156-5"),
         1503297: ("RxNorm", "6809"), 21600712: ("ATC", "A10"), 1585857: ("PPI", "Smoking_100CigsLifetime"),
         1586000: ("PPI", "Smoking_SmokeFrequency"), 1585858: ("PPI", "CigsLifetime_Yes"),
         1585859: ("PPI", "CigsLifetime_No"), 903096: ("PPI", "PMI_Skip")}
# A synthetic lab measured several times per person, for the within-person (panel) tests: it rises with a
# person's steps over the 90 days before each measurement relative to their own average (a planted
# within-person effect) and with their average steps (a between-person effect).
REPEATED_LAB = 990001
_ANCESTRY = [(201826, 201826), (201826, 4193704), (3004410, 3004410), (3007070, 3007070),
             (440383, 440383), (3038553, 3038553), (1503297, 1503297), (374375, 374375), (380706, 380706),
             (140641, 140641), (990001, 990001), (21600712, 21600712), (21600712, 1503297)]


def build_synthetic_cdr(con: Any, n_people: int = 600, n_days: int = 240, seed: int = 0) -> None:
    """Create person, concept, concept_ancestor, measurement, condition_occurrence, drug_exposure,
    visit_occurrence, observation, activity_summary, sleep_daily_summary, steps_intraday, sleep_level,
    heart_rate_summary, heart_rate_minute_level and wear_study in the DuckDB connection ``con``."""
    rng = np.random.default_rng(seed)
    # Body size draws from its own stream, so adding it left every other synthetic value unchanged.
    body_rng = np.random.default_rng(seed + 1)
    body = body_rng.normal(0, 1, n_people)        # latent: drives calories, BMI, HbA1c, HDL
    start = date(2023, 1, 1)
    people = pd.DataFrame({
        # All of Us person ids have 12 digits; matching their size keeps id arithmetic honest.
        "person_id": np.arange(1, n_people + 1, dtype=np.int64) + 200_000_000_000,
        "year_of_birth": rng.integers(1945, 2000, n_people),
        "sex_at_birth_concept_id": rng.choice([1585847, 1585846], n_people),
    })
    fitness = rng.normal(0, 1, n_people)          # latent: drives steps, HbA1c, diabetes
    sleep_var = rng.gamma(2.0, 20.0, n_people)    # latent: night-to-night sleep variability (min)

    act_rows, sleep_rows = [], []
    for i, pid in enumerate(people.person_id):
        worn = rng.random(n_days) < rng.uniform(0.4, 0.95)
        for d in np.flatnonzero(worn):
            day = start + timedelta(days=int(d))
            steps = max(0.0, rng.normal(7000 + 2500 * fitness[i], 1500))
            act_rows.append((pid, day, steps, max(0.0, rng.normal(20 + 10 * fitness[i], 8)),
                             max(0.0, rng.normal(15, 6)), max(0.0, rng.normal(180, 40)),
                             max(0.0, rng.normal(700 - 40 * fitness[i], 90)),
                             rng.normal(2200 + 150 * fitness[i], 200) + 300 * body[i]))
            asleep = max(120.0, rng.normal(420 - 0.4 * sleep_var[i], sleep_var[i]))
            sleep_rows.append((pid, day, True, asleep, asleep + rng.uniform(20, 60), asleep * 0.18,
                               asleep * 0.55, asleep * 0.22, rng.uniform(20, 60)))
            if rng.random() < 0.1:  # a nap: is_main_sleep = false, must be filtered out
                sleep_rows.append((pid, day, False, 40.0, 50.0, 0.0, 30.0, 10.0, 5.0))

    meas_rows, cond_rows, drug_rows = [], [], []
    # EHR visits (for follow-up in the incident design) draw from their own stream too.
    visit_rng = np.random.default_rng(seed + 2)
    visit_rows = [(pid, start + timedelta(days=int(d)))
                  for pid in people.person_id for d in visit_rng.integers(0, n_days + 400, visit_rng.integers(1, 6))]
    # A routine visit in the first half-year for most people (own stream), always before the person's last
    # recorded visit, so everyone's follow-up is unchanged.
    routine_rng = np.random.default_rng(seed + 14)
    last: dict[Any, date] = {}
    for pid, day in visit_rows:
        last[pid] = max(last.get(pid, day), day)
    for pid in people.person_id:
        span = min(180, (last[pid] - start).days)
        if routine_rng.random() < 0.9 and span > 0:
            visit_rows.append((pid, start + timedelta(days=int(routine_rng.integers(0, span)))))
    mid = 1
    for i, pid in enumerate(people.person_id):
        for concept, base, slope in ((3004410, 5.7, -0.35), (3007070, 52.0, 6.0)):
            if rng.random() < 0.8:
                day = start + timedelta(days=int(rng.integers(90, n_days + 60)))
                value = base + slope * fitness[i] + rng.normal(0, 0.4 if concept == 3004410 else 8)
                value += (0.25 if concept == 3004410 else -5.0) * body[i]
                meas_rows.append((mid, pid, concept, day, float(value)))
                mid += 1
        for _ in range(int(body_rng.integers(0, 3))):  # 0-2 BMI readings over the study period
            day = start + timedelta(days=int(body_rng.integers(0, n_days + 60)))
            meas_rows.append((mid, pid, 3038553, day, float(28 + 4.5 * body[i] + body_rng.normal(0, 0.8))))
            mid += 1
        if body_rng.random() < 0.15:
            drug_rows.append((pid, 1503297, start + timedelta(days=int(body_rng.integers(0, n_days + 60)))))
        if rng.random() < 1 / (1 + np.exp(2.2 + 1.1 * fitness[i])):
            cond_rows.append((pid, rng.choice([201826, 4193704]), start + timedelta(days=int(rng.integers(120, n_days + 30)))))
        if rng.random() < 1 / (1 + np.exp(3.0 - 0.03 * sleep_var[i])):
            cond_rows.append((pid, 440383, start + timedelta(days=int(rng.integers(120, n_days + 30)))))

    # The negative-control condition draws from its own stream and depends on nothing.
    control_rng = np.random.default_rng(seed + 3)
    for pid in people.person_id:
        if control_rng.random() < 0.3:
            cond_rows.append((pid, 374375, start + timedelta(days=int(control_rng.integers(0, n_days + 400)))))

    # Most diagnoses are recorded again within months (own stream; every first date is unchanged), so a
    # two-date case rule keeps most cases and leaves out the rest.
    repeat_rng = np.random.default_rng(seed + 13)
    cond_rows += [(pid, concept, day + timedelta(days=int(repeat_rng.integers(14, 180))))
                  for pid, concept, day in list(cond_rows) if repeat_rng.random() < 0.85]

    intraday, stages = _intraday(people.person_id.to_numpy(), act_rows, seed)
    observation = _survey(people.person_id.to_numpy(), start, n_days, seed)
    minute_hr = _minute_heart_rate(people.person_id.to_numpy(), fitness, intraday, seed)
    meas_rows += _repeated_lab(act_rows, start, n_days, mid, seed)
    measurement = pd.DataFrame(meas_rows, columns=["measurement_id", "person_id", "measurement_concept_id",
                                                   "measurement_date", "value_as_number"])
    measurement["unit_concept_id"] = measurement.measurement_concept_id.map(MEASUREMENT_UNITS).astype("int64")
    # Enrolment in a sub-study (its own stream, unrelated to anything else).
    wear_rng = np.random.default_rng(seed + 11)
    wear_study = pd.DataFrame({"person_id": people.person_id[wear_rng.random(n_people) < 1 / 3].to_numpy()})

    frames = {
        "person": people,
        "concept": _concepts(),
        "observation": observation,
        "concept_ancestor": pd.DataFrame(_ANCESTRY, columns=["ancestor_concept_id", "descendant_concept_id"]),
        "measurement": measurement,
        "condition_occurrence": pd.DataFrame(cond_rows, columns=["person_id", "condition_concept_id",
                                                                 "condition_start_date"]),
        "visit_occurrence": pd.DataFrame(visit_rows, columns=["person_id", "visit_start_date"]),
        "drug_exposure": pd.DataFrame(drug_rows, columns=["person_id", "drug_concept_id", "drug_exposure_start_date"]),
        "activity_summary": pd.DataFrame(act_rows, columns=["person_id", "date", "steps", "very_active_minutes",
                                                            "fairly_active_minutes", "lightly_active_minutes",
                                                            "sedentary_minutes", "calories_out"]),
        "sleep_daily_summary": pd.DataFrame(sleep_rows, columns=["person_id", "sleep_date", "is_main_sleep",
                                                                 "minute_asleep", "minute_in_bed", "minute_deep",
                                                                 "minute_light", "minute_rem", "minute_awake"]),
        "steps_intraday": intraday,
        "sleep_level": stages,
        "heart_rate_summary": _zones(act_rows, seed),
        "heart_rate_minute_level": minute_hr,
        "wear_study": wear_study,
    }
    for name, frame in frames.items():
        for column in frame.columns:
            if column.endswith("date") or column == "date":
                frame[column] = pd.to_datetime(frame[column]).dt.date
        con.register(f"_{name}_df", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}_df")
        con.unregister(f"_{name}_df")


def _concepts() -> pd.DataFrame:
    frame = pd.DataFrame(CONCEPTS, columns=["concept_id", "concept_name", "standard_concept", "domain_id"])
    frame["vocabulary_id"] = [CODES.get(c, ("None", ""))[0] for c in frame.concept_id]
    frame["concept_code"] = [CODES.get(c, ("None", str(c)))[1] for c in frame.concept_id]
    return frame


def _survey(person_ids: np.ndarray, start: date, n_days: int, seed: int) -> pd.DataFrame:
    """One survey question (ever smoked 100 cigarettes) answered by most people, some after their index
    date, with skips; and a second smoking question, so a vague question text matches two. Own stream."""
    rng = np.random.default_rng(seed + 8)
    rows = []
    for k, pid in enumerate(person_ids):
        if rng.random() < 0.85:
            answer = rng.choice([1585858, 1585859, 903096], p=[0.35, 0.58, 0.07])
            rows.append((k * 2 + 1, pid, 1585857, int(answer), start + timedelta(days=int(rng.integers(-200, n_days)))))
        if rng.random() < 0.3:
            rows.append((k * 2 + 2, pid, 1586000, 903096, start + timedelta(days=int(rng.integers(0, n_days)))))
    return pd.DataFrame(rows, columns=["observation_id", "person_id", "observation_source_concept_id",
                                       "value_source_concept_id", "observation_date"])


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    return "'" + str(value).replace("'", "''") + "'"


def inline_params(sql: str, params: list[tuple[str, str, Any]]) -> str:
    """Replace each ``@name`` with a SQL literal (longest names first, so @a does not eat @ab)."""
    for name, _typ, value in sorted(params, key=lambda p: -len(p[0])):
        sql = re.sub(rf"@{re.escape(name)}\b", _literal(value), sql)
    return sql


def bigquery_name_clashes(sql: str) -> list[str]:
    """Names BigQuery reads differently from DuckDB: a CTE or subquery that outputs a column with its own
    name. BigQuery resolves that bare name to the whole row (a STRUCT), DuckDB to the column, so such a
    query passes here and fails on the real CDR."""
    import sqlglot
    from sqlglot import exp

    clashes = []
    for tree in sqlglot.parse(sql, read="bigquery"):
        for node in tree.find_all(exp.CTE, exp.Subquery):
            name = (node.alias or "").lower()
            body = node.this
            outputs = {str(c).lower() for c in getattr(body, "named_selects", [])}
            if name and name in outputs:
                clashes.append(name)
    return sorted(set(clashes))


def duckdb_query(con: Any) -> Callable[[Any, str, list], pd.DataFrame]:
    """A drop-in for ``codas.aou.cohort._query`` that runs against the synthetic DuckDB CDR. A query that
    BigQuery would read differently (``bigquery_name_clashes``) is refused here rather than on the CDR."""
    import sqlglot

    def run(_client: Any, sql: str, params: list[tuple[str, str, Any]]) -> pd.DataFrame:
        clashes = bigquery_name_clashes(sql)
        if clashes:
            raise ValueError(f"CTE or subquery named like one of its own columns ({', '.join(clashes)}): "
                             "BigQuery reads the bare name as the whole row; rename the column")
        sql = inline_params(sql, params)
        sql = re.sub(r"`[^`]+\.INFORMATION_SCHEMA\.COLUMNS`", "__info_columns", sql)
        duck = sqlglot.transpile(sql, read="bigquery", write="duckdb")[0]
        return con.execute(duck.replace("__info_columns", "information_schema.columns")).df()

    return run


def intraday_latents(n_people: int, seed: int = 0) -> dict[str, np.ndarray]:
    """The per-person latents behind the synthetic intraday tables (for tests that check a phenotype
    tracks what generated it): rhythm strength, activity peak hour, sleep-onset night-to-night SD."""
    rng = np.random.default_rng(seed + 4)
    return {"rhythm": rng.uniform(0.5, 3.0, n_people), "peak_hour": rng.uniform(12.0, 18.0, n_people),
            "onset_sd": rng.gamma(2.0, 0.4, n_people)}


def _intraday(person_ids: np.ndarray, act_rows: list[tuple], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minute steps and sleep-stage segments on the activity days, from their own random streams (so the
    other synthetic tables are unchanged). Each person's hourly profile is a von Mises-shaped curve of
    their daily steps with their own rhythm strength and peak hour; quiet hours carry no minutes."""
    latents = intraday_latents(len(person_ids), seed)
    index = {int(pid): i for i, pid in enumerate(person_ids)}
    rows = pd.DataFrame([(r[0], r[1], r[2]) for r in act_rows], columns=["person_id", "day", "steps"])
    who = rows.person_id.map(index).to_numpy()
    hours = np.arange(24)
    shape = np.exp(latents["rhythm"][who, None]
                   * np.cos(2 * np.pi * (hours[None, :] - latents["peak_hour"][who, None]) / 24))
    shape /= shape.sum(axis=1, keepdims=True)
    hourly = rows.steps.to_numpy()[:, None] * shape
    hourly[shape < 0.3 / 24] = 0.0  # night hours: no recorded minutes
    rng = np.random.default_rng(seed + 5)
    day_idx, hour = np.nonzero(hourly > 1.0)
    minutes = rng.integers(0, 59, size=(len(day_idx), 2))
    minutes[:, 1] = np.minimum(minutes[:, 0] + rng.integers(1, 30, len(day_idx)), 59)
    split = rng.uniform(0.3, 0.7, len(day_idx))
    base = pd.to_datetime(rows.day.to_numpy()[day_idx]) + pd.to_timedelta(hour, unit="h")
    steps = hourly[day_idx, hour]
    intraday = pd.DataFrame({
        "person_id": np.repeat(rows.person_id.to_numpy()[day_idx], 2),
        "datetime": np.column_stack([base + pd.to_timedelta(minutes[:, 0], unit="m"),
                                     base + pd.to_timedelta(minutes[:, 1], unit="m")]).ravel(),
        "steps": np.column_stack([steps * split, steps * (1 - split)]).ravel().round(),
    })
    intraday = intraday[intraday.steps > 0].reset_index(drop=True)

    # Sleep: the night ending on each activity day, eight stage segments from a person-specific onset.
    n = len(rows)
    onset = (pd.to_datetime(rows.day.to_numpy()) - pd.Timedelta(hours=1)
             + pd.to_timedelta(rng.normal(0, 1, n) * latents["onset_sd"][who], unit="h"))
    levels = rng.choice(np.array(["light", "deep", "rem", "wake"]), size=(n, 8), p=[0.5, 0.2, 0.2, 0.1])
    durations = np.round(rng.uniform(15, 75, size=(n, 8)))
    starts = onset.to_numpy()[:, None] + (np.cumsum(durations, axis=1) - durations).astype("timedelta64[m]")
    stages = pd.DataFrame({
        "person_id": np.repeat(rows.person_id.to_numpy(), 8),
        "sleep_date": np.repeat(rows.day.to_numpy(), 8),
        "is_main_sleep": "true",
        "level": levels.ravel(),
        "start_datetime": starts.ravel(),
        "duration_in_min": durations.ravel(),
    })
    nap = rng.random(n) < 0.05  # a daytime nap, is_main_sleep = false, must be filtered out
    naps = pd.DataFrame({"person_id": rows.person_id.to_numpy()[nap], "sleep_date": rows.day.to_numpy()[nap],
                         "is_main_sleep": "false", "level": "light",
                         "start_datetime": pd.to_datetime(rows.day.to_numpy()[nap]) + pd.Timedelta(hours=14),
                         "duration_in_min": 40.0})
    return intraday, pd.concat([stages, naps], ignore_index=True)


HR_PEOPLE = 30  # minute heart rate for the first people only, to keep the synthetic database small


def heart_rate_latents(fitness: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-person resting heart rate (lower with fitness) and heart-rate rise per step/min, behind the
    synthetic minute heart-rate table (own stream)."""
    rng = np.random.default_rng(seed + 9)
    n = min(HR_PEOPLE, len(fitness))
    return {"rest": 68 - 5 * fitness[:n] + rng.normal(0, 2, n), "per_step": rng.uniform(0.05, 0.25, n)}


def _minute_heart_rate(person_ids: np.ndarray, fitness: np.ndarray, intraday: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Minute heart rate on the days a person has minute steps: two consecutive night hours (01:00-02:59),
    a reading every 10 minutes from 10:00 to 19:59, and one at every minute with steps; on about 85% of the
    days (their own stream) also a reading at each remaining minute from 08:00 to 17:59, so those days
    have more than 600 minutes of data and the rest about 200."""
    lat = heart_rate_latents(fitness, seed)
    rng = np.random.default_rng(seed + 10)
    who = {int(pid): k for k, pid in enumerate(person_ids[:HR_PEOPLE])}
    steps = intraday[intraday.person_id.isin(list(who))].copy()
    steps["day"] = steps.datetime.dt.normalize()
    days = steps[["person_id", "day"]].drop_duplicates()
    k = days.person_id.map(who).to_numpy()
    night = np.arange(60, 180)                       # minutes after midnight
    daytime = np.arange(600, 1200, 10)
    frames = []
    for minutes, offset in ((night, -6.0), (daytime, 8.0)):
        reps = len(minutes)
        frames.append(pd.DataFrame({
            "person_id": np.repeat(days.person_id.to_numpy(), reps),
            "datetime": np.repeat(days.day.to_numpy(), reps) + pd.to_timedelta(np.tile(minutes, len(days)), unit="m"),
            "heart_rate_value": np.round(np.repeat(lat["rest"][k], reps) + offset
                                         + rng.normal(0, 3, len(days) * reps))}))
    ks = steps.person_id.map(who).to_numpy()
    frames.append(pd.DataFrame({"person_id": steps.person_id.to_numpy(), "datetime": steps.datetime.to_numpy(),
                                "heart_rate_value": np.round(lat["rest"][ks] + 10 + lat["per_step"][ks] * steps.steps
                                                             + rng.normal(0, 3, len(steps)))}))
    fill_rng = np.random.default_rng(seed + 12)
    dense = days[fill_rng.random(len(days)) < 0.85]
    worn = np.arange(480, 1080)
    kd = dense.person_id.map(who).to_numpy()
    frames.append(pd.DataFrame({  # appended last: where a minute already has a reading, that one is kept
        "person_id": np.repeat(dense.person_id.to_numpy(), len(worn)),
        "datetime": np.repeat(dense.day.to_numpy(), len(worn)) + pd.to_timedelta(np.tile(worn, len(dense)), unit="m"),
        "heart_rate_value": np.round(np.repeat(lat["rest"][kd], len(worn)) + 8.0
                                     + fill_rng.normal(0, 3, len(dense) * len(worn)))}))
    hr = pd.concat(frames, ignore_index=True).drop_duplicates(["person_id", "datetime"])
    return hr.sort_values(["person_id", "datetime"]).reset_index(drop=True)


def _zones(act_rows: list[tuple], seed: int) -> pd.DataFrame:
    """Daily minutes per heart-rate zone on the activity days (own random stream), growing with the
    day's very active minutes."""
    rng = np.random.default_rng(seed + 6)
    rows = pd.DataFrame([(r[0], r[1], r[3]) for r in act_rows], columns=["person_id", "date", "very_active"])
    zones = {"Out of Range": 1300.0, "Fat Burn": 90.0, "Cardio": 15.0, "Peak": 3.0}
    frames = []
    for zone, base in zones.items():
        scale = 1.0 if zone == "Out of Range" else 1 + rows.very_active.to_numpy() / 30
        frames.append(pd.DataFrame({"person_id": rows.person_id, "date": rows.date, "zone_name": zone,
                                    "minute_in_zone": np.round(rng.gamma(4.0, base / 4.0, len(rows)) * scale)}))
    return pd.concat(frames, ignore_index=True)


def _repeated_lab(act_rows: list[tuple], start: date, n_days: int, first_id: int, seed: int) -> list[tuple]:
    """Four values per person between day 100 and the end of wear (own random stream): 50 + 4 per 1,000
    steps of the 90-day mean before the value above the person's own mean, + 2 per 1,000 steps of the
    person's mean, + noise."""
    rng = np.random.default_rng(seed + 7)
    act = pd.DataFrame([(r[0], r[1], r[2]) for r in act_rows], columns=["person_id", "day", "steps"])
    act["d"] = (pd.to_datetime(act.day) - pd.Timestamp(start)).dt.days
    rows, mid = [], first_id
    for pid, own in act.groupby("person_id"):
        days, steps = own.d.to_numpy(), own.steps.to_numpy()
        person_mean = steps.mean()
        intercept = rng.normal(0, 2)
        for d in np.sort(rng.integers(100, n_days, 4)):
            window = (days >= d - 90) & (days < d)
            if window.sum() < 10:
                continue
            value = (50 + intercept + 4 * (steps[window].mean() - person_mean) / 1000 + 2 * person_mean / 1000
                     + rng.normal(0, 0.5))
            rows.append((mid, int(pid), REPEATED_LAB, start + timedelta(days=int(d)), float(value)))
            mid += 1
    return rows
