"""A synthetic, All of Us–shaped CDR in DuckDB, for testing and demos outside the Workbench.

No row here describes a real person. The tables carry the OMOP and Fitbit columns the extractor
reads, with planted associations (more steps -> lower HbA1c and less type 2 diabetes; shorter,
more variable sleep -> more depression) so an end-to-end run has something to find. A separate latent
body size raises calories burned, BMI and HbA1c and lowers HDL, so calories look like a biomarker
until BMI is adjusted for. One negative-control condition (impacted cerumen) is diagnosed at random,
unrelated to anything else, so a confirmation step should find no association with it.

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
]
_ANCESTRY = [(201826, 201826), (201826, 4193704), (3004410, 3004410), (3007070, 3007070),
             (440383, 440383), (3038553, 3038553), (1503297, 1503297), (374375, 374375), (380706, 380706),
             (140641, 140641)]


def build_synthetic_cdr(con: Any, n_people: int = 600, n_days: int = 240, seed: int = 0) -> None:
    """Create person, concept, concept_ancestor, measurement, condition_occurrence, drug_exposure,
    visit_occurrence, activity_summary, sleep_daily_summary, steps_intraday, sleep_level and
    heart_rate_summary in the DuckDB connection ``con``."""
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

    intraday, stages = _intraday(people.person_id.to_numpy(), act_rows, seed)

    frames = {
        "person": people,
        "concept": pd.DataFrame(CONCEPTS, columns=["concept_id", "concept_name", "standard_concept", "domain_id"]),
        "concept_ancestor": pd.DataFrame(_ANCESTRY, columns=["ancestor_concept_id", "descendant_concept_id"]),
        "measurement": pd.DataFrame(meas_rows, columns=["measurement_id", "person_id", "measurement_concept_id",
                                                        "measurement_date", "value_as_number"]),
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
    }
    for name, frame in frames.items():
        for column in frame.columns:
            if column.endswith("date") or column == "date":
                frame[column] = pd.to_datetime(frame[column]).dt.date
        con.register(f"_{name}_df", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}_df")
        con.unregister(f"_{name}_df")


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


def duckdb_query(con: Any) -> Callable[[Any, str, list], pd.DataFrame]:
    """A drop-in for ``codas.aou.cohort._query`` that runs against the synthetic DuckDB CDR."""
    import sqlglot

    def run(_client: Any, sql: str, params: list[tuple[str, str, Any]]) -> pd.DataFrame:
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
