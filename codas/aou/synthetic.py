"""A synthetic, All of Us–shaped CDR in DuckDB, for testing and demos outside the Workbench.

No row here describes a real person. The tables carry the OMOP and Fitbit columns the extractor
reads, with planted associations (more steps -> lower HbA1c and less type 2 diabetes; shorter,
more variable sleep -> more depression) so an end-to-end run has something to find.

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
    (1585847, "Female", None, "Gender"),
    (1585846, "Male", None, "Gender"),
]
_ANCESTRY = [(201826, 201826), (201826, 4193704), (3004410, 3004410), (3007070, 3007070),
             (440383, 440383)]


def build_synthetic_cdr(con: Any, n_people: int = 600, n_days: int = 240, seed: int = 0) -> None:
    """Create person, concept, concept_ancestor, measurement, condition_occurrence,
    activity_summary and sleep_daily_summary in the DuckDB connection ``con``."""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    people = pd.DataFrame({
        "person_id": np.arange(1, n_people + 1),
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
                             rng.normal(2200 + 150 * fitness[i], 200)))
            asleep = max(120.0, rng.normal(420 - 0.4 * sleep_var[i], sleep_var[i]))
            sleep_rows.append((pid, day, True, asleep, asleep + rng.uniform(20, 60), asleep * 0.18,
                               asleep * 0.55, asleep * 0.22, rng.uniform(20, 60)))
            if rng.random() < 0.1:  # a nap: is_main_sleep = false, must be filtered out
                sleep_rows.append((pid, day, False, 40.0, 50.0, 0.0, 30.0, 10.0, 5.0))

    meas_rows, cond_rows = [], []
    mid = 1
    for i, pid in enumerate(people.person_id):
        for concept, base, slope in ((3004410, 5.7, -0.35), (3007070, 52.0, 6.0)):
            if rng.random() < 0.8:
                day = start + timedelta(days=int(rng.integers(90, n_days + 60)))
                meas_rows.append((mid, pid, concept, day, float(base + slope * fitness[i] + rng.normal(0, 0.4 if concept == 3004410 else 8))))
                mid += 1
        if rng.random() < 1 / (1 + np.exp(2.2 + 1.1 * fitness[i])):
            cond_rows.append((pid, rng.choice([201826, 4193704]), start + timedelta(days=int(rng.integers(120, n_days + 30)))))
        if rng.random() < 1 / (1 + np.exp(3.0 - 0.03 * sleep_var[i])):
            cond_rows.append((pid, 440383, start + timedelta(days=int(rng.integers(120, n_days + 30)))))

    frames = {
        "person": people,
        "concept": pd.DataFrame(CONCEPTS, columns=["concept_id", "concept_name", "standard_concept", "domain_id"]),
        "concept_ancestor": pd.DataFrame(_ANCESTRY, columns=["ancestor_concept_id", "descendant_concept_id"]),
        "measurement": pd.DataFrame(meas_rows, columns=["measurement_id", "person_id", "measurement_concept_id",
                                                        "measurement_date", "value_as_number"]),
        "condition_occurrence": pd.DataFrame(cond_rows, columns=["person_id", "condition_concept_id",
                                                                 "condition_start_date"]),
        "activity_summary": pd.DataFrame(act_rows, columns=["person_id", "date", "steps", "very_active_minutes",
                                                            "fairly_active_minutes", "lightly_active_minutes",
                                                            "sedentary_minutes", "calories_out"]),
        "sleep_daily_summary": pd.DataFrame(sleep_rows, columns=["person_id", "sleep_date", "is_main_sleep",
                                                                 "minute_asleep", "minute_in_bed", "minute_deep",
                                                                 "minute_light", "minute_rem", "minute_awake"]),
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
