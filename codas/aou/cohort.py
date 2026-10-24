"""Build a one-row-per-participant CoDaS analysis table from the All of Us CDR (OMOP, BigQuery).

A ``CohortSpec`` names an outcome (a lab measurement or a condition, by OMOP concept id) and the
wearable tables to summarise. For each participant the extractor fixes an index date, summarises
every wearable column over the ``window_days`` before it (mean, SD, and the number of days worn),
adds age and sex at birth and any covariates the spec names, and writes the table to the workspace
disk under ``.codas_runs/aou/``.

Index date:

* ``measurement`` outcome: the participant's first qualifying measurement that has wearable data in
  the window before it. The outcome is ``value_as_number``.
* ``condition`` outcome: the first diagnosis date for cases and the last wearable date for
  everyone else; the outcome is 1 for cases and 0 otherwise. Cases and controls get index dates by
  different rules, so treat associations as hypothesis-generating.

With ``design = "incident"`` the cohort is prospective instead: everyone's index date is the end of a
baseline window that starts at their first valid wear day (``window_days`` long), the features
summarise that baseline window, and the outcome is observed after ``index + lag_days``. For a
condition, anyone diagnosed before ``index + lag_days`` is excluded, the target is a new diagnosis
(1/0) and ``followup_days`` runs from ``index + lag_days`` to the diagnosis or the last EHR visit; for
a measurement, the target is the first value at or after ``index + lag_days`` (within
``horizon_days`` if set). Cases and non-cases share one index rule.

``holdout_fraction`` seals a deterministic share of participants (by a hash of the person id) into a
separate replication table that the agents never see.

Negative-control outcomes (optional, ``CohortSpec.negative_controls``, incident design only) are
conditions chosen in advance as ones the exposures should not cause. Each adds two columns to the
sealed table only, ``nc_<name>`` (a new diagnosis during follow-up, 1/0; empty for anyone diagnosed
before follow-up starts) and ``nc_<name>_days`` (follow-up to that diagnosis or the last EHR visit).
The confirmation step tests every pre-registered candidate against them: an association with an
outcome the exposure cannot cause points to confounding or to differences in health-care contact.

Covariates (optional, ``CohortSpec.covariates``) are clinical context for confounder adjustment, not
candidate biomarkers: a measurement covariate is the most recent value in the ``lookback_days`` up
to and including the index date; a condition or drug covariate is 1 if a record starts in the
lookback before the index date, else 0. Which covariates to extract is configuration; whether to
adjust for one is decided during the analysis.

Quality rules (all optional, all in the spec, all listed in words by ``quality_rules``):

* ``WearableSource.value_bounds`` drops a day whose value lies outside plausible bounds (a main-sleep night
  under 3 h or over 16 h) from that source's summaries;
* ``OutcomeSpec.min_occurrences`` makes a condition case need diagnoses on that many distinct dates (2 is a
  common guard against rule-out codes); people with fewer are left out, not counted as controls, and any
  diagnosis before follow-up still excludes the person;
* ``OutcomeSpec.unit_concept_ids`` keeps only lab values recorded in those units. Whatever the setting, the
  units of the values used are summarised from the discovery share (``unit_mix``), so a lab recorded in two
  units is visible before anyone analyses it; the same summary (``extraction_qc``, ``<name>.qc.json``) gives
  each daily feature's split-half reliability: how well its first and second half of the window agree;
* ``CohortSpec.ehr_lookback_days`` keeps only people with an EHR visit in that many days before the index
  date, so a missing diagnosis means no diagnosis rather than no records;
* the phenotype sources' own rules (heart-rate wear days, night length, contributing days; see
  ``codas.aou.phenotypes``).

Covariate kinds beyond measurements, conditions, drugs and survey answers: ``visits`` counts the distinct
visit dates in the lookback before the index date (how much health care a person uses, which drives how
much gets diagnosed), and ``in_table`` is 1 for people listed in a named table (e.g. a sub-study's
enrolment table), else 0.

The table is participant-level data. It stays inside the Workbench: it carries no person_id or
dates, it is git-ignored, and CoDaS only ever shows the model aggregates computed from it.

Under the incident design the intraday phenotypes depend on the person only (the index date is fixed
by the first wear day), so ``extract_cohort`` computes them once per CDR and phenotype definition,
keeps them in ``.codas_runs/aou/sealed/phenotype_cache/`` (participant-level, keyed by person for the merge,
never read by the agents) and merges them into each outcome's table, dropping the person id. The
minute-level tables are terabytes; reading them once per run instead of once per outcome is most of
the BigQuery cost. ``CODAS_PHENOTYPE_CACHE=0`` computes them inline instead. Every BigQuery job's
bytes billed are recorded in ``QUERY_LOG`` and large ones printed to the job log.

The SQL builder is pure (no BigQuery import) so it can be unit-tested anywhere; ``extract_cohort``
imports ``google.cloud.bigquery`` only when it runs.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codas.aou.phenotypes import (PhenotypeSource, bounds as range_ok, minute_activity_ctes, minute_heart_rate_ctes,
                                  quality_rules as phenotype_rules, range_text, resolve, select_columns,
                                  sleep_stage_ctes, whole)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / ".codas_runs" / "aou"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CDR = re.compile(r"^[A-Za-z0-9_\-]+\.[A-Za-z0-9_]+$")


def _ident(value: str, what: str) -> str:
    if not _IDENT.match(value or ""):
        raise ValueError(f"Invalid {what} {value!r}: use letters, digits and underscores only.")
    return value


def resolve_cdr(explicit: str | None = None) -> str:
    """The CDR dataset as ``project.dataset``: the argument, else CODAS_AOU_CDR, else WORKSPACE_CDR."""
    cdr = (explicit or os.getenv("CODAS_AOU_CDR") or os.getenv("WORKSPACE_CDR") or "").strip()
    if not cdr:
        raise ValueError("No CDR dataset given. Pass --cdr project.dataset or set CODAS_AOU_CDR "
                         "(the Workbench may also expose it as WORKSPACE_CDR).")
    if not _CDR.match(cdr):
        raise ValueError(f"CDR {cdr!r} is not of the form project.dataset.")
    return cdr


@dataclass
class WearableSource:
    """One wearable table to summarise per participant over the pre-index window."""

    table: str
    date_column: str
    columns: list[str]
    prefix: str
    # Optional equality filter, e.g. is_main_sleep = true, or zone_name = 'Cardio'.
    filter_column: str | None = None
    filter_value: Any = None
    # An optional source is dropped (columns it lacks are dropped) when the CDR does not have it,
    # instead of stopping the run. The primary (first) source cannot be optional.
    optional: bool = False
    # Plausible values, {column: [low, high]} (either may be null): a day (row) with the column outside its
    # range or missing is left out of the source's summaries, e.g. {"minute_asleep": [180, 960]}.
    value_bounds: dict[str, list[float | None]] | None = None

    def validate(self) -> None:
        _ident(self.table, "table")
        _ident(self.date_column, "date column")
        _ident(self.prefix, "prefix")
        if not self.columns:
            raise ValueError(f"Wearable source {self.table!r} lists no columns.")
        for column in self.columns:
            _ident(column, "column")
        if self.filter_column is not None:
            _ident(self.filter_column, "filter column")
            if self.filter_value is None:
                raise ValueError(f"filter_column {self.filter_column!r} needs a filter_value.")
        for column, limits in (self.value_bounds or {}).items():
            _ident(column, "value_bounds column")
            if not range_ok(limits):
                raise ValueError(f"value_bounds[{column!r}] must be [low, high] (numbers or null), low <= high.")


@dataclass
class OutcomeSpec:
    kind: str  # "measurement" or "condition"
    concept_ids: list[int]
    name: str = "outcome"
    include_descendants: bool = True
    # Plausibility bounds for a measurement value (drop unit errors and sentinels).
    value_min: float | None = None
    value_max: float | None = None
    # Text the concept name must contain, checked against the CDR concept table before a run.
    expect_name: str = ""
    # Public codes ("SNOMED:44054006", "LOINC:4548-4") resolved to this CDR's concept ids before a run.
    codes: list[str] = field(default_factory=list)
    # Condition: a case needs diagnoses on at least this many distinct dates; people with fewer are left out.
    min_occurrences: int = 1
    # Measurement: keep only values recorded in these OMOP unit concepts (empty keeps every unit).
    unit_concept_ids: list[int] = field(default_factory=list)

    def validate(self) -> None:
        if self.kind not in {"measurement", "condition"}:
            raise ValueError(f"outcome.kind must be 'measurement' or 'condition', not {self.kind!r}.")
        _check_concepts(self.concept_ids, self.codes, f"outcome {self.name!r}")
        _ident(self.name, "outcome name")
        if not whole(self.min_occurrences, 1, 100) or (self.min_occurrences > 1 and self.kind != "condition"):
            raise ValueError(f"outcome {self.name!r}: min_occurrences is a whole number >= 1, above 1 for "
                             "conditions only.")
        if self.unit_concept_ids and (self.kind != "measurement"
                                      or not all(whole(u, 0, 10**12) for u in self.unit_concept_ids)):
            raise ValueError(f"outcome {self.name!r}: unit_concept_ids are OMOP unit concept ids, for "
                             "measurements only.")


_COVARIATE_TABLES = {
    # kind: (table, concept column, date column)
    "measurement": ("measurement", "measurement_concept_id", "measurement_date"),
    "condition": ("condition_occurrence", "condition_concept_id", "condition_start_date"),
    "drug": ("drug_exposure", "drug_concept_id", "drug_exposure_start_date"),
    # A survey answer: the question is the source concept, the answer the value's source concept.
    "survey": ("observation", "observation_source_concept_id", "observation_date"),
}
# Covariates that name no concept: distinct visit dates in the lookback, and membership of a named table.
_CONCEPT_FREE = ("visits", "in_table")
SURVEY_MISSING_ANSWERS = "skip|prefer not|don.?t know|not sure"


def _check_concepts(ids: list[int], codes: list[str], what: str, question: str = "") -> None:
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in ids):
        raise ValueError(f"{what}: concept_ids must be integer OMOP concept ids.")
    if any(not isinstance(c, str) or c.count(":") != 1 or not all(c.split(":")) for c in codes):
        raise ValueError(f"{what}: codes must look like 'VOCABULARY:CODE' (e.g. 'SNOMED:44054006').")
    if not ids and not codes and not question:
        raise ValueError(f"{what}: give concept_ids, codes or (for a survey) question_name.")


@dataclass
class CovariateSpec:
    """A clinical covariate extracted alongside the outcome, for adjustment rather than discovery."""

    name: str
    kind: str  # "measurement", "condition", "drug", "survey", "visits" or "in_table"
    concept_ids: list[int] = field(default_factory=list)
    include_descendants: bool = True
    # Days before the index date to look back; None looks back without limit.
    lookback_days: int | None = 365
    value_min: float | None = None
    value_max: float | None = None
    # Text the concept name must contain, checked against the CDR concept table before a run.
    expect_name: str = ""
    codes: list[str] = field(default_factory=list)
    # Survey covariates: text of the question (resolved to one question concept in this CDR), and
    # regular expressions over the answer's name: a match of ``positive_answers`` is 1, of
    # ``missing_answers`` (skips, refusals) missing, anything else 0. ``any_time`` takes the answer nearest
    # the index date even if given after it (for lifetime traits); otherwise the latest one before it.
    question_name: str = ""
    positive_answers: str = ""
    missing_answers: str = SURVEY_MISSING_ANSWERS
    any_time: bool = False
    # in_table: the table whose person_id list defines the flag.
    table: str = ""

    def validate(self) -> None:
        _ident(self.name, "covariate name")
        kinds = sorted([*_COVARIATE_TABLES, *_CONCEPT_FREE])
        if self.kind not in kinds:
            raise ValueError(f"covariate {self.name!r}: kind must be one of {kinds}, not {self.kind!r}.")
        if self.kind in _CONCEPT_FREE:
            if self.concept_ids or self.codes:
                raise ValueError(f"covariate {self.name!r}: a {self.kind} covariate names no concepts.")
            if (self.kind == "in_table") != bool(self.table):
                raise ValueError(f"covariate {self.name!r}: table is required for in_table and only for it.")
            if self.table:
                _ident(self.table, f"covariate {self.name!r} table")
        else:
            _check_concepts(self.concept_ids, self.codes, f"covariate {self.name!r}",
                            self.question_name if self.kind == "survey" else "")
        if self.kind == "survey" and not self.positive_answers:
            raise ValueError(f"covariate {self.name!r}: a survey covariate needs positive_answers.")
        if self.lookback_days is not None and self.lookback_days < 0:
            raise ValueError(f"covariate {self.name!r}: lookback_days must be non-negative or null.")
        if self.kind != "measurement" and (self.value_min is not None or self.value_max is not None):
            raise ValueError(f"covariate {self.name!r}: value bounds apply to measurement covariates only.")


@dataclass
class CohortSpec:
    name: str
    outcome: OutcomeSpec
    wearables: list[WearableSource]
    window_days: int = 180
    min_wear_days: int = 30
    include_demographics: bool = True
    covariates: list[CovariateSpec] = field(default_factory=list)
    notes: str = ""
    design: str = "window"  # "window": outcome-anchored window; "incident": prospective, shared index rule
    lag_days: int = 0  # incident: outcomes before index + lag exclude the person (reverse-causation guard)
    horizon_days: int | None = None  # incident measurement: first value within this many days after the lag
    # Optional valid-day rule for the primary wearable, e.g. {"column": "steps", "min": 100, "max": 45000}.
    valid_day: dict[str, Any] | None = None
    holdout_fraction: float = 0.0  # share of participants sealed for replication (0 = no split)
    split_salt: int = 0
    # Conditions the exposures should not cause, tested on the sealed table only (incident design).
    negative_controls: list[OutcomeSpec] = field(default_factory=list)
    # Columns of the table with a known association with the outcome, tested on the sealed table the
    # same way as the candidates to show the confirmation can detect a real effect.
    positive_controls: list[str] = field(default_factory=list)
    # Intraday digital phenotypes from minute- or segment-level tables (codas/aou/phenotypes.py).
    phenotypes: list[PhenotypeSource] = field(default_factory=list)
    # Repeated-measures panel for a within-person model (measurement outcomes): every qualifying value
    # after the first period, paired with the wearable summary of the period before it, e.g.
    # {"period_days": 90, "min_days": 30}.
    panel: dict[str, Any] | None = None
    # Keep only people with an EHR visit in this many days before the index date (None: no such rule).
    ehr_lookback_days: int | None = None
    # The day-by-day baseline new phenotypes are defined on (codas/aou/daystore.py), e.g.
    # {"from": ["hrmin", "sleepseg"]}: the daily sources always, and the minute-level sources named here.
    day_store: dict[str, Any] | None = None

    def validate(self) -> None:
        _ident(self.name, "cohort name")
        self.outcome.validate()
        if not self.wearables:
            raise ValueError("A cohort needs at least one wearable source.")
        if self.wearables[0].optional:
            raise ValueError("The primary (first) wearable source cannot be optional.")
        prefixes = set()
        for source in [*self.wearables, *self.phenotypes]:
            source.validate()
            if source.prefix in prefixes:
                raise ValueError(f"Duplicate wearable prefix {source.prefix!r}.")
            prefixes.add(source.prefix)
        if self.window_days < 1 or self.min_wear_days < 1:
            raise ValueError("window_days and min_wear_days must be positive.")
        if self.min_wear_days > self.window_days:
            raise ValueError("min_wear_days cannot exceed window_days.")
        if self.design not in {"window", "incident"}:
            raise ValueError(f"design must be 'window' or 'incident', not {self.design!r}.")
        if self.lag_days < 0 or (self.horizon_days is not None and self.horizon_days < 1):
            raise ValueError("lag_days must be non-negative and horizon_days positive.")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1).")
        if not 0 <= self.split_salt <= MAX_SPLIT_SALT:
            raise ValueError(f"split_salt must be in [0, {MAX_SPLIT_SALT}].")
        if self.valid_day is not None:
            _ident(str(self.valid_day.get("column", "")), "valid_day column")
            if self.valid_day.get("column") not in self.wearables[0].columns:
                raise ValueError("valid_day.column must be one of the primary wearable's columns.")
        taken = set(_base_columns(self))
        for covariate in self.covariates:
            covariate.validate()
            if covariate.name in taken:
                raise ValueError(f"Covariate name {covariate.name!r} clashes with another column.")
            taken.add(covariate.name)
            shared = set(covariate.concept_ids) & set(self.outcome.concept_ids)
            # In the incident design a covariate is fixed at the index date, before the outcome is
            # observed, so a baseline value of the outcome's own measure is a legitimate covariate.
            if shared and self.design == "window":
                raise ValueError(f"Covariate {covariate.name!r} shares concept ids {sorted(shared)} with the "
                                 "outcome; a covariate cannot be the outcome itself.")
        if self.negative_controls and self.design != "incident":
            raise ValueError("negative_controls need the incident design (follow-up after a shared index date).")
        for control in self.negative_controls:
            control.validate()
            if control.kind != "condition":
                raise ValueError(f"negative control {control.name!r}: kind must be 'condition'.")
            if set(control.concept_ids) & set(self.outcome.concept_ids):
                raise ValueError(f"negative control {control.name!r} shares concept ids with the outcome.")
        nc_columns = negative_control_columns(self)
        if len(set(nc_columns)) != len(nc_columns) or set(nc_columns) & taken:
            raise ValueError("Negative-control names must be unique and must not clash with other columns.")
        for column in self.positive_controls:
            if column not in taken:
                raise ValueError(f"positive control {column!r} is not a column of the table.")
        if self.panel is not None:
            period, least = int(self.panel.get("period_days", 0)), int(self.panel.get("min_days", 1))
            if period < 7 or not 1 <= least <= period:
                raise ValueError("panel needs period_days >= 7 and 1 <= min_days <= period_days.")
        if self.ehr_lookback_days is not None and not whole(self.ehr_lookback_days, 1, 100_000):
            raise ValueError("ehr_lookback_days must be a whole number of days, at least 1 (or null).")
        if self.day_store is not None:  # its sources are checked when a spec file is read (codas.aou.daystore)
            named = self.day_store.get("from") if isinstance(self.day_store, dict) else None
            if not isinstance(named, list) or not all(isinstance(n, str) and _IDENT.match(n) for n in named):
                raise ValueError('day_store must be {"from": [phenotype source prefixes]}.')

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CohortSpec":
        spec = cls(
            name=data["name"],
            outcome=OutcomeSpec(**data["outcome"]),
            wearables=[WearableSource(**w) for w in data["wearables"]],
            window_days=int(data.get("window_days", 180)),
            min_wear_days=int(data.get("min_wear_days", 30)),
            include_demographics=bool(data.get("include_demographics", True)),
            covariates=[CovariateSpec(**c) for c in data.get("covariates", [])],
            notes=str(data.get("notes", "")),
            design=str(data.get("design", "window")),
            lag_days=int(data.get("lag_days", 0)),
            horizon_days=None if data.get("horizon_days") is None else int(data["horizon_days"]),
            valid_day=data.get("valid_day"),
            holdout_fraction=float(data.get("holdout_fraction", 0.0)),
            split_salt=int(data.get("split_salt", 0)),
            negative_controls=[OutcomeSpec(**{"kind": "condition", **c}) for c in data.get("negative_controls", [])],
            positive_controls=[str(c) for c in data.get("positive_controls", [])],
            phenotypes=[PhenotypeSource(**p) for p in data.get("phenotypes", [])],
            panel=data.get("panel"),
            ehr_lookback_days=None if data.get("ehr_lookback_days") is None else int(data["ehr_lookback_days"]),
            day_store=data.get("day_store"),
        )
        spec.validate()
        if spec.day_store is not None:  # a spec as written: its day store must name minute-level sources it has
            from codas.aou import daystore

            daystore.validate(spec)
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> "CohortSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# A query parameter as (name, BigQuery type, value); array types are written "ARRAY<INT64>".
QueryParam = tuple[str, str, Any]


def _scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    return "STRING"


def _base_columns(spec: CohortSpec) -> list[str]:
    cols = [spec.outcome.name]
    if is_survival(spec):
        cols.append(FOLLOWUP_COLUMN)
    if spec.include_demographics:
        cols += ["age", "female"]
    for source in spec.wearables:
        cols.append(f"{source.prefix}_days")
        for column in source.columns:
            cols += [f"{source.prefix}_{column}_mean", f"{source.prefix}_{column}_sd"]
    for phenotype in spec.phenotypes:
        cols += phenotype.output_columns()
    return cols


def availability_columns(spec: CohortSpec) -> list[str]:
    """The per-source counts of days with data in the window (``<prefix>_days``): how much a participant
    wore or synced the device, not a physiological measurement. Adjustment-only, never a candidate."""
    return [f"{source.prefix}_days" for source in [*spec.wearables, *spec.phenotypes]]


def quality_rules(spec: CohortSpec, *, negative_controls: bool = False) -> list[str]:
    """Every quality rule the spec applies, in words: what a reader needs to know about which days, nights,
    values and people count. The negative controls' rules (sealed-table columns the agents never see) are
    listed only when asked for, for the report."""
    rules = []
    if spec.valid_day is not None:
        limits = [spec.valid_day.get("min"), spec.valid_day.get("max")]
        if limits != [None, None]:
            rules.append(f"valid wear day: {spec.wearables[0].table}.{spec.valid_day['column']} "
                         f"{range_text(limits)}")
    for source in spec.wearables:
        for column, limits in sorted((source.value_bounds or {}).items()):
            if list(limits) != [None, None]:
                rules.append(f"{source.prefix}: days with {column} outside {range_text(limits)} (or missing) "
                             "are left out")
    for source in spec.phenotypes:
        rules += phenotype_rules(source)
    rules.append(f"at least {spec.min_wear_days} valid wear days in the {spec.window_days}-day window")
    outcomes = [("outcome", spec.outcome)] + [(f"negative control {c.name}", c)
                                              for c in (spec.negative_controls if negative_controls else [])]
    for what, outcome in outcomes:
        if outcome.min_occurrences > 1:
            rules.append(f"{what}: a case needs diagnoses on at least {outcome.min_occurrences} distinct dates; "
                         "people with fewer are left out")
    if spec.outcome.kind == "measurement":
        rules.append("outcome units: " + (f"only unit concepts {', '.join(map(str, spec.outcome.unit_concept_ids))}"
                                          if spec.outcome.unit_concept_ids else "every unit kept (see the unit summary)"))
    if spec.ehr_lookback_days:
        rules.append(f"everyone has an EHR visit in the {spec.ehr_lookback_days} days before the index date")
    return rules


UNIT_COLUMN, UNIT_NAME_COLUMN = "_outcome_unit", "_outcome_unit_name"


def unit_mix(frame: Any, value_column: str, *, tail: int = 20, share: float = 0.05) -> dict[str, Any]:
    """The units of a lab outcome's values in ``frame`` (the discovery share): each unit's share of the
    people, in whole percent (a unit with 1-20 people is shown only as that, so no small count can be read
    off), with quartiles when at least ``2 * tail`` people have it; and whether the values are on more than
    one scale (two or more units each holding at least ``share`` of them)."""
    import pandas as pd

    from codas.aou.privacy import SMALL_CELL_MAX

    if UNIT_COLUMN not in frame.columns or not len(frame):
        return {"units": [], "mixed": False}
    names = frame[UNIT_NAME_COLUMN] if UNIT_NAME_COLUMN in frame.columns else pd.Series(None, index=frame.index)
    units = pd.DataFrame({"unit": frame[UNIT_COLUMN], "name": names, "value": frame[value_column]})
    rows, shares = [], []
    for (unit, name), group in units.groupby(["unit", "name"], dropna=False):
        n = len(group)
        shares.append(n / len(units))
        row = {"unit_concept_id": None if pd.isna(unit) else int(unit),
               "unit": "not recorded" if pd.isna(name) else str(name),
               "share": f"{SMALL_CELL_MAX} people or fewer" if n <= SMALL_CELL_MAX else f"{100 * n / len(units):.0f}%",
               "_n": n}
        if n >= 2 * tail:
            q = group["value"].quantile([0.25, 0.5, 0.75])
            row.update(p25=round(float(q.iloc[0]), 4), median=round(float(q.iloc[1]), 4), p75=round(float(q.iloc[2]), 4))
        rows.append(row)
    rows.sort(key=lambda r: -r["_n"])
    for row in rows:
        del row["_n"]
    return {"units": rows, "mixed": sum(1 for x in shares if x >= share) > 1}


def half_columns(source: WearableSource) -> list[str]:
    """Internal columns: each column's mean and SD over the first (h1) and the second (h2) half of the
    window, inputs of the split-half reliability; never written to a table."""
    return [f"_{source.prefix}_{column}_{stat}_{half}" for column in source.columns
            for half in ("h1", "h2") for stat in ("mean", "sd")]


def split_half(frame: Any, spec: CohortSpec, *, tail: int = 20) -> dict[str, dict[str, Any]]:
    """Each daily feature's agreement between the first and the second half of the window across people
    (Spearman rho), and the reliability of the whole-window value it implies (Spearman-Brown: 2 rho / (1 +
    rho)). Two halves that barely agree say the value describes those weeks more than the person. Computed
    only when at least ``2 * tail`` people have both halves."""
    import pandas as pd

    out: dict[str, dict[str, Any]] = {}
    for source in spec.wearables:
        for column in source.columns:
            for stat in ("mean", "sd"):
                first, second = (f"_{source.prefix}_{column}_{stat}_{half}" for half in ("h1", "h2"))
                if first not in frame.columns or second not in frame.columns:
                    continue
                both = frame[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(both) < 2 * tail or both[first].nunique() < 2 or both[second].nunique() < 2:
                    continue
                rho = float(both[first].rank().corr(both[second].rank()))
                out[f"{source.prefix}_{column}_{stat}"] = {
                    "split_half_rho": round(rho, 3), "reliability": round(2 * rho / (1 + rho), 3) if rho > -1 else None}
    return out


def extraction_qc(frame: Any, spec: CohortSpec, *, tail: int = 20) -> dict[str, Any]:
    """The extraction's quality summary of ``frame`` (give it the discovery share only): the units of a lab
    outcome's values and each daily feature's split-half reliability. Aggregate, small counts withheld."""
    units = (unit_mix(frame, spec.outcome.name, tail=tail) if spec.outcome.kind == "measurement"
             else {"units": [], "mixed": False})
    return {"units": units, "reliability": split_half(frame, spec, tail=tail)}


def _bounds_rule(source: WearableSource, alias: str, tag: str, params: list[QueryParam]) -> str:
    """The source's value bounds on rows aliased ``alias``, bound as ``<tag>_<column>_low/high``."""
    rule = ""
    for column, limits in sorted((source.value_bounds or {}).items()):
        for bound, op, value in zip(("low", "high"), (">=", "<="), limits):
            if value is not None:
                rule += f"\n    AND {alias}.{column} {op} @{tag}_{column}_{bound}"
                params.append((f"{tag}_{column}_{bound}", "FLOAT64", float(value)))
    return rule


def output_columns(spec: CohortSpec) -> list[str]:
    """The columns of the extracted table, in order."""
    return _base_columns(spec) + [c.name for c in spec.covariates]


FOLLOWUP_COLUMN = "followup_days"
HOLDOUT_COLUMN = "_holdout"
# The calendar year of the index date: the enrolment era the holdout checks each effect across. It is written to
# the sealed table only, so no agent sees it.
ERA_COLUMN = "_index_year"
# Holdout split: the person id is reduced modulo the prime 2**31 - 1, mixed (multiply, add the salt,
# square, multiply), and mapped to a bucket in 0..999. Every intermediate stays below 2**62, so
# 12-digit All of Us person ids cannot overflow INT64; the squaring makes a new salt an unrelated
# split rather than a shifted one.
_SPLIT_MOD, _SPLIT_MUL, MAX_SPLIT_SALT = 2147483647, 48271, 10**9
_SPLIT_MIXED = f"MOD(MOD(person_id, {_SPLIT_MOD}) * {_SPLIT_MUL} + @split_salt, {_SPLIT_MOD})"
_SPLIT_SQL = f"MOD(MOD(MOD({_SPLIT_MIXED} * {_SPLIT_MIXED}, {_SPLIT_MOD}) * {_SPLIT_MUL}, {_SPLIT_MOD}), 1000)"


def holdout_bucket(person_id: int, salt: int = 0) -> int:
    """The split bucket (0-999) the extraction SQL assigns to ``person_id``; below the share x 1000 is sealed."""
    h = (person_id % _SPLIT_MOD * _SPLIT_MUL + salt) % _SPLIT_MOD
    return h * h % _SPLIT_MOD * _SPLIT_MUL % _SPLIT_MOD % 1000
NC_PREFIX = "nc_"


def negative_control_columns(spec: CohortSpec) -> list[str]:
    """The sealed table's negative-control columns: an event flag and its follow-up days per control."""
    return [col for c in spec.negative_controls for col in (f"{NC_PREFIX}{c.name}", f"{NC_PREFIX}{c.name}_days")]


def holdout_columns(spec: CohortSpec) -> list[str]:
    """The columns of the sealed replication table: the analysis columns plus the negative controls and, when
    there is a sealed holdout, the enrolment era."""
    return output_columns(spec) + negative_control_columns(spec) + ([ERA_COLUMN] if spec.holdout_fraction > 0 else [])


def heterogeneity_columns(spec: CohortSpec) -> list[str]:
    """What the holdout checks each effect's consistency across: sex and age, sub-study membership (each
    ``in_table`` covariate, e.g. a device brought by the participant or one handed out by a study) and, with a
    sealed holdout, the enrolment era."""
    columns = ["female", "age"] if spec.include_demographics else []
    columns += [c.name for c in spec.covariates if c.kind == "in_table"]
    return columns + ([ERA_COLUMN] if spec.holdout_fraction > 0 else [])


def negative_controls_without(controls: list[OutcomeSpec], outcome: OutcomeSpec) -> list[OutcomeSpec]:
    """``controls`` minus any that shares a concept id with ``outcome`` (it cannot be its own control)."""
    return [c for c in controls if not set(c.concept_ids) & set(outcome.concept_ids)]


def is_survival(spec: CohortSpec) -> bool:
    """An incident condition: the outcome is time to a new diagnosis (event + follow-up days)."""
    return spec.design == "incident" and spec.outcome.kind == "condition"


def covariates_without(covariates: list[CovariateSpec], outcome: OutcomeSpec) -> tuple[list[CovariateSpec], list[str]]:
    """Split ``covariates`` into those usable with ``outcome`` and the names of those that share a
    concept id with it (a covariate cannot be the outcome itself)."""
    keep, dropped = [], []
    for covariate in covariates:
        (dropped.append(covariate.name) if set(covariate.concept_ids) & set(outcome.concept_ids)
         else keep.append(covariate))
    return keep, dropped


_LAST_SEEN = ("last_seen AS (\n"
              "  SELECT person_id, MAX(visit_start_date) AS last_seen FROM {visits}\n"
              "  GROUP BY person_id\n"
              ")")


def _valid_rule(spec: CohortSpec, params: list[QueryParam]) -> str:
    """The optional valid-day rule on the primary wearable's rows (aliased as ``{a}``); binds its bounds."""
    valid = ""
    if spec.valid_day is not None:
        column = spec.valid_day["column"]
        if spec.valid_day.get("min") is not None:
            valid += f"\n      AND {{a}}.{column} >= @valid_min"
            params.append(("valid_min", "FLOAT64", float(spec.valid_day["min"])))
        if spec.valid_day.get("max") is not None:
            valid += f"\n      AND {{a}}.{column} <= @valid_max"
            params.append(("valid_max", "FLOAT64", float(spec.valid_day["max"])))
    return valid


def _first_wear_cte(primary: WearableSource, t: Any, valid: str) -> str:
    return ("first_wear AS (\n"
            f"  SELECT w.person_id, MIN(DATE(w.{primary.date_column})) AS first_wear\n"
            f"  FROM {t(primary.table)} w\n"
            f"  WHERE TRUE{valid.format(a='w')}\n"
            "  GROUP BY w.person_id\n"
            ")")


def _phenotype_parts(spec: CohortSpec, t: Any, valid: str,
                     params: list[QueryParam]) -> tuple[list[str], list[tuple[str, PhenotypeSource]]]:
    """The intraday phenotype CTEs over ``index_events`` (person_id, index_date), and each source's CTE
    alias. They read only the person and the index date, never the outcome."""
    primary = spec.wearables[0]
    ctes: list[str] = []
    if any(ph.kind in ("minute_activity", "minute_heart_rate") for ph in spec.phenotypes):
        # The primary wearable's valid days in the window: the only days intraday phenotypes use.
        ctes.append(
            "valid_days AS (\n"
            f"  SELECT DISTINCT ie.person_id, DATE(s.{primary.date_column}) AS d\n"
            "  FROM index_events ie\n"
            f"  JOIN {t(primary.table)} s ON s.person_id = ie.person_id\n"
            f"    AND DATE(s.{primary.date_column}) >= DATE_SUB(ie.index_date, INTERVAL @window_days DAY)\n"
            f"    AND DATE(s.{primary.date_column}) < ie.index_date{valid.format(a='s')}"
            + (f"\n    AND LOWER(CAST(s.{primary.filter_column} AS STRING)) = LOWER(CAST(@w0_filter AS STRING))"
               if primary.filter_column is not None else "") + "\n"
            ")"
        )
    aliases: list[tuple[str, PhenotypeSource]] = []
    steps_min = {ph.prefix: f"q{k}_min" for k, ph in enumerate(spec.phenotypes) if ph.kind == "minute_activity"}
    sleeps = {ph.prefix: f"z{k}" for k, ph in enumerate(spec.phenotypes) if ph.kind == "sleep_stages"}
    # Heart-rate sources come last: they read a minute-activity source's minutes (per_step) and a sleep-stage
    # source's sleep periods (sleep_hr, sedentary_hr).
    order = sorted(enumerate(spec.phenotypes), key=lambda kp: kp[1].kind == "minute_heart_rate")
    for k, phenotype in order:
        if phenotype.kind == "minute_activity":
            aliases.append((f"q{k}", phenotype))
            ctes += minute_activity_ctes(k, phenotype, t(phenotype.tables()[0]))
        elif phenotype.kind == "minute_heart_rate":
            aliases.append((f"h{k}", phenotype))
            ctes += minute_heart_rate_ctes(k, phenotype, t(phenotype.tables()[0]), steps_min.get(phenotype.steps_from),
                                           sleeps.get(phenotype.sleep_from))
        else:
            aliases.append((f"z{k}", phenotype))
            sleep_sql, sleep_params = sleep_stage_ctes(k, phenotype, t(phenotype.tables()[0]))
            ctes += sleep_sql
            params += sleep_params
    return ctes, aliases


def build_phenotype_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """The intraday phenotypes of every participant with a valid wear day, under the incident design.

    There the index date is the end of the baseline window that starts at the first valid wear day: a
    property of the person, not of the outcome. So these values are the same in every outcome's cohort,
    and the minute-level tables (terabytes) can be read once per run instead of once per outcome. The
    result carries ``person_id`` for the merge and is never written where the agents read."""
    spec.validate()
    if spec.design != "incident":
        raise ValueError("Phenotypes are shared across outcomes only under the incident design.")
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    primary = spec.wearables[0]
    params: list[QueryParam] = [("window_days", "INT64", int(spec.window_days))]
    valid = _valid_rule(spec, params)
    if primary.filter_column is not None:
        params.append(("w0_filter", _scalar_type(primary.filter_value), primary.filter_value))
    ctes = [_first_wear_cte(primary, t, valid),
            "index_events AS (\n  SELECT person_id, DATE_ADD(first_wear, INTERVAL @window_days DAY) AS index_date\n"
            "  FROM first_wear\n)"]
    extra, aliases = _phenotype_parts(spec, t, valid, params)
    select = ["ie.person_id"] + [item for alias, ph in aliases for item in select_columns(alias, ph)]
    joins = [f"LEFT JOIN {alias} ON {alias}.person_id = ie.person_id" for alias, _ph in aliases]
    sql = ("WITH " + ",\n".join(ctes + extra) + "\n"
           "SELECT\n  " + ",\n  ".join(select) + "\n"
           "FROM index_events ie\n" + "\n".join(joins))
    return sql, params


def build_cohort_sql(spec: CohortSpec, cdr: str, *, phenotypes: bool = True, qc: bool = False,
                     person_id: bool = False, person_level: bool = False) -> tuple[str, list[QueryParam]]:
    """Return the extraction SQL and its parameters. Identifiers are validated, values are bound.

    With ``phenotypes=False`` the intraday phenotypes are left out and ``person_id`` is selected so that
    the phenotypes computed once by ``build_phenotype_sql`` can be merged in (then dropped). With ``qc=True``
    the inputs of ``extraction_qc`` are selected too (a lab outcome's unit, id and name; each daily column's
    mean and SD over each half of the window); the extraction drops them before writing any table.

    With ``person_level`` (incident design) the table holds the discovery share's participants with a valid
    baseline whatever their outcome: ``person_id``, the days from the index date to the last EHR visit
    (``last_seen_day``) and the person-level columns, built by the same code as each outcome's table, for the
    phenome scan (``codas.aou.phenome``). The sealed share is left out in the query."""
    spec.validate()
    if person_level and spec.design != "incident":
        raise ValueError("A person-level table needs the incident design, whose index date is a person's own.")
    qc = qc and not person_level
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = [
        ("window_days", "INT64", int(spec.window_days)),
        ("min_wear_days", "INT64", int(spec.min_wear_days)),
    ]
    ctes: list[str] = []
    if not person_level:
        params.insert(0, ("outcome_concept_ids", "ARRAY<INT64>", list(spec.outcome.concept_ids)))
        concepts = "SELECT concept_id FROM UNNEST(@outcome_concept_ids) AS concept_id"
        if spec.outcome.include_descendants:
            concepts += (f"\n  UNION DISTINCT\n  SELECT descendant_concept_id FROM {t('concept_ancestor')}"
                         "\n  WHERE ancestor_concept_id IN UNNEST(@outcome_concept_ids)")
        ctes.append(f"outcome_concepts AS (\n  {concepts}\n)")

    primary = spec.wearables[0]
    valid = _valid_rule(spec, params)
    bounds = ""
    if spec.outcome.kind == "measurement" and not person_level:
        if spec.outcome.value_min is not None:
            bounds += "\n    AND m.value_as_number >= @value_min"
            params.append(("value_min", "FLOAT64", float(spec.outcome.value_min)))
        if spec.outcome.value_max is not None:
            bounds += "\n    AND m.value_as_number <= @value_max"
            params.append(("value_max", "FLOAT64", float(spec.outcome.value_max)))
        if spec.outcome.unit_concept_ids:
            bounds += "\n    AND m.unit_concept_id IN UNNEST(@outcome_units)"
            params.append(("outcome_units", "ARRAY<INT64>", list(spec.outcome.unit_concept_ids)))
    # With qc=True a lab outcome's value carries its unit, summarised by the extraction and then dropped.
    unit = ", m.unit_concept_id AS outcome_unit" if qc and spec.outcome.kind == "measurement" else ""
    unit_out = ", outcome_unit" if unit else ""

    if person_level:  # everyone with a baseline; each outcome's rules are applied by the caller
        ctes += [_first_wear_cte(primary, t, valid),
                 "index_events AS (\n  SELECT person_id, DATE_ADD(first_wear, INTERVAL @window_days DAY) AS index_date\n"
                 "  FROM first_wear\n)", _LAST_SEEN.format(visits=t("visit_occurrence"))]
    elif spec.design == "incident":
        # One index rule for everyone: the end of a baseline window that starts at the first valid
        # wear day. Outcomes are observed from index + lag; earlier ones exclude the person.
        params.append(("outcome_from", "INT64", int(spec.window_days + spec.lag_days)))
        ctes.append(_first_wear_cte(primary, t, valid))
        if spec.outcome.kind == "measurement":
            horizon = ""
            if spec.horizon_days is not None:
                params.append(("outcome_until", "INT64", int(spec.window_days + spec.lag_days + spec.horizon_days)))
                horizon = "\n    AND m.measurement_date < DATE_ADD(f.first_wear, INTERVAL @outcome_until DAY)"
            ctes.append(
                "outcome_events AS (\n"
                f"  SELECT m.person_id, f.first_wear, m.value_as_number AS outcome_value{unit},\n"
                "    ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date, m.measurement_id) AS rn\n"
                f"  FROM {t('measurement')} m\n"
                "  JOIN outcome_concepts oc ON m.measurement_concept_id = oc.concept_id\n"
                "  JOIN first_wear f ON f.person_id = m.person_id\n"
                f"  WHERE m.value_as_number IS NOT NULL{bounds}\n"
                f"    AND m.measurement_date >= DATE_ADD(f.first_wear, INTERVAL @outcome_from DAY){horizon}\n"
                ")"
            )
            ctes.append("index_events AS (\n  SELECT person_id, DATE_ADD(first_wear, INTERVAL @window_days DAY) AS index_date,"
                        f" outcome_value{unit_out}\n  FROM outcome_events WHERE rn = 1\n)")
            if spec.negative_controls:
                ctes.append(_LAST_SEEN.format(visits=t("visit_occurrence")))
        else:
            ctes.append(_cases_cte(spec.outcome, t, params))
            ctes.append(_LAST_SEEN.format(visits=t("visit_occurrence")))
            start_fu = "DATE_ADD(f.first_wear, INTERVAL @outcome_from DAY)"
            ctes.append(
                "index_events AS (\n"
                "  SELECT f.person_id, DATE_ADD(f.first_wear, INTERVAL @window_days DAY) AS index_date,\n"
                "    IF(c.dx_date IS NULL, 0, 1) AS outcome_value,\n"
                f"    DATE_DIFF(COALESCE(c.dx_date, l.last_seen), {start_fu}, DAY) AS {FOLLOWUP_COLUMN}\n"
                "  FROM first_wear f\n"
                "  JOIN last_seen l ON l.person_id = f.person_id\n"
                "  LEFT JOIN cases c ON c.person_id = f.person_id\n"
                f"  WHERE (c.dx_date IS NULL OR (c.dx_date >= {start_fu}{_enough('c', spec.outcome)}))\n"
                f"    AND COALESCE(c.dx_date, l.last_seen) > {start_fu}\n"
                ")"
            )
    elif spec.outcome.kind == "measurement":
        ctes.append(
            "outcome_events AS (\n"
            f"  SELECT m.person_id, m.measurement_date AS index_date, m.value_as_number AS outcome_value{unit},\n"
            "    ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date, m.measurement_id) AS rn\n"
            f"  FROM {t('measurement')} m\n"
            "  JOIN outcome_concepts oc ON m.measurement_concept_id = oc.concept_id\n"
            f"  WHERE m.value_as_number IS NOT NULL{bounds}\n"
            f"    AND EXISTS (SELECT 1 FROM {t(primary.table)} w WHERE w.person_id = m.person_id\n"
            f"      AND DATE(w.{primary.date_column}) >= DATE_SUB(m.measurement_date, INTERVAL @window_days DAY)\n"
            f"      AND DATE(w.{primary.date_column}) < m.measurement_date{valid.format(a='w')})\n"
            ")"
        )
        ctes.append(f"index_events AS (\n  SELECT person_id, index_date, outcome_value{unit_out} FROM outcome_events"
                    " WHERE rn = 1\n)")
    else:
        ctes.append(_cases_cte(spec.outcome, t, params))
        ctes.append(
            "wear_persons AS (\n"
            f"  SELECT w.person_id, MAX(DATE(w.{primary.date_column})) AS last_wear_date\n"
            f"  FROM {t(primary.table)} w\n"
            f"  WHERE TRUE{valid.format(a='w')}\n"
            "  GROUP BY w.person_id\n"
            ")"
        )
        ctes.append(
            "index_events AS (\n"
            "  SELECT w.person_id, COALESCE(c.dx_date, w.last_wear_date) AS index_date,\n"
            "    IF(c.person_id IS NULL, 0, 1) AS outcome_value\n"
            "  FROM wear_persons w LEFT JOIN cases c ON c.person_id = w.person_id\n"
            + ("  WHERE c.person_id IS NULL OR c.dx_dates >= @min_occurrences\n"
               if spec.outcome.min_occurrences > 1 else "") +
            ")"
        )

    if qc:
        params.append(("half_days", "INT64", int(spec.window_days) // 2))
    for i, source in enumerate(spec.wearables):
        aggregates = [f"COUNT(DISTINCT DATE(s.{source.date_column})) AS {source.prefix}_days"]
        for column in source.columns:
            aggregates.append(f"AVG(s.{column}) AS {source.prefix}_{column}_mean")
            aggregates.append(f"STDDEV_SAMP(s.{column}) AS {source.prefix}_{column}_sd")
        filt = valid.format(a="s") if i == 0 else ""
        if source.filter_column is not None:
            name = f"w{i}_filter"
            # Compare as lowercase text: CDR flag columns differ in type between releases (the
            # Registered Tier v9 stores is_main_sleep as STRING 'true'/'false', not BOOL).
            filt += (f"\n    AND LOWER(CAST(s.{source.filter_column} AS STRING))"
                     f" = LOWER(CAST(@{name} AS STRING))")
            params.append((name, _scalar_type(source.filter_value), source.filter_value))
        filt += _bounds_rule(source, "s", f"w{i}b", params)
        if qc:  # each column over the first and the second half of the window, for split-half reliability
            first = f"DATE(s.{source.date_column}) < DATE_SUB(ie.index_date, INTERVAL @half_days DAY)"
            for column in source.columns:
                for half in ("h1", "h2"):
                    inside = first if half == "h1" else f"NOT ({first})"
                    aggregates += [f"AVG(IF({inside}, s.{column}, NULL)) AS _{source.prefix}_{column}_mean_{half}",
                                   f"STDDEV_SAMP(IF({inside}, s.{column}, NULL)) AS _{source.prefix}_{column}_sd_{half}"]
        ctes.append(
            f"w{i} AS (\n"
            "  SELECT ie.person_id,\n    " + ",\n    ".join(aggregates) + "\n"
            "  FROM index_events ie\n"
            f"  JOIN {t(source.table)} s ON s.person_id = ie.person_id\n"
            f"    AND DATE(s.{source.date_column}) >= DATE_SUB(ie.index_date, INTERVAL @window_days DAY)\n"
            f"    AND DATE(s.{source.date_column}) < ie.index_date{filt}\n"
            "  GROUP BY ie.person_id\n"
            ")"
        )

    for i, covariate in enumerate(spec.covariates):
        if covariate.kind in _CONCEPT_FREE:
            ctes.append(_concept_free_cte(i, covariate, t, params))
            continue
        table, concept_col, date_col = _COVARIATE_TABLES[covariate.kind]
        ids = f"c{i}_concept_ids"
        params.append((ids, "ARRAY<INT64>", list(covariate.concept_ids)))
        concepts = f"SELECT concept_id FROM UNNEST(@{ids}) AS concept_id"
        if covariate.include_descendants:
            concepts += (f"\n    UNION DISTINCT\n    SELECT descendant_concept_id FROM {t('concept_ancestor')}"
                         f"\n    WHERE ancestor_concept_id IN UNNEST(@{ids})")
        # A measurement may be taken on the index date itself; a condition or drug record must start
        # before it, so a diagnosis made at the index visit is not counted as prior history.
        upper = "<=" if covariate.kind == "measurement" else "<"
        where = f"r.{date_col} {upper} ie.index_date"
        if covariate.lookback_days is not None:
            params.append((f"c{i}_lookback", "INT64", int(covariate.lookback_days)))
            where += f"\n      AND r.{date_col} >= DATE_SUB(ie.index_date, INTERVAL @c{i}_lookback DAY)"
        join = (f"  FROM index_events ie\n"
                f"  JOIN {t(table)} r ON r.person_id = ie.person_id\n"
                f"  JOIN (\n    {concepts}\n  ) cc ON r.{concept_col} = cc.concept_id\n")
        if covariate.kind == "measurement":
            if covariate.value_min is not None:
                where += f"\n      AND r.value_as_number >= @c{i}_min"
                params.append((f"c{i}_min", "FLOAT64", float(covariate.value_min)))
            if covariate.value_max is not None:
                where += f"\n      AND r.value_as_number <= @c{i}_max"
                params.append((f"c{i}_max", "FLOAT64", float(covariate.value_max)))
            ctes.append(
                f"c{i} AS (\n"
                f"  SELECT person_id, value AS {covariate.name} FROM (\n"
                "  SELECT ie.person_id, r.value_as_number AS value,\n"
                f"    ROW_NUMBER() OVER (PARTITION BY ie.person_id ORDER BY r.{date_col} DESC, r.measurement_id DESC) AS rn\n"
                + join +
                f"  WHERE r.value_as_number IS NOT NULL\n      AND {where}\n"
                "  ) AS ranked WHERE rn = 1\n"
                ")"
            )
        elif covariate.kind == "survey":
            params += [(f"c{i}_positive", "STRING", covariate.positive_answers.lower()),
                       (f"c{i}_missing", "STRING", covariate.missing_answers.lower())]
            if covariate.any_time:  # the answer nearest the index date, before or after it
                where = "TRUE"
                order = f"ABS(DATE_DIFF(r.{date_col}, ie.index_date, DAY)), r.observation_id DESC"
            else:  # the latest answer on or before the index date (within the lookback, if any)
                where = f"r.{date_col} <= ie.index_date" + (
                    f"\n      AND r.{date_col} >= DATE_SUB(ie.index_date, INTERVAL @c{i}_lookback DAY)"
                    if covariate.lookback_days is not None else "")
                order = f"r.{date_col} DESC, r.observation_id DESC"
            ctes.append(
                f"c{i} AS (\n"
                f"  SELECT person_id, {covariate.name} FROM (\n"
                "  SELECT ie.person_id,\n"
                f"    CASE WHEN a.concept_name IS NULL OR REGEXP_CONTAINS(LOWER(a.concept_name), @c{i}_missing) THEN NULL\n"
                f"      WHEN REGEXP_CONTAINS(LOWER(a.concept_name), @c{i}_positive) THEN 1 ELSE 0 END AS {covariate.name},\n"
                f"    ROW_NUMBER() OVER (PARTITION BY ie.person_id ORDER BY {order}) AS rn\n"
                + join +
                f"  LEFT JOIN {t('concept')} a ON a.concept_id = r.value_source_concept_id\n"
                f"  WHERE {where}\n"
                "  ) AS ranked WHERE rn = 1\n"
                ")"
            )
        else:
            ctes.append(
                f"c{i} AS (\n"
                "  SELECT DISTINCT ie.person_id, 1 AS present\n"
                + join +
                f"  WHERE {where}\n"
                ")"
            )

    phenotype_ctes: list[tuple[str, PhenotypeSource]] = []
    if phenotypes:
        extra, phenotype_ctes = _phenotype_parts(spec, t, valid, params)
        ctes += extra

    if person_level:
        select = ["ie.person_id", "DATE_DIFF(ls.last_seen, ie.index_date, DAY) AS last_seen_day"]
        joins = ["LEFT JOIN last_seen ls ON ls.person_id = ie.person_id"]
    else:
        select = [f"ie.outcome_value AS {spec.outcome.name}"]
        if unit:  # the value's unit, summarised for the report and then dropped
            select += [f"ie.outcome_unit AS {UNIT_COLUMN}", f"uc.concept_name AS {UNIT_NAME_COLUMN}"]
        if not phenotypes or person_id:  # the caller merges by person (phenotype_table, the day store), then drops it
            select.insert(0, "ie.person_id")
        if is_survival(spec):
            select.append(f"ie.{FOLLOWUP_COLUMN}")
        joins = [f"LEFT JOIN {t('concept')} uc ON uc.concept_id = ie.outcome_unit"] if unit else []
    if spec.include_demographics:
        select += [
            "EXTRACT(YEAR FROM ie.index_date) - p.year_of_birth AS age",
            "CASE LOWER(sc.concept_name) WHEN 'female' THEN 1 WHEN 'male' THEN 0 END AS female",
        ]
        joins += [
            f"JOIN {t('person')} p ON p.person_id = ie.person_id",
            f"LEFT JOIN {t('concept')} sc ON sc.concept_id = p.sex_at_birth_concept_id",
        ]
    for i, source in enumerate(spec.wearables):
        select.append(f"w{i}.{source.prefix}_days")
        for column in source.columns:
            select += [f"w{i}.{source.prefix}_{column}_mean", f"w{i}.{source.prefix}_{column}_sd"]
        if qc:
            select += [f"w{i}.{column}" for column in half_columns(source)]
        joins.append(f"LEFT JOIN w{i} ON w{i}.person_id = ie.person_id")
    for alias, phenotype in phenotype_ctes:
        select += select_columns(alias, phenotype)
        joins.append(f"LEFT JOIN {alias} ON {alias}.person_id = ie.person_id")
    for i, covariate in enumerate(spec.covariates):
        if covariate.kind in ("measurement", "survey"):
            select.append(f"c{i}.{covariate.name}")
        elif covariate.kind == "visits":
            select.append(f"COALESCE(c{i}.visits, 0) AS {covariate.name}")
        else:
            select.append(f"IF(c{i}.present IS NULL, 0, 1) AS {covariate.name}")
        joins.append(f"LEFT JOIN c{i} ON c{i}.person_id = ie.person_id")
    if spec.negative_controls and not person_level:
        # Follow-up for a negative control starts where the outcome's does (index + lag); a diagnosis
        # before then leaves that control empty for the person rather than excluding them.
        params.append(("nc_lag_days", "INT64", int(spec.lag_days)))
        start = "DATE_ADD(ie.index_date, INTERVAL @nc_lag_days DAY)"
        joins.append("LEFT JOIN last_seen ls ON ls.person_id = ie.person_id")
        for i, control in enumerate(spec.negative_controls):
            ids = f"n{i}_concept_ids"
            params.append((ids, "ARRAY<INT64>", list(control.concept_ids)))
            concepts = f"SELECT concept_id FROM UNNEST(@{ids}) AS concept_id"
            if control.include_descendants:
                concepts += (f"\n    UNION DISTINCT\n    SELECT descendant_concept_id FROM {t('concept_ancestor')}"
                             f"\n    WHERE ancestor_concept_id IN UNNEST(@{ids})")
            dates = ",\n    COUNT(DISTINCT co.condition_start_date) AS dx_dates" if control.min_occurrences > 1 else ""
            ctes.append(
                f"n{i} AS (\n"
                f"  SELECT co.person_id, MIN(co.condition_start_date) AS dx_date{dates}\n"
                f"  FROM {t('condition_occurrence')} co\n"
                f"  JOIN (\n    {concepts}\n  ) nc ON co.condition_concept_id = nc.concept_id\n"
                "  GROUP BY co.person_id\n"
                ")"
            )
            col = f"{NC_PREFIX}{control.name}"
            # Diagnosed before follow-up, or too few diagnosis dates to count as a case: left out.
            excluded = f"n{i}.dx_date < {start}"
            if control.min_occurrences > 1:
                params.append((f"n{i}_min_occurrences", "INT64", int(control.min_occurrences)))
                excluded += f" OR n{i}.dx_dates < @n{i}_min_occurrences"
            select += [f"CASE WHEN {excluded} THEN NULL WHEN n{i}.dx_date IS NULL THEN 0 ELSE 1 END AS {col}",
                       f"CASE WHEN {excluded} THEN NULL ELSE DATE_DIFF(COALESCE(n{i}.dx_date, ls.last_seen), {start}, DAY)"
                       f" END AS {col}_days"]
            joins.append(f"LEFT JOIN n{i} ON n{i}.person_id = ie.person_id")
    if spec.ehr_lookback_days:
        params.append(("ehr_lookback_days", "INT64", int(spec.ehr_lookback_days)))
        ctes.append("ehr_active AS (\n"
                    "  SELECT DISTINCT ie.person_id FROM index_events ie\n"
                    f"  JOIN {t('visit_occurrence')} v ON v.person_id = ie.person_id\n"
                    "    AND v.visit_start_date < ie.index_date\n"
                    "    AND v.visit_start_date >= DATE_SUB(ie.index_date, INTERVAL @ehr_lookback_days DAY)\n"
                    ")")
        joins.append("JOIN ehr_active ea ON ea.person_id = ie.person_id")
    if spec.holdout_fraction > 0:
        # Deterministic, roughly uniform split on the person id, computed apart from the selected
        # columns so the output carries only the flag; extract_cohort drops it before the agents see rows.
        params += [("holdout_permille", "INT64", int(round(spec.holdout_fraction * 1000))),
                   ("split_salt", "INT64", int(spec.split_salt))]
        ctes.append(f"split AS (\n  SELECT person_id, {_SPLIT_SQL} < @holdout_permille AS {HOLDOUT_COLUMN}\n"
                    "  FROM index_events\n)")
        if not person_level:
            select += [f"sp.{HOLDOUT_COLUMN}", f"EXTRACT(YEAR FROM ie.index_date) AS {ERA_COLUMN}"]
        joins.append("JOIN split sp ON sp.person_id = ie.person_id")

    sql = (
        "WITH " + ",\n".join(ctes) + "\n"
        "SELECT\n  " + ",\n  ".join(select) + "\n"
        "FROM index_events ie\n" + "\n".join(joins) + "\n"
        f"WHERE w0.{primary.prefix}_days >= @min_wear_days"
        # A person-level table holds the discovery share only: the sealed rows never leave BigQuery.
        + (f"\n  AND NOT sp.{HOLDOUT_COLUMN}" if person_level and spec.holdout_fraction > 0 else "")
    )
    return sql, params


def _cases_cte(outcome: OutcomeSpec, t: Any, params: list[QueryParam]) -> str:
    """Each person's first diagnosis date, and with ``min_occurrences`` how many distinct dates they have."""
    dates = ""
    if outcome.min_occurrences > 1:
        dates = ",\n    COUNT(DISTINCT co.condition_start_date) AS dx_dates"
        params.append(("min_occurrences", "INT64", int(outcome.min_occurrences)))
    return ("cases AS (\n"
            f"  SELECT co.person_id, MIN(co.condition_start_date) AS dx_date{dates}\n"
            f"  FROM {t('condition_occurrence')} co\n"
            "  JOIN outcome_concepts oc ON co.condition_concept_id = oc.concept_id\n"
            "  GROUP BY co.person_id\n"
            ")")


def _enough(alias: str, outcome: OutcomeSpec) -> str:
    """`` AND <alias>.dx_dates >= @min_occurrences`` when the case rule needs more than one date."""
    return f" AND {alias}.dx_dates >= @min_occurrences" if outcome.min_occurrences > 1 else ""


def _concept_free_cte(i: int, covariate: CovariateSpec, t: Any, params: list[QueryParam]) -> str:
    """``c{i}``: distinct visit dates in the lookback before the index date (``visits``), or the people in a
    named table (``in_table``)."""
    if covariate.kind == "in_table":
        return f"c{i} AS (\n  SELECT DISTINCT person_id, 1 AS present FROM {t(covariate.table)}\n)"
    where = "v.visit_start_date < ie.index_date"
    if covariate.lookback_days is not None:
        params.append((f"c{i}_lookback", "INT64", int(covariate.lookback_days)))
        where += f"\n    AND v.visit_start_date >= DATE_SUB(ie.index_date, INTERVAL @c{i}_lookback DAY)"
    return (f"c{i} AS (\n"
            "  SELECT ie.person_id, COUNT(DISTINCT v.visit_start_date) AS visits\n"
            "  FROM index_events ie\n"
            f"  JOIN {t('visit_occurrence')} v ON v.person_id = ie.person_id\n"
            f"  WHERE {where}\n"
            "  GROUP BY ie.person_id\n"
            ")")


def has_panel(spec: CohortSpec) -> bool:
    return spec.panel is not None and spec.outcome.kind == "measurement"


def panel_columns(spec: CohortSpec) -> list[str]:
    """Columns of the repeated-measures panel, in order: a pseudonymous person index (a rank, never the
    CDR id), the outcome value, days since the first valid wear day, demographics, wearable summaries."""
    cols = ["person_index", spec.outcome.name, "days_from_start"]
    if spec.include_demographics:
        cols += ["age", "female"]
    for source in spec.wearables:
        cols.append(f"{source.prefix}_days")
        for column in source.columns:
            cols += [f"{source.prefix}_{column}_mean", f"{source.prefix}_{column}_sd"]
    return cols


def build_panel_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """SQL for the repeated-measures panel: every qualifying outcome value at least ``period_days``
    after the first valid wear day (values on the same day averaged), with each wearable source's days,
    means and SDs over the ``period_days`` before it; occasions with fewer than ``min_days`` primary
    wear days in that period are dropped. Carries the same holdout flag as the cohort."""
    spec.validate()
    if not has_panel(spec):
        raise ValueError("The spec has no panel or its outcome is not a measurement.")
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731
    primary = spec.wearables[0]
    params: list[QueryParam] = [("outcome_concept_ids", "ARRAY<INT64>", list(spec.outcome.concept_ids)),
                                ("period_days", "INT64", int(spec.panel["period_days"])),
                                ("panel_min_days", "INT64", int(spec.panel.get("min_days", 1)))]
    concepts = "SELECT concept_id FROM UNNEST(@outcome_concept_ids) AS concept_id"
    if spec.outcome.include_descendants:
        concepts += (f"\n  UNION DISTINCT\n  SELECT descendant_concept_id FROM {t('concept_ancestor')}"
                     "\n  WHERE ancestor_concept_id IN UNNEST(@outcome_concept_ids)")
    valid = ""
    if spec.valid_day is not None:
        column = spec.valid_day["column"]
        for bound, op in (("min", ">="), ("max", "<=")):
            if spec.valid_day.get(bound) is not None:
                valid += f"\n    AND {{a}}.{column} {op} @valid_{bound}"
                params.append((f"valid_{bound}", "FLOAT64", float(spec.valid_day[bound])))
    bounds = ""
    for bound, op in (("value_min", ">="), ("value_max", "<=")):
        if getattr(spec.outcome, bound) is not None:
            bounds += f"\n    AND m.value_as_number {op} @{bound}"
            params.append((bound, "FLOAT64", float(getattr(spec.outcome, bound))))
    if spec.outcome.unit_concept_ids:
        bounds += "\n    AND m.unit_concept_id IN UNNEST(@outcome_units)"
        params.append(("outcome_units", "ARRAY<INT64>", list(spec.outcome.unit_concept_ids)))
    ctes = [f"outcome_concepts AS (\n  {concepts}\n)",
            "first_wear AS (\n"
            f"  SELECT w.person_id, MIN(DATE(w.{primary.date_column})) AS first_wear\n"
            f"  FROM {t(primary.table)} w\n  WHERE TRUE{valid.format(a='w')}\n  GROUP BY w.person_id\n)",
            "occasions AS (\n"
            "  SELECT m.person_id, m.measurement_date AS occasion, AVG(m.value_as_number) AS outcome_value,\n"
            "    MIN(f.first_wear) AS first_wear\n"
            f"  FROM {t('measurement')} m\n"
            "  JOIN outcome_concepts oc ON m.measurement_concept_id = oc.concept_id\n"
            "  JOIN first_wear f ON f.person_id = m.person_id\n"
            f"  WHERE m.value_as_number IS NOT NULL{bounds}\n"
            "    AND m.measurement_date >= DATE_ADD(f.first_wear, INTERVAL @period_days DAY)\n"
            "  GROUP BY m.person_id, m.measurement_date\n)"]
    select = ["DENSE_RANK() OVER (ORDER BY o.person_id) AS person_index", f"o.outcome_value AS {spec.outcome.name}",
              "DATE_DIFF(o.occasion, o.first_wear, DAY) AS days_from_start"]
    joins = []
    for i, source in enumerate(spec.wearables):
        aggregates = [f"COUNT(DISTINCT DATE(s.{source.date_column})) AS {source.prefix}_days"]
        for column in source.columns:
            aggregates += [f"AVG(s.{column}) AS {source.prefix}_{column}_mean",
                           f"STDDEV_SAMP(s.{column}) AS {source.prefix}_{column}_sd"]
        filt = valid.format(a="s") if i == 0 else ""
        if source.filter_column is not None:
            filt += (f"\n    AND LOWER(CAST(s.{source.filter_column} AS STRING)) = LOWER(CAST(@pf{i} AS STRING))")
            params.append((f"pf{i}", _scalar_type(source.filter_value), source.filter_value))
        filt += _bounds_rule(source, "s", f"pb{i}", params)
        ctes.append(f"v{i} AS (\n  SELECT o.person_id, o.occasion,\n    " + ",\n    ".join(aggregates) + "\n"
                    f"  FROM occasions o\n  JOIN {t(source.table)} s ON s.person_id = o.person_id\n"
                    f"    AND DATE(s.{source.date_column}) >= DATE_SUB(o.occasion, INTERVAL @period_days DAY)\n"
                    f"    AND DATE(s.{source.date_column}) < o.occasion{filt}\n"
                    "  GROUP BY o.person_id, o.occasion\n)")
        select.append(f"v{i}.{source.prefix}_days")
        for column in source.columns:
            select += [f"v{i}.{source.prefix}_{column}_mean", f"v{i}.{source.prefix}_{column}_sd"]
        joins.append(f"{'JOIN' if i == 0 else 'LEFT JOIN'} v{i} ON v{i}.person_id = o.person_id "
                     f"AND v{i}.occasion = o.occasion")
    if spec.include_demographics:
        select[3:3] = ["EXTRACT(YEAR FROM o.occasion) - p.year_of_birth AS age",
                       "CASE LOWER(sc.concept_name) WHEN 'female' THEN 1 WHEN 'male' THEN 0 END AS female"]
        joins += [f"JOIN {t('person')} p ON p.person_id = o.person_id",
                  f"LEFT JOIN {t('concept')} sc ON sc.concept_id = p.sex_at_birth_concept_id"]
    if spec.holdout_fraction > 0:
        params += [("holdout_permille", "INT64", int(round(spec.holdout_fraction * 1000))),
                   ("split_salt", "INT64", int(spec.split_salt))]
        ctes.append(f"split AS (\n  SELECT person_id, {_SPLIT_SQL} < @holdout_permille AS {HOLDOUT_COLUMN}\n"
                    "  FROM (SELECT DISTINCT person_id FROM occasions)\n)")
        select.append(f"sp.{HOLDOUT_COLUMN}")
        joins.append("JOIN split sp ON sp.person_id = o.person_id")
    sql = ("WITH " + ",\n".join(ctes) + "\nSELECT\n  " + ",\n  ".join(select) + "\nFROM occasions o\n"
           + "\n".join(joins) + f"\nWHERE v0.{primary.prefix}_days >= @panel_min_days")
    return sql, params


def qc_path(discovery_csv: str | Path) -> Path:
    """The extraction's quality summary next to a discovery table (``<name>.qc.json``)."""
    p = Path(discovery_csv)
    return p.with_name(p.stem + ".qc.json")


def panel_path(discovery_csv: str | Path) -> Path:
    """The repeated-measures panel next to a discovery table (``<name>.panel.csv``)."""
    p = Path(discovery_csv)
    return p.with_name(p.stem + ".panel.csv")


def schema_check_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """SQL listing the (table, column) pairs the spec needs that exist in the CDR."""
    cdr = resolve_cdr(cdr)
    tables = sorted({s.table for s in spec.wearables} | {t for ph in spec.phenotypes for t in ph.tables()}
                    | {_covariate_table(c) for c in spec.covariates}
                    | ({"visit_occurrence"} if _needs_visits(spec) else set())
                    | ({"condition_occurrence"} if spec.negative_controls else set()))
    sql = (f"SELECT table_name, column_name FROM `{cdr}.INFORMATION_SCHEMA.COLUMNS`\n"
           "WHERE table_name IN UNNEST(@tables)")
    return sql, [("tables", "ARRAY<STRING>", tables)]


def missing_columns(spec: CohortSpec, present: list[tuple[str, str]]) -> list[str]:
    """``table.column`` names the spec needs but the CDR does not have."""
    have = {(str(tbl), str(col)) for tbl, col in present}
    needed = []
    for s in spec.wearables:
        cols = [s.date_column, "person_id", *s.columns, *([s.filter_column] if s.filter_column else []),
                *sorted(s.value_bounds or {})]
        needed += [(s.table, c) for c in cols]
    for ph in spec.phenotypes:
        needed += [(ph.tables()[0], c) for c in ph.needed()]
    for c in spec.covariates:
        if c.kind in _CONCEPT_FREE:
            needed.append((_covariate_table(c), "person_id"))
            continue
        table, concept_col, date_col = _COVARIATE_TABLES[c.kind]
        needed += [(table, col) for col in ("person_id", concept_col, date_col)]
        if c.kind == "measurement":
            needed += [(table, "value_as_number"), (table, "measurement_id")]
        if c.kind == "survey":
            needed += [(table, "value_source_concept_id"), (table, "observation_id")]
    if _needs_visits(spec):
        needed += [("visit_occurrence", "person_id"), ("visit_occurrence", "visit_start_date")]
    if spec.negative_controls:
        needed += [("condition_occurrence", c) for c in ("person_id", "condition_concept_id", "condition_start_date")]
    return [f"{tbl}.{col}" for tbl, col in dict.fromkeys(needed) if (tbl, col) not in have]


def _covariate_table(covariate: CovariateSpec) -> str:
    if covariate.kind == "in_table":
        return covariate.table
    return "visit_occurrence" if covariate.kind == "visits" else _COVARIATE_TABLES[covariate.kind][0]


def _needs_visits(spec: CohortSpec) -> bool:
    return bool(is_survival(spec) or spec.negative_controls or spec.ehr_lookback_days
                or any(c.kind == "visits" for c in spec.covariates))


def resolve_sources(spec: CohortSpec, present: list[tuple[str, str]]) -> tuple[CohortSpec, list[str]]:
    """Fit the spec's optional parts to the columns this CDR has: an optional wearable source keeps the
    value columns it finds (and is dropped without its date, person, filter or bounded column or any value
    column); each phenotype role takes the first acceptable column the table has, and an optional
    phenotype source that cannot be resolved is dropped; a table-membership covariate whose table this CDR
    lacks is dropped. Returns the fitted spec and what was dropped.
    Required parts are left as they are, for ``missing_columns`` to report."""
    from dataclasses import replace

    have = {(str(tbl), str(col)) for tbl, col in present}
    notes: list[str] = []
    wearables = []
    for source in spec.wearables:
        if not source.optional:
            wearables.append(source)
            continue
        keys = ([source.date_column, "person_id"] + ([source.filter_column] if source.filter_column else [])
                + sorted(source.value_bounds or {}))
        found = [c for c in source.columns if (source.table, c) in have]
        lacking = [c for c in keys if (source.table, c) not in have]
        if lacking or not found:
            notes.append(f"{source.table} ({source.prefix}) skipped: no "
                         + ", ".join(lacking or source.columns) + " in this CDR")
            continue
        if len(found) < len(source.columns):
            notes.append(f"{source.table} ({source.prefix}): columns not in this CDR skipped: "
                         + ", ".join(c for c in source.columns if c not in found))
        wearables.append(replace(source, columns=found))
    phenotypes = []
    for phenotype in spec.phenotypes:
        fixed, reason = resolve(phenotype, have)
        if fixed is not None:
            phenotypes.append(fixed)
        elif phenotype.optional:
            notes.append(f"phenotypes from {reason}; skipped")
        else:
            phenotypes.append(phenotype)
    minute_activity = {ph.prefix for ph in phenotypes if ph.kind == "minute_activity"}
    for k, ph in enumerate(phenotypes):
        if ph.steps_from and ph.steps_from not in minute_activity:
            notes.append(f"{ph.prefix}_per_step skipped: its minute-activity source {ph.steps_from} is not available")
            phenotypes[k] = ph = replace(ph, steps_from="")
        if ph.sleep_from and ph.sleep_from not in {q.prefix for q in phenotypes if q.kind == "sleep_stages"}:
            notes.append(f"{ph.prefix}_sleep_hr and _sleep_dip skipped: their sleep-stage source {ph.sleep_from} is "
                         "not available")
            phenotypes[k] = replace(ph, sleep_from="")
    covariates = []
    for c in spec.covariates:
        if c.kind == "in_table" and (c.table, "person_id") not in have:
            notes.append(f"covariate {c.name} skipped: no table {c.table} with person_id in this CDR")
            continue
        covariates.append(c)
    fitted = replace(spec, wearables=wearables, phenotypes=phenotypes, covariates=covariates)
    fitted.positive_controls = [c for c in spec.positive_controls if c in _base_columns(fitted)
                                or c in {cv.name for cv in covariates}]
    return fitted, notes


def _bq_params(params: list[QueryParam]) -> list[Any]:
    from google.cloud import bigquery

    out = []
    for name, typ, value in params:
        if typ.startswith("ARRAY<"):
            out.append(bigquery.ArrayQueryParameter(name, typ[6:-1], value))
        else:
            out.append(bigquery.ScalarQueryParameter(name, typ, value))
    return out


# Every BigQuery job this process ran: bytes billed and seconds, for the job log and the run summary.
QUERY_LOG: list[dict[str, Any]] = []
_LOG_BILLED_FROM = 10 * 10**9  # jobs billing at least this many bytes are printed to the job log
USD_PER_TIB = float(os.getenv("CODAS_BQ_USD_PER_TIB", "6.25"))  # on-demand list price; set yours


# Cost guard: every job is dry-run first. One that would bill more than a job's cap, or push this process's
# total past the run's cap, is refused before it runs; the real job also carries BigQuery's own
# maximum_bytes_billed, so the cap holds even if the estimate was low.
MAX_BYTES_PER_JOB = int(float(os.getenv("CODAS_BQ_MAX_TB_PER_JOB", "16")) * 10**12)
MAX_BYTES_PER_RUN = int(float(os.getenv("CODAS_BQ_MAX_TB_PER_RUN", "40")) * 10**12)


class CostLimitError(RuntimeError):
    """A BigQuery job was refused because it would exceed the per-job or per-run byte cap."""


def _job_config(params: list[QueryParam], *, dry_run: bool = False, max_bytes: int | None = None) -> Any:
    from google.cloud import bigquery

    options: dict[str, Any] = {"query_parameters": _bq_params(params)}
    if dry_run:
        options.update(dry_run=True, use_query_cache=False)
    if max_bytes:
        options["maximum_bytes_billed"] = int(max_bytes)
    return bigquery.QueryJobConfig(**options)


def billed_text(n_bytes: int, verb: str = "billed") -> str:
    """Bytes as TB with the cost at ``USD_PER_TIB`` (an estimate: the workspace bill is the truth)."""
    return f"{n_bytes / 10**12:.2f} TB {verb} (about ${n_bytes / 2**40 * USD_PER_TIB:,.0f} at ${USD_PER_TIB}/TiB)"


def _estimate(client: Any, sql: str, params: list[QueryParam]) -> int | None:
    """Bytes the job would process (a free dry run), or None when the dry run itself fails."""
    try:
        dry = client.query(sql, job_config=_job_config(params, dry_run=True))
        return int(getattr(dry, "total_bytes_processed", None) or 0)
    except Exception:  # noqa: BLE001 - the real job then raises its own, more useful error
        return None


def _query(client: Any, sql: str, params: list[QueryParam]):
    start = time.monotonic()
    estimate = _estimate(client, sql, params)
    spent = sum(e["bytes_billed"] for e in QUERY_LOG)
    if estimate is not None and estimate > MAX_BYTES_PER_JOB:
        raise CostLimitError(f"BigQuery job refused: it would process {billed_text(estimate, 'processed')}, above "
                             f"the per-job cap of {MAX_BYTES_PER_JOB / 10**12:g} TB (CODAS_BQ_MAX_TB_PER_JOB).")
    if estimate is not None and spent + estimate > MAX_BYTES_PER_RUN:
        raise CostLimitError(f"BigQuery job refused: this run has billed {spent / 10**12:.2f} TB and the job would add "
                             f"{estimate / 10**12:.2f} TB, above the per-run cap of {MAX_BYTES_PER_RUN / 10**12:g} TB "
                             "(CODAS_BQ_MAX_TB_PER_RUN).")
    if estimate is not None and estimate >= _LOG_BILLED_FROM:
        print(f"BigQuery job starting: {billed_text(estimate, 'to process')}", flush=True)
    job = client.query(sql, job_config=_job_config(params, max_bytes=MAX_BYTES_PER_JOB))
    frame = job.result().to_dataframe()
    billed = int(getattr(job, "total_bytes_billed", None) or 0)
    QUERY_LOG.append({"bytes_billed": billed, "seconds": round(time.monotonic() - start, 1)})
    if billed >= _LOG_BILLED_FROM:
        print(f"BigQuery job: {billed_text(billed)}, {QUERY_LOG[-1]['seconds']:.0f} s", flush=True)
    return frame


def _phenotype_table(client: Any, spec: CohortSpec, cdr: str, out: Path):
    """The phenotypes of ``build_phenotype_sql``, computed once per CDR and phenotype definition and kept
    in the git-ignored runs folder (participant-level, like the cohort tables) for the other outcomes."""
    import hashlib

    import pandas as pd

    sql, params = build_phenotype_sql(spec, cdr)
    key = hashlib.sha256(json.dumps([cdr, sql, params], default=str).encode()).hexdigest()[:16]
    path = sealed_dir(out) / "phenotype_cache" / f"phenotypes_{key}.csv"
    if path.exists():
        return pd.read_csv(path, float_precision="round_trip")
    frame = _query(client, sql, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)  # complete or absent, never half-written
    return pd.read_csv(path, float_precision="round_trip")


def _check_out_dir(out_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    runs = (ROOT / ".codas_runs").resolve()
    if out_dir != runs and runs not in out_dir.parents:
        raise ValueError(f"Write the participant-level table under {runs} (git-ignored and readable by "
                         f"the CoDaS tools), not {out_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_cohort(
    spec: CohortSpec,
    cdr: str | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    client: Any = None,
    check_schema: bool = True,
) -> Path:
    """Run the extraction in BigQuery and write ``<out_dir>/<spec.name>.csv`` on the workspace disk.

    With ``holdout_fraction`` the sealed share goes to ``holdout_path(...)`` instead, and the returned
    table (the one the agents analyse) holds only the discovery share. Negative-control columns are
    written to the sealed table only."""
    spec.validate()
    cdr = resolve_cdr(cdr)
    out = _check_out_dir(Path(out_dir))
    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)

    if check_schema:
        sql, params = schema_check_sql(spec, cdr)
        present = [(r["table_name"], r["column_name"]) for _, r in _query(client, sql, params).iterrows()]
        spec, _skipped = resolve_sources(spec, present)
        missing = missing_columns(spec, present)
        if missing:
            raise ValueError("The CDR lacks columns this spec needs: " + ", ".join(missing)
                             + ". Check the table schema in the Workbench and edit the spec.")

    shared = (spec.design == "incident" and bool(spec.phenotypes)
              and os.getenv("CODAS_PHENOTYPE_CACHE", "1") != "0")
    days = spec.day_store is not None and spec.design == "incident"
    frame = _query(client, *build_cohort_sql(spec, cdr, phenotypes=not shared, qc=True, person_id=days))
    if shared:  # the minute-level tables are read once per run, not once per outcome
        frame = frame.merge(_phenotype_table(client, spec, cdr, out), on="person_id", how="left",
                            validate="many_to_one")
    people = frame.pop("person_id") if "person_id" in frame.columns else None  # never written to a cohort table
    path = out / f"{spec.name}.csv"
    sealed_people = None
    if spec.holdout_fraction > 0:
        sealed = frame[HOLDOUT_COLUMN].astype(bool)
        holdout_path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.loc[sealed, holdout_columns(spec)].to_csv(holdout_path(path), index=False)
        if people is not None:
            sealed_people, people = people[sealed], people[~sealed]
        frame = frame.loc[~sealed]
    frame[output_columns(spec)].to_csv(path, index=False)
    if days and people is not None:  # each share's days, keyed by its rows, from the day store built once per CDR
        from codas.aou import daystore

        daystore.split(daystore.store(client, spec, cdr, out), people, sealed_people, path)
    # Aggregate and masked, from the discovery share only.
    qc_path(path).write_text(json.dumps(extraction_qc(frame, spec), indent=1), encoding="utf-8")
    if has_panel(spec):
        panel = _query(client, *build_panel_sql(spec, cdr))
        if spec.holdout_fraction > 0:
            sealed = panel[HOLDOUT_COLUMN].astype(bool)
            panel.loc[sealed, panel_columns(spec)].to_csv(holdout_path(panel_path(path)), index=False)
            panel = panel.loc[~sealed]
        panel[panel_columns(spec)].to_csv(panel_path(path), index=False)
    return path


SEALED_DIR = "sealed"  # sealed shares and every cache holding all participants; no agent tool may open it


def sealed_dir(out_dir: str | Path) -> Path:
    """Where the sealed share and the all-participant caches of a run folder live."""
    return Path(out_dir) / SEALED_DIR


def holdout_path(discovery_csv: str | Path) -> Path:
    """The sealed replication table of a discovery table (``sealed/<name>.holdout.csv`` beside it)."""
    p = Path(discovery_csv)
    return sealed_dir(p.parent) / (p.stem + ".holdout.csv")
