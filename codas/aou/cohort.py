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

The table is participant-level data. It stays inside the Workbench: it carries no person_id or
dates, it is git-ignored, and CoDaS only ever shows the model aggregates computed from it.

The SQL builder is pure (no BigQuery import) so it can be unit-tested anywhere; ``extract_cohort``
imports ``google.cloud.bigquery`` only when it runs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def validate(self) -> None:
        if self.kind not in {"measurement", "condition"}:
            raise ValueError(f"outcome.kind must be 'measurement' or 'condition', not {self.kind!r}.")
        if not self.concept_ids or not all(isinstance(c, int) and not isinstance(c, bool) for c in self.concept_ids):
            raise ValueError("outcome.concept_ids must be a non-empty list of integer OMOP concept ids.")
        _ident(self.name, "outcome name")


_COVARIATE_TABLES = {
    # kind: (table, concept column, date column)
    "measurement": ("measurement", "measurement_concept_id", "measurement_date"),
    "condition": ("condition_occurrence", "condition_concept_id", "condition_start_date"),
    "drug": ("drug_exposure", "drug_concept_id", "drug_exposure_start_date"),
}


@dataclass
class CovariateSpec:
    """A clinical covariate extracted alongside the outcome, for adjustment rather than discovery."""

    name: str
    kind: str  # "measurement", "condition" or "drug"
    concept_ids: list[int]
    include_descendants: bool = True
    # Days before the index date to look back; None looks back without limit.
    lookback_days: int | None = 365
    value_min: float | None = None
    value_max: float | None = None
    # Text the concept name must contain, checked against the CDR concept table before a run.
    expect_name: str = ""

    def validate(self) -> None:
        _ident(self.name, "covariate name")
        if self.kind not in _COVARIATE_TABLES:
            raise ValueError(f"covariate {self.name!r}: kind must be one of {sorted(_COVARIATE_TABLES)}, "
                             f"not {self.kind!r}.")
        if not self.concept_ids or not all(isinstance(c, int) and not isinstance(c, bool) for c in self.concept_ids):
            raise ValueError(f"covariate {self.name!r}: concept_ids must be a non-empty list of integer OMOP concept ids.")
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

    def validate(self) -> None:
        _ident(self.name, "cohort name")
        self.outcome.validate()
        if not self.wearables:
            raise ValueError("A cohort needs at least one wearable source.")
        prefixes = set()
        for source in self.wearables:
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
        )
        spec.validate()
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
    return cols


def output_columns(spec: CohortSpec) -> list[str]:
    """The columns of the extracted table, in order."""
    return _base_columns(spec) + [c.name for c in spec.covariates]


FOLLOWUP_COLUMN = "followup_days"
HOLDOUT_COLUMN = "_holdout"
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
    """The columns of the sealed replication table: the analysis columns plus the negative controls."""
    return output_columns(spec) + negative_control_columns(spec)


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


def build_cohort_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """Return the extraction SQL and its parameters. Identifiers are validated, values are bound."""
    spec.validate()
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = [
        ("outcome_concept_ids", "ARRAY<INT64>", list(spec.outcome.concept_ids)),
        ("window_days", "INT64", int(spec.window_days)),
        ("min_wear_days", "INT64", int(spec.min_wear_days)),
    ]

    concepts = "SELECT concept_id FROM UNNEST(@outcome_concept_ids) AS concept_id"
    if spec.outcome.include_descendants:
        concepts += (f"\n  UNION DISTINCT\n  SELECT descendant_concept_id FROM {t('concept_ancestor')}"
                     "\n  WHERE ancestor_concept_id IN UNNEST(@outcome_concept_ids)")
    ctes = [f"outcome_concepts AS (\n  {concepts}\n)"]

    primary = spec.wearables[0]
    valid = ""  # optional valid-day rule on the primary wearable's rows, aliased as {a}
    if spec.valid_day is not None:
        column = spec.valid_day["column"]
        if spec.valid_day.get("min") is not None:
            valid += f"\n      AND {{a}}.{column} >= @valid_min"
            params.append(("valid_min", "FLOAT64", float(spec.valid_day["min"])))
        if spec.valid_day.get("max") is not None:
            valid += f"\n      AND {{a}}.{column} <= @valid_max"
            params.append(("valid_max", "FLOAT64", float(spec.valid_day["max"])))
    bounds = ""
    if spec.outcome.kind == "measurement":
        if spec.outcome.value_min is not None:
            bounds += "\n    AND m.value_as_number >= @value_min"
            params.append(("value_min", "FLOAT64", float(spec.outcome.value_min)))
        if spec.outcome.value_max is not None:
            bounds += "\n    AND m.value_as_number <= @value_max"
            params.append(("value_max", "FLOAT64", float(spec.outcome.value_max)))

    if spec.design == "incident":
        # One index rule for everyone: the end of a baseline window that starts at the first valid
        # wear day. Outcomes are observed from index + lag; earlier ones exclude the person.
        params.append(("outcome_from", "INT64", int(spec.window_days + spec.lag_days)))
        ctes.append(
            "first_wear AS (\n"
            f"  SELECT w.person_id, MIN(DATE(w.{primary.date_column})) AS first_wear\n"
            f"  FROM {t(primary.table)} w\n"
            f"  WHERE TRUE{valid.format(a='w')}\n"
            "  GROUP BY w.person_id\n"
            ")"
        )
        if spec.outcome.kind == "measurement":
            horizon = ""
            if spec.horizon_days is not None:
                params.append(("outcome_until", "INT64", int(spec.window_days + spec.lag_days + spec.horizon_days)))
                horizon = "\n    AND m.measurement_date < DATE_ADD(f.first_wear, INTERVAL @outcome_until DAY)"
            ctes.append(
                "outcome_events AS (\n"
                "  SELECT m.person_id, f.first_wear, m.value_as_number AS outcome_value,\n"
                "    ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date, m.measurement_id) AS rn\n"
                f"  FROM {t('measurement')} m\n"
                "  JOIN outcome_concepts oc ON m.measurement_concept_id = oc.concept_id\n"
                "  JOIN first_wear f ON f.person_id = m.person_id\n"
                f"  WHERE m.value_as_number IS NOT NULL{bounds}\n"
                f"    AND m.measurement_date >= DATE_ADD(f.first_wear, INTERVAL @outcome_from DAY){horizon}\n"
                ")"
            )
            ctes.append("index_events AS (\n  SELECT person_id, DATE_ADD(first_wear, INTERVAL @window_days DAY) AS index_date,"
                        " outcome_value\n  FROM outcome_events WHERE rn = 1\n)")
            if spec.negative_controls:
                ctes.append(_LAST_SEEN.format(visits=t("visit_occurrence")))
        else:
            ctes.append(
                "cases AS (\n"
                "  SELECT co.person_id, MIN(co.condition_start_date) AS dx_date\n"
                f"  FROM {t('condition_occurrence')} co\n"
                "  JOIN outcome_concepts oc ON co.condition_concept_id = oc.concept_id\n"
                "  GROUP BY co.person_id\n"
                ")"
            )
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
                f"  WHERE (c.dx_date IS NULL OR c.dx_date >= {start_fu})\n"
                f"    AND COALESCE(c.dx_date, l.last_seen) > {start_fu}\n"
                ")"
            )
    elif spec.outcome.kind == "measurement":
        ctes.append(
            "outcome_events AS (\n"
            "  SELECT m.person_id, m.measurement_date AS index_date, m.value_as_number AS outcome_value,\n"
            "    ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date, m.measurement_id) AS rn\n"
            f"  FROM {t('measurement')} m\n"
            "  JOIN outcome_concepts oc ON m.measurement_concept_id = oc.concept_id\n"
            f"  WHERE m.value_as_number IS NOT NULL{bounds}\n"
            f"    AND EXISTS (SELECT 1 FROM {t(primary.table)} w WHERE w.person_id = m.person_id\n"
            f"      AND DATE(w.{primary.date_column}) >= DATE_SUB(m.measurement_date, INTERVAL @window_days DAY)\n"
            f"      AND DATE(w.{primary.date_column}) < m.measurement_date{valid.format(a='w')})\n"
            ")"
        )
        ctes.append("index_events AS (\n  SELECT person_id, index_date, outcome_value FROM outcome_events WHERE rn = 1\n)")
    else:
        ctes.append(
            "cases AS (\n"
            "  SELECT co.person_id, MIN(co.condition_start_date) AS dx_date\n"
            f"  FROM {t('condition_occurrence')} co\n"
            "  JOIN outcome_concepts oc ON co.condition_concept_id = oc.concept_id\n"
            "  GROUP BY co.person_id\n"
            ")"
        )
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
            ")"
        )

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
        else:
            ctes.append(
                f"c{i} AS (\n"
                "  SELECT DISTINCT ie.person_id\n"
                + join +
                f"  WHERE {where}\n"
                ")"
            )

    select = [f"ie.outcome_value AS {spec.outcome.name}"]
    if is_survival(spec):
        select.append(f"ie.{FOLLOWUP_COLUMN}")
    joins = []
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
        joins.append(f"LEFT JOIN w{i} ON w{i}.person_id = ie.person_id")
    for i, covariate in enumerate(spec.covariates):
        if covariate.kind == "measurement":
            select.append(f"c{i}.{covariate.name}")
        else:
            select.append(f"IF(c{i}.person_id IS NULL, 0, 1) AS {covariate.name}")
        joins.append(f"LEFT JOIN c{i} ON c{i}.person_id = ie.person_id")
    if spec.negative_controls:
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
            ctes.append(
                f"n{i} AS (\n"
                "  SELECT co.person_id, MIN(co.condition_start_date) AS dx_date\n"
                f"  FROM {t('condition_occurrence')} co\n"
                f"  JOIN (\n    {concepts}\n  ) nc ON co.condition_concept_id = nc.concept_id\n"
                "  GROUP BY co.person_id\n"
                ")"
            )
            col = f"{NC_PREFIX}{control.name}"
            prevalent = f"n{i}.dx_date < {start}"
            select += [f"CASE WHEN {prevalent} THEN NULL WHEN n{i}.dx_date IS NULL THEN 0 ELSE 1 END AS {col}",
                       f"CASE WHEN {prevalent} THEN NULL ELSE DATE_DIFF(COALESCE(n{i}.dx_date, ls.last_seen), {start}, DAY)"
                       f" END AS {col}_days"]
            joins.append(f"LEFT JOIN n{i} ON n{i}.person_id = ie.person_id")
    if spec.holdout_fraction > 0:
        # Deterministic, roughly uniform split on the person id, computed apart from the selected
        # columns so the output carries only the flag; extract_cohort drops it before the agents see rows.
        params += [("holdout_permille", "INT64", int(round(spec.holdout_fraction * 1000))),
                   ("split_salt", "INT64", int(spec.split_salt))]
        ctes.append(f"split AS (\n  SELECT person_id, {_SPLIT_SQL} < @holdout_permille AS {HOLDOUT_COLUMN}\n"
                    "  FROM index_events\n)")
        select.append(f"sp.{HOLDOUT_COLUMN}")
        joins.append("JOIN split sp ON sp.person_id = ie.person_id")

    sql = (
        "WITH " + ",\n".join(ctes) + "\n"
        "SELECT\n  " + ",\n  ".join(select) + "\n"
        "FROM index_events ie\n" + "\n".join(joins) + "\n"
        f"WHERE w0.{primary.prefix}_days >= @min_wear_days"
    )
    return sql, params


def schema_check_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """SQL listing the (table, column) pairs the spec needs that exist in the CDR."""
    cdr = resolve_cdr(cdr)
    tables = sorted({s.table for s in spec.wearables} | {_COVARIATE_TABLES[c.kind][0] for c in spec.covariates}
                    | ({"visit_occurrence"} if is_survival(spec) or spec.negative_controls else set())
                    | ({"condition_occurrence"} if spec.negative_controls else set()))
    sql = (f"SELECT table_name, column_name FROM `{cdr}.INFORMATION_SCHEMA.COLUMNS`\n"
           "WHERE table_name IN UNNEST(@tables)")
    return sql, [("tables", "ARRAY<STRING>", tables)]


def missing_columns(spec: CohortSpec, present: list[tuple[str, str]]) -> list[str]:
    """``table.column`` names the spec needs but the CDR does not have."""
    have = {(str(tbl), str(col)) for tbl, col in present}
    needed = []
    for s in spec.wearables:
        cols = [s.date_column, "person_id", *s.columns] + ([s.filter_column] if s.filter_column else [])
        needed += [(s.table, c) for c in cols]
    for c in spec.covariates:
        table, concept_col, date_col = _COVARIATE_TABLES[c.kind]
        needed += [(table, col) for col in ("person_id", concept_col, date_col)]
        if c.kind == "measurement":
            needed += [(table, "value_as_number"), (table, "measurement_id")]
    if is_survival(spec) or spec.negative_controls:
        needed += [("visit_occurrence", "person_id"), ("visit_occurrence", "visit_start_date")]
    if spec.negative_controls:
        needed += [("condition_occurrence", c) for c in ("person_id", "condition_concept_id", "condition_start_date")]
    return [f"{tbl}.{col}" for tbl, col in dict.fromkeys(needed) if (tbl, col) not in have]


def _bq_params(params: list[QueryParam]) -> list[Any]:
    from google.cloud import bigquery

    out = []
    for name, typ, value in params:
        if typ.startswith("ARRAY<"):
            out.append(bigquery.ArrayQueryParameter(name, typ[6:-1], value))
        else:
            out.append(bigquery.ScalarQueryParameter(name, typ, value))
    return out


def _query(client: Any, sql: str, params: list[QueryParam]):
    from google.cloud import bigquery

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=_bq_params(params)))
    return job.result().to_dataframe()


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
        missing = missing_columns(spec, present)
        if missing:
            raise ValueError("The CDR lacks columns this spec needs: " + ", ".join(missing)
                             + ". Check the table schema in the Workbench and edit the spec.")

    sql, params = build_cohort_sql(spec, cdr)
    frame = _query(client, sql, params)
    path = out / f"{spec.name}.csv"
    if spec.holdout_fraction > 0:
        sealed = frame[HOLDOUT_COLUMN].astype(bool)
        frame.loc[sealed, holdout_columns(spec)].to_csv(holdout_path(path), index=False)
        frame = frame.loc[~sealed]
    frame[output_columns(spec)].to_csv(path, index=False)
    return path


def holdout_path(discovery_csv: str | Path) -> Path:
    """The sealed replication table next to a discovery table (``<name>.holdout.csv``)."""
    p = Path(discovery_csv)
    return p.with_name(p.stem + ".holdout.csv")
