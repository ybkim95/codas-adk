"""Build a one-row-per-participant CoDaS analysis table from the All of Us CDR (OMOP, BigQuery).

A ``CohortSpec`` names an outcome (a lab measurement or a condition, by OMOP concept id) and the
wearable tables to summarise. For each participant the extractor fixes an index date, summarises
every wearable column over the ``window_days`` before it (mean, SD, and the number of days worn),
adds age and sex at birth, and writes the table to the workspace disk under ``.codas_runs/aou/``.

Index date:

* ``measurement`` outcome: the participant's first qualifying measurement that has wearable data in
  the window before it. The outcome is ``value_as_number``.
* ``condition`` outcome: the first diagnosis date for cases and the last wearable date for
  everyone else; the outcome is 1 for cases and 0 otherwise. Cases and controls get index dates by
  different rules, so treat associations as hypothesis-generating.

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

    def validate(self) -> None:
        if self.kind not in {"measurement", "condition"}:
            raise ValueError(f"outcome.kind must be 'measurement' or 'condition', not {self.kind!r}.")
        if not self.concept_ids or not all(isinstance(c, int) and not isinstance(c, bool) for c in self.concept_ids):
            raise ValueError("outcome.concept_ids must be a non-empty list of integer OMOP concept ids.")
        _ident(self.name, "outcome name")


@dataclass
class CohortSpec:
    name: str
    outcome: OutcomeSpec
    wearables: list[WearableSource]
    window_days: int = 180
    min_wear_days: int = 30
    include_demographics: bool = True
    notes: str = ""

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CohortSpec":
        spec = cls(
            name=data["name"],
            outcome=OutcomeSpec(**data["outcome"]),
            wearables=[WearableSource(**w) for w in data["wearables"]],
            window_days=int(data.get("window_days", 180)),
            min_wear_days=int(data.get("min_wear_days", 30)),
            include_demographics=bool(data.get("include_demographics", True)),
            notes=str(data.get("notes", "")),
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


def output_columns(spec: CohortSpec) -> list[str]:
    """The columns of the extracted table, in order."""
    cols = [spec.outcome.name]
    if spec.include_demographics:
        cols += ["age", "female"]
    for source in spec.wearables:
        cols.append(f"{source.prefix}_days")
        for column in source.columns:
            cols += [f"{source.prefix}_{column}_mean", f"{source.prefix}_{column}_sd"]
    return cols


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
    if spec.outcome.kind == "measurement":
        bounds = ""
        if spec.outcome.value_min is not None:
            bounds += "\n    AND m.value_as_number >= @value_min"
            params.append(("value_min", "FLOAT64", float(spec.outcome.value_min)))
        if spec.outcome.value_max is not None:
            bounds += "\n    AND m.value_as_number <= @value_max"
            params.append(("value_max", "FLOAT64", float(spec.outcome.value_max)))
        ctes.append(
            "outcome_events AS (\n"
            "  SELECT m.person_id, m.measurement_date AS index_date, m.value_as_number AS outcome_value,\n"
            "    ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date, m.measurement_id) AS rn\n"
            f"  FROM {t('measurement')} m\n"
            "  JOIN outcome_concepts oc ON m.measurement_concept_id = oc.concept_id\n"
            f"  WHERE m.value_as_number IS NOT NULL{bounds}\n"
            f"    AND EXISTS (SELECT 1 FROM {t(primary.table)} w WHERE w.person_id = m.person_id\n"
            f"      AND DATE(w.{primary.date_column}) >= DATE_SUB(m.measurement_date, INTERVAL @window_days DAY)\n"
            f"      AND DATE(w.{primary.date_column}) < m.measurement_date)\n"
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
            f"  SELECT person_id, MAX(DATE({primary.date_column})) AS last_wear_date\n"
            f"  FROM {t(primary.table)}\n"
            "  GROUP BY person_id\n"
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
        filt = ""
        if source.filter_column is not None:
            name = f"w{i}_filter"
            # Compare as lowercase text: CDR flag columns differ in type between releases (the
            # Registered Tier v9 stores is_main_sleep as STRING 'true'/'false', not BOOL).
            filt = (f"\n    AND LOWER(CAST(s.{source.filter_column} AS STRING))"
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

    select = [f"ie.outcome_value AS {spec.outcome.name}"]
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
    tables = sorted({s.table for s in spec.wearables})
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
    """Run the extraction in BigQuery and write ``<out_dir>/<spec.name>.csv`` on the workspace disk."""
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
    frame = frame[output_columns(spec)]
    path = out / f"{spec.name}.csv"
    frame.to_csv(path, index=False)
    return path
