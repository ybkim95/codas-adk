"""What this CDR holds, by schema and aggregate counts only: every table with its columns, how many
participants each person-level table covers and how many of them also have wearable data, and which
tables look like minute-level heart rate. No row is read; counts of 1-20 are masked.

The minute heart-rate table is found by its shape (a person id, a date-time column and a numeric
column whose name says heart rate or bpm), not by a fixed name, because table names differ between
data releases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from codas.aou import cohort
from codas.aou.cohort import _ident, resolve_cdr
from codas.aou.privacy import mask_small_counts

_TIME_TYPES = ("DATETIME", "TIMESTAMP")
_NUMERIC_TYPES = ("INT64", "INTEGER", "FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC", "DOUBLE", "BIGINT", "DECIMAL")
_HEART = re.compile(r"heart|bpm|pulse", re.I)


def schema(cdr: str, client: Any) -> pd.DataFrame:
    """(table_name, column_name, data_type) for every table of the CDR."""
    sql = (f"SELECT table_name, column_name, data_type FROM `{resolve_cdr(cdr)}.INFORMATION_SCHEMA.COLUMNS`\n"
           "ORDER BY table_name, column_name")
    return cohort._query(client, sql, [])


def minute_heart_rate_tables(columns: pd.DataFrame) -> list[dict[str, str]]:
    """Tables shaped like minute-level heart rate: person_id, a date-time column, and a numeric column
    named for heart rate (or a numeric value column in a table named for it)."""
    found = []
    for table, part in columns.groupby("table_name"):
        names = dict(zip(part["column_name"].str.lower(), part["data_type"].astype(str).str.upper()))
        if "person_id" not in names:
            continue
        times = [c for c, t in names.items() if t.startswith(_TIME_TYPES)]
        values = [c for c, t in names.items() if t.startswith(_NUMERIC_TYPES) and c != "person_id"
                  and (_HEART.search(c) or (_HEART.search(str(table)) and c in ("value", "heart_rate_value")))]
        if times and values:
            found.append({"table": str(table), "time": times[0], "value": values[0]})
    return found


def person_tables(columns: pd.DataFrame) -> list[str]:
    return sorted({str(t) for t, c in zip(columns["table_name"], columns["column_name"]) if str(c).lower() == "person_id"})


def coverage_sql(cdr: str, tables: list[str], wearable_table: str = "activity_summary") -> str:
    """Participants per person-level table, and how many of them have rows in ``wearable_table``."""
    cdr = resolve_cdr(cdr)
    parts = [f"SELECT '{_ident(t, 'table')}' AS table_name, COUNT(DISTINCT x.person_id) AS participants,\n"
             f"  COUNT(DISTINCT w.person_id) AS wearable_participants\n"
             f"FROM `{cdr}.{t}` x LEFT JOIN (SELECT DISTINCT person_id FROM `{cdr}.{_ident(wearable_table, 'table')}`) w\n"
             "  ON w.person_id = x.person_id" for t in tables]
    return "\nUNION ALL\n".join(parts)


def inventory(cdr: str, client: Any, wearable_table: str = "activity_summary",
              cache_dir: str | Path | None = None) -> dict[str, Any]:
    """The CDR's tables (with column counts and, for person-level tables, masked participant counts)
    and the minute heart-rate tables found by shape. A CDR release never changes, so with ``cache_dir`` the
    coverage counts (a scan of every person-level table) are computed once per CDR and table list."""
    columns = schema(cdr, client)
    tables = person_tables(columns)
    cache = Path(cache_dir) / f"{resolve_cdr(cdr)}.{_ident(wearable_table, 'table')}.json" if cache_dir else None
    counts = None
    if cache is not None and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("tables") == tables:
            counts = pd.DataFrame(saved["counts"])
    if counts is None:
        counts = cohort._query(client, coverage_sql(cdr, tables, wearable_table), []) if tables else pd.DataFrame()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"tables": tables, "counts": counts.to_dict("records")}, default=str),
                             encoding="utf-8")
    summary = columns.groupby("table_name").size().rename("columns").reset_index()
    if not counts.empty:
        summary = summary.merge(counts, on="table_name", how="left")
    records = mask_small_counts(summary.to_dict("records"))
    return {"tables": pd.DataFrame(records), "minute_heart_rate": minute_heart_rate_tables(columns),
            "columns": columns}
