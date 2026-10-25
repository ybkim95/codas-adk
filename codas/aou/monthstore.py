"""Each participant's daily wearable sources over the whole time they wore the device, in blocks of
``block_days`` (30 by default) counted from the index date: the table a feature is followed over time on
(``codas.core.trajectory``).

One row per person and block with a valid day of the primary source: the block (days from the index date divided
by ``block_days``, rounded down; negative before the index), the valid days in it (``days``), and each daily
source column's mean over the block (``<prefix>_<column>``, the days a source's value bounds keep). It reads only
the daily tables, once per CDR and definition, and is kept in the sealed folder since it holds ``person_id`` and
every participant. Per outcome, ``split`` writes the two shares keyed by the cohort table's row, without dates or
identifiers: the discovery share beside the cohort table, the sealed share beside the sealed table.

A registered feature that is a daily column's mean over the baseline (``<prefix>_<column>_mean``) is followed in
its block column (``column_of``); the holdout then reads it as a time-varying exposure and before each diagnosis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from codas.aou import cohort

BLOCK_DAYS = 30


def block_days(spec: "cohort.CohortSpec") -> int:
    return int((spec.month_store or {}).get("block_days") or BLOCK_DAYS)


def block_columns(spec: "cohort.CohortSpec") -> list[str]:
    """The value columns of the store, in order."""
    return [f"{w.prefix}_{c}" for w in spec.wearables for c in w.columns]


def column_of(feature: str, columns: list[str]) -> str | None:
    """The block column a baseline feature follows over time: ``<prefix>_<column>_mean`` -> ``<prefix>_<column>``."""
    return feature[: -len("_mean")] if feature.endswith("_mean") and feature[: -len("_mean")] in columns else None


def build_monthstore_sql(spec: "cohort.CohortSpec", cdr: str) -> tuple[str, list[cohort.QueryParam]]:
    """The store's SQL: person_id, block, days and ``block_columns``."""
    spec.validate()
    if spec.design != "incident" or spec.month_store is None:
        raise ValueError("The month store needs the incident design and a month_store section.")
    cdr = cohort.resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    primary = spec.wearables[0]
    params: list[cohort.QueryParam] = [("window_days", "INT64", int(spec.window_days)),
                                       ("block_days", "INT64", block_days(spec))]
    valid = cohort._valid_rule(spec, params)
    primary_filter = ""
    if primary.filter_column is not None:
        params.append(("w0_filter", cohort._scalar_type(primary.filter_value), primary.filter_value))
        primary_filter = f"\n    AND LOWER(CAST(s.{primary.filter_column} AS STRING)) = LOWER(CAST(@w0_filter AS STRING))"
    block = "CAST(FLOOR(DATE_DIFF(DATE(s.{c}), ie.index_date, DAY) / @block_days) AS INT64)"  # also before index
    ctes = [cohort._first_wear_cte(primary, t, valid),
            "index_events AS (\n  SELECT person_id, DATE_ADD(first_wear_date, INTERVAL @window_days DAY) AS index_date\n"
            "  FROM first_wear\n)",
            "blocks AS (\n"
            f"  SELECT ie.person_id, {block.format(c=primary.date_column)} AS block,\n"
            f"    COUNT(DISTINCT DATE(s.{primary.date_column})) AS days\n"
            "  FROM index_events ie\n"
            f"  JOIN {t(primary.table)} s ON s.person_id = ie.person_id\n"
            f"  WHERE TRUE{valid.format(a='s')}{primary_filter}\n"
            "  GROUP BY ie.person_id, block\n)"]
    select, joins = [], []
    for i, source in enumerate(spec.wearables):
        filt = valid.format(a="s") + primary_filter if i == 0 else ""
        if i and source.filter_column is not None:
            name = f"mw{i}_filter"
            filt += f"\n    AND LOWER(CAST(s.{source.filter_column} AS STRING)) = LOWER(CAST(@{name} AS STRING))"
            params.append((name, cohort._scalar_type(source.filter_value), source.filter_value))
        filt += cohort._bounds_rule(source, "s", f"mw{i}b", params)
        values = ",\n    ".join(f"AVG(s.{c}) AS {source.prefix}_{c}" for c in source.columns)
        ctes.append(f"mw{i} AS (\n"
                    f"  SELECT ie.person_id, {block.format(c=source.date_column)} AS block,\n    {values}\n"
                    "  FROM index_events ie\n"
                    f"  JOIN {t(source.table)} s ON s.person_id = ie.person_id\n"
                    f"  WHERE TRUE{filt}\n"
                    "  GROUP BY ie.person_id, block\n)")
        select += [f"mw{i}.{source.prefix}_{c}" for c in source.columns]
        joins.append(f"LEFT JOIN mw{i} ON mw{i}.person_id = b.person_id AND mw{i}.block = b.block")
    sql = ("WITH " + ",\n".join(ctes) + "\n"
           "SELECT b.person_id, b.block, b.days,\n  " + ",\n  ".join(select) + "\n"
           "FROM blocks b\n" + "\n".join(joins))
    return sql, params


def store_path(out: Path, spec: "cohort.CohortSpec", cdr: str) -> Path:
    sql, params = build_monthstore_sql(spec, cdr)
    key = hashlib.sha256(json.dumps([cohort.resolve_cdr(cdr), sql, params], default=str).encode()).hexdigest()[:16]
    return cohort.sealed_dir(out) / "month_store" / f"months_{key}.csv.gz"


def store(client: Any, spec: "cohort.CohortSpec", cdr: str, out: Path) -> pd.DataFrame:
    """The store, from BigQuery the first time (dry-run first, under the cost caps) and from disk after."""
    path = store_path(out, spec, cdr)
    if not path.exists():
        frame = cohort._query(client, *build_monthstore_sql(spec, cdr))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        frame.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)  # complete or absent, never half-written
    return pd.read_csv(path, float_precision="round_trip")


def months_path(discovery_csv: str | Path) -> Path:
    """The discovery share's block table, beside the cohort table (``<name>.months.csv.gz``)."""
    p = Path(discovery_csv)
    return p.with_name(p.stem + ".months.csv.gz")


def holdout_months_path(discovery_csv: str | Path) -> Path:
    """The sealed share's block table, beside the sealed table."""
    p = Path(discovery_csv)
    return cohort.sealed_dir(p.parent) / (p.stem + ".months.holdout.csv.gz")


def _keyed(blocks: pd.DataFrame, people: pd.Series) -> pd.DataFrame:
    row_of = pd.Series(range(len(people)), index=people.to_numpy())
    rows = blocks[blocks["person_id"].isin(row_of.index)]
    keyed = rows.assign(row=row_of.reindex(rows["person_id"]).to_numpy()).drop(columns="person_id")
    return keyed[["row", *[c for c in keyed.columns if c != "row"]]].sort_values(["row", "block"])


def split(blocks: pd.DataFrame, discovery_people: pd.Series, sealed_people: pd.Series | None,
          discovery_csv: str | Path) -> Path:
    """Write the outcome's block tables (the sealed one in the sealed folder); return the discovery path."""
    path = months_path(discovery_csv)
    _keyed(blocks, discovery_people).to_csv(path, index=False, compression="gzip")
    if sealed_people is not None:
        holdout = holdout_months_path(discovery_csv)
        holdout.parent.mkdir(parents=True, exist_ok=True)
        _keyed(blocks, sealed_people).to_csv(holdout, index=False, compression="gzip")
    return path
