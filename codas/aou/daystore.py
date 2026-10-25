"""Each participant's baseline window, day by day: the table new phenotypes are defined on.

One row per person and valid day of the primary wearable in the baseline window (incident design, where the
window is a property of the person, not of the outcome):

* every daily source's columns for that day (``<prefix>_<column>``; a day outside a source's value bounds,
  or missing from it, is empty for that source);
* from the minute-level sources the spec's ``day_store`` names (``{"from": [prefixes]}``):
  - a minute heart-rate source: the day's resting heart rate (mean of its 30 lowest minutes),
    ``<prefix>_day_rest``, its mean, ``<prefix>_day_mean``, and its heart-rate minutes,
    ``<prefix>_day_minutes`` (only days with the source's minimum minutes, when it sets one);
  - a sleep-stage source: the main-sleep night ending on that date, ``<prefix>_onset`` and
    ``<prefix>_midpoint`` (clock hours after noon), ``<prefix>_asleep`` (minutes) and ``<prefix>_wake_bouts``
    (only nights the source counts);
  - a minute-activity source: ``<prefix>_active_min`` (minutes with steps), ``<prefix>_brisk_min`` (100 or
    more a minute, moderate intensity), ``<prefix>_peak30`` (mean of the 30 highest minutes) and the clock
    hours of the first and last active minute, ``<prefix>_first_active`` and ``<prefix>_last_active``.

It is computed once per CDR and definition in BigQuery and kept in the sealed folder, since it holds
``person_id`` and every participant. Per outcome, ``split`` writes the outcome's two shares keyed by the row
of the cohort table (``row``), with days counted from the index date (``day``, negative) and the weekday
(``weekday``, 1 = Sunday ... 7 = Saturday), and no dates or identifiers: the discovery share beside the
cohort table, where the agents' tools read it, the sealed share beside the sealed table. Phenotypes defined on
it (``codas.core.phenotype_dsl``) become columns of the cohort tables (``phenotype_dsl.add_columns``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from codas.aou import cohort
from codas.aou.phenotypes import PhenotypeSource

KINDS = ("minute_heart_rate", "sleep_stages", "minute_activity")


def sources(spec: "cohort.CohortSpec") -> list[PhenotypeSource]:
    """The minute-level sources the day store draws on (none without a ``day_store`` section)."""
    named = list((spec.day_store or {}).get("from") or [])
    return [p for p in spec.phenotypes if p.prefix in named and p.kind in KINDS]


def validate(spec: "cohort.CohortSpec") -> None:
    if spec.day_store is None:
        return
    if spec.design != "incident":
        raise ValueError("day_store needs the incident design: there the baseline window belongs to the person.")
    named = (spec.day_store or {}).get("from") or []
    known = {p.prefix: p.kind for p in spec.phenotypes}
    for prefix in named:
        if known.get(prefix) not in KINDS:
            raise ValueError(f"day_store.from: {prefix!r} is not a minute heart-rate, sleep-stage or minute-activity "
                             "phenotype source of this spec.")


def day_columns(spec: "cohort.CohortSpec") -> list[str]:
    """The day-level signal columns of the store, in order."""
    out = [f"{w.prefix}_{c}" for w in spec.wearables for c in w.columns]
    for p in sources(spec):
        out += {"minute_heart_rate": [f"{p.prefix}_day_rest", f"{p.prefix}_day_mean", f"{p.prefix}_day_minutes"],
                "sleep_stages": [f"{p.prefix}_onset", f"{p.prefix}_midpoint", f"{p.prefix}_asleep",
                                 f"{p.prefix}_wake_bouts"],
                "minute_activity": [f"{p.prefix}_active_min", f"{p.prefix}_brisk_min", f"{p.prefix}_peak30",
                                    f"{p.prefix}_first_active", f"{p.prefix}_last_active"]}[p.kind]
    return out


def build_daystore_sql(spec: "cohort.CohortSpec", cdr: str) -> tuple[str, list[cohort.QueryParam]]:
    """The day store's SQL: person_id, day (relative to the index date), weekday and ``day_columns``."""
    spec.validate()
    if spec.design != "incident" or spec.day_store is None:
        raise ValueError("The day store needs the incident design and a day_store section.")
    cdr = cohort.resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    primary = spec.wearables[0]
    params: list[cohort.QueryParam] = [("window_days", "INT64", int(spec.window_days))]
    valid = cohort._valid_rule(spec, params)
    if primary.filter_column is not None:
        params.append(("w0_filter", cohort._scalar_type(primary.filter_value), primary.filter_value))
    ctes = [cohort._first_wear_cte(primary, t, valid),
            "index_events AS (\n  SELECT person_id, DATE_ADD(first_wear_date, INTERVAL @window_days DAY) AS index_date\n"
            "  FROM first_wear\n)",
            "day_rows AS (\n"
            f"  SELECT DISTINCT ie.person_id, ie.index_date, DATE(s.{primary.date_column}) AS d\n"
            "  FROM index_events ie\n"
            f"  JOIN {t(primary.table)} s ON s.person_id = ie.person_id\n"
            f"    AND DATE(s.{primary.date_column}) >= DATE_SUB(ie.index_date, INTERVAL @window_days DAY)\n"
            f"    AND DATE(s.{primary.date_column}) < ie.index_date{valid.format(a='s')}"
            + (f"\n    AND LOWER(CAST(s.{primary.filter_column} AS STRING)) = LOWER(CAST(@w0_filter AS STRING))"
               if primary.filter_column is not None else "") + "\n)"]
    select, joins = [], []
    for i, source in enumerate(spec.wearables):
        filt = valid.format(a="s") if i == 0 else ""
        if source.filter_column is not None:
            name = f"dw{i}_filter"
            filt += (f"\n    AND LOWER(CAST(s.{source.filter_column} AS STRING)) = LOWER(CAST(@{name} AS STRING))")
            params.append((name, cohort._scalar_type(source.filter_value), source.filter_value))
        filt += cohort._bounds_rule(source, "s", f"dw{i}b", params)
        values = ",\n    ".join(f"AVG(s.{c}) AS {source.prefix}_{c}" for c in source.columns)
        ctes.append(
            f"dw{i} AS (\n"
            f"  SELECT ie.person_id, DATE(s.{source.date_column}) AS d,\n    {values}\n"
            "  FROM index_events ie\n"
            f"  JOIN {t(source.table)} s ON s.person_id = ie.person_id\n"
            f"    AND DATE(s.{source.date_column}) >= DATE_SUB(ie.index_date, INTERVAL @window_days DAY)\n"
            f"    AND DATE(s.{source.date_column}) < ie.index_date{filt}\n"
            f"  GROUP BY ie.person_id, DATE(s.{source.date_column})\n"
            ")")
        select += [f"dw{i}.{source.prefix}_{c}" for c in source.columns]
        joins.append(f"LEFT JOIN dw{i} ON dw{i}.person_id = r.person_id AND dw{i}.d = r.d")
    chosen = sources(spec)
    if chosen:
        extra, aliases = cohort._phenotype_parts(replace(spec, phenotypes=chosen), t, valid, params)
        ctes += extra
        for alias, ph in aliases:
            p = ph.prefix
            if ph.kind == "minute_heart_rate":
                ctes.append(
                    f"{alias}_day AS (\n"
                    f"  SELECT m.person_id, m.d, COUNT(*) AS {p}_day_minutes, AVG(m.hr) AS {p}_day_mean,\n"
                    f"    AVG(lw.low30) AS {p}_day_rest\n"
                    f"  FROM {alias}_min m JOIN {alias}_low lw ON lw.person_id = m.person_id AND lw.d = m.d\n"
                    "  GROUP BY m.person_id, m.d\n"
                    ")")
                select += [f"{alias}_day.{p}_day_rest", f"{alias}_day.{p}_day_mean", f"{alias}_day.{p}_day_minutes"]
                joins.append(f"LEFT JOIN {alias}_day ON {alias}_day.person_id = r.person_id AND {alias}_day.d = r.d")
            elif ph.kind == "sleep_stages":
                select += [f"{alias}_clock.onset_h AS {p}_onset", f"{alias}_clock.mid_h AS {p}_midpoint",
                           f"{alias}_clock.asleep_min AS {p}_asleep", f"{alias}_clock.wake_bouts AS {p}_wake_bouts"]
                joins.append(f"LEFT JOIN {alias}_clock ON {alias}_clock.person_id = r.person_id "
                             f"AND {alias}_clock.d = r.d")
            else:
                ctes.append(
                    f"{alias}_day AS (\n"
                    f"  SELECT person_id, d, SUM(IF(v > 0, 1, 0)) AS {p}_active_min,\n"
                    f"    SUM(IF(v >= 100, 1, 0)) AS {p}_brisk_min,\n"
                    f"    SUM(IF(rn <= 30, v, 0)) / 30 AS {p}_peak30,\n"
                    f"    MIN(IF(v > 0, minute_of_day, NULL)) / 60 AS {p}_first_active,\n"
                    f"    MAX(IF(v > 0, minute_of_day, NULL)) / 60 AS {p}_last_active\n"
                    "  FROM (\n"
                    "    SELECT person_id, d, minute_of_day, v,\n"
                    "      ROW_NUMBER() OVER (PARTITION BY person_id, d ORDER BY v DESC) AS rn\n"
                    f"    FROM {alias}_min\n"
                    "  ) GROUP BY person_id, d\n"
                    ")")
                select += [f"{alias}_day.{c}" for c in (f"{p}_active_min", f"{p}_brisk_min", f"{p}_peak30",
                                                        f"{p}_first_active", f"{p}_last_active")]
                joins.append(f"LEFT JOIN {alias}_day ON {alias}_day.person_id = r.person_id AND {alias}_day.d = r.d")
    sql = ("WITH " + ",\n".join(ctes) + "\n"
           "SELECT\n  r.person_id, DATE_DIFF(r.d, r.index_date, DAY) AS day,\n"
           "  MOD(DATE_DIFF(r.d, DATE '1900-01-07', DAY), 7) + 1 AS weekday,\n  "  # 1900-01-07 was a Sunday
           + ",\n  ".join(select) + "\n"
           "FROM day_rows r\n" + "\n".join(joins))
    return sql, params


def store_path(out: Path, spec: "cohort.CohortSpec", cdr: str) -> Path:
    """Where the day store of this CDR and definition is kept (sealed: every participant, with person_id)."""
    sql, params = build_daystore_sql(spec, cdr)
    key = hashlib.sha256(json.dumps([cohort.resolve_cdr(cdr), sql, params], default=str).encode()).hexdigest()[:16]
    return cohort.sealed_dir(out) / "day_store" / f"days_{key}.csv.gz"


def store(client: Any, spec: "cohort.CohortSpec", cdr: str, out: Path) -> pd.DataFrame:
    """The day store, from BigQuery the first time (dry-run first, under the cost caps) and from disk after."""
    path = store_path(out, spec, cdr)
    if not path.exists():
        frame = cohort._query(client, *build_daystore_sql(spec, cdr))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        frame.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)  # complete or absent, never half-written
    return pd.read_csv(path, float_precision="round_trip")


def days_path(discovery_csv: str | Path) -> Path:
    """The discovery share's day table, beside the cohort table (``<name>.days.csv.gz``)."""
    p = Path(discovery_csv)
    return p.with_name(p.stem + ".days.csv.gz")


def holdout_days_path(discovery_csv: str | Path) -> Path:
    """The sealed share's day table, beside the sealed table."""
    p = Path(discovery_csv)
    return cohort.sealed_dir(p.parent) / (p.stem + ".days.holdout.csv.gz")


def _keyed(days: pd.DataFrame, people: pd.Series) -> pd.DataFrame:
    """The day rows of ``people`` (person_id in cohort-table order), keyed by that order and without person_id."""
    row_of = pd.Series(range(len(people)), index=people.to_numpy())
    rows = days[days["person_id"].isin(row_of.index)]
    keyed = rows.assign(row=row_of.reindex(rows["person_id"]).to_numpy()).drop(columns="person_id")
    return keyed[["row", *[c for c in keyed.columns if c != "row"]]].sort_values(["row", "day"])


def split(days: pd.DataFrame, discovery_people: pd.Series, sealed_people: pd.Series | None,
          discovery_csv: str | Path) -> Path:
    """Write the outcome's day tables: the discovery share beside the cohort table, the sealed share (when the
    outcome has one) in the sealed folder; return the discovery path."""
    path = days_path(discovery_csv)
    _keyed(days, discovery_people).to_csv(path, index=False, compression="gzip")
    if sealed_people is not None:
        holdout = holdout_days_path(discovery_csv)
        holdout.parent.mkdir(parents=True, exist_ok=True)
        _keyed(days, sealed_people).to_csv(holdout, index=False, compression="gzip")
    return path
