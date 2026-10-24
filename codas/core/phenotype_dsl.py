"""A small language for defining a phenotype from a person-day table, the way an analyst would.

A summary of daily values is only one way to read a wearable series. How a person's days vary, drift, differ
at weekends, or respond to the day before (next-day resting heart rate after a short night, say) are
constructs a mean cannot show, and they are where new markers come from. A definition names them in JSON:

* ``signal``: a day-level column, or ``{"op": "ratio" | "difference" | "product" | "sum", "a": col, "b": col}``;
* ``where`` (optional): conditions that select the days used, each ``{"column": col, "op": "<" | "<=" | ">" |
  ">=" | "==" | "!=", "value": number}``; ``"prev:col"`` is the previous day's value of ``col`` (the day before,
  when it was recorded);
* ``summary``: ``mean``, ``sd``, ``cv`` (SD / mean), ``slope_per_30d`` (least-squares trend per 30 days),
  ``mean_abs_change`` (mean absolute change between consecutive days: regularity), ``weekend_minus_weekday``
  (mean on Saturdays and Sundays minus the rest), ``fraction`` (share of days whose signal meets ``fraction``,
  ``{"op": ..., "value": ...}``) or ``coupling`` (the within-person least-squares slope of the signal on
  ``couple.with``, that column ``couple.lag`` days earlier, 0 or 1: how the signal responds to it);
* ``min_days`` (default 14): with fewer usable days the person's value is missing.

The table has one row per person and day: ``row`` (the person: the row of the cohort table), ``day`` (days
relative to the index date, negative in the baseline window), ``weekday`` (1 = Sunday ... 7 = Saturday) and
the day-level signals. ``validate`` checks a definition against the table's columns, ``compute`` evaluates it
deterministically, ``describe`` says it in words and ``digest`` is its identity, so a column name never
changes meaning.

Nothing here names a dataset, a signal or an outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ("row", "day", "weekday")
OPS = {"ratio": "/", "difference": "-", "product": "*", "sum": "+"}
COMPARE = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal,
           "!=": np.not_equal}
SUMMARIES = ("mean", "sd", "cv", "slope_per_30d", "mean_abs_change", "weekend_minus_weekday", "fraction", "coupling")
MIN_DAYS = 14
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
_WEEKEND = (1, 7)  # Sunday and Saturday


def signal_columns(days: pd.DataFrame) -> list[str]:
    """The day-level signals a definition may use: every numeric column but the keys."""
    return [c for c in days.columns if c not in KEYS and pd.api.types.is_numeric_dtype(days[c])]


def _column_ok(name: Any, allowed: set[str]) -> bool:
    name = str(name)
    return (name[5:] if name.startswith("prev:") else name) in allowed


def validate(definition: dict[str, Any], allowed: set[str]) -> list[str]:
    """Every problem with ``definition`` given the ``allowed`` signal columns (none: it can be computed)."""
    problems = []
    if not isinstance(definition, dict):
        return ["the definition must be a JSON object"]
    if not _NAME.match(str(definition.get("name") or "")):
        problems.append("name: lowercase letters, digits and underscores, 2-31 characters, starting with a letter")
    signal = definition.get("signal")
    if isinstance(signal, dict):
        if signal.get("op") not in OPS:
            problems.append(f"signal op must be one of {', '.join(OPS)}")
        for side in ("a", "b"):
            if not _column_ok(signal.get(side), allowed) or str(signal.get(side)).startswith("prev:"):
                problems.append(f"signal {side}: {signal.get(side)!r} is not a day-level column")
    elif not (isinstance(signal, str) and signal in allowed):
        problems.append(f"signal {signal!r} is not a day-level column")
    for cond in definition.get("where") or []:
        if not isinstance(cond, dict) or not _column_ok(cond.get("column"), allowed):
            problems.append(f"where: {cond!r} needs a day-level column (or prev:column)")
        elif cond.get("op") not in COMPARE or not isinstance(cond.get("value"), (int, float)) \
                or isinstance(cond.get("value"), bool):
            problems.append(f"where on {cond.get('column')}: op from {', '.join(COMPARE)} and a number")
    summary = definition.get("summary")
    if summary not in SUMMARIES:
        problems.append(f"summary must be one of {', '.join(SUMMARIES)}")
    if summary == "fraction":
        f = definition.get("fraction") or {}
        if f.get("op") not in COMPARE or not isinstance(f.get("value"), (int, float)) or isinstance(f.get("value"), bool):
            problems.append("a fraction needs fraction: {op, value}")
    if summary == "coupling":
        c = definition.get("couple") or {}
        if str(c.get("with")) not in allowed:
            problems.append("a coupling needs couple.with, a day-level column")
        if c.get("lag", 0) not in (0, 1):
            problems.append("couple.lag must be 0 (the same day) or 1 (the day before)")
    min_days = definition.get("min_days", MIN_DAYS)
    if not isinstance(min_days, int) or isinstance(min_days, bool) or not 3 <= min_days <= 366:
        problems.append("min_days must be a whole number of days, 3-366")
    return problems


def canonical(definition: dict[str, Any]) -> dict[str, Any]:
    """The definition with only the keys that change its meaning, in a fixed form."""
    keep = {k: definition[k] for k in ("signal", "where", "summary", "fraction", "couple") if definition.get(k)}
    keep["min_days"] = int(definition.get("min_days", MIN_DAYS))
    if keep["summary"] != "fraction":
        keep.pop("fraction", None)
    if keep["summary"] != "coupling":
        keep.pop("couple", None)
    return keep


def digest(definition: dict[str, Any]) -> str:
    """A short identity of what the definition computes (its name aside)."""
    return hashlib.sha256(json.dumps(canonical(definition), sort_keys=True).encode()).hexdigest()[:8]


def column_name(definition: dict[str, Any]) -> str:
    """The column a defined phenotype gets: its name and its identity, so a reused name never changes meaning."""
    return f"dsl_{definition['name']}_{digest(definition)}"


def _with_previous(days: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """``prev:col`` for each column: the value on the calendar day before, when that day was recorded."""
    if not columns:
        return days
    base = days[["row", "day", *columns]].copy()
    base["day"] = base["day"] + 1
    base = base.rename(columns={c: f"prev:{c}" for c in columns})
    return days.merge(base, on=["row", "day"], how="left")


def _signal(days: pd.DataFrame, signal: Any) -> pd.Series:
    if isinstance(signal, str):
        return pd.to_numeric(days[signal], errors="coerce")
    a, b = (pd.to_numeric(days[signal[k]], errors="coerce") for k in ("a", "b"))
    if signal["op"] == "ratio":
        return a / b.where(b != 0)
    return {"difference": a - b, "product": a * b, "sum": a + b}[signal["op"]]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    denom = float(xc @ xc)
    return float(xc @ (y - y.mean()) / denom) if denom > 0 else math.nan


def compute(days: pd.DataFrame, definition: dict[str, Any]) -> pd.Series:
    """The phenotype for every person (indexed by ``row``); missing with fewer than ``min_days`` usable days."""
    allowed = set(signal_columns(days))
    problems = validate(definition, allowed)
    if problems:
        raise ValueError("; ".join(problems))
    d = canonical(definition)
    needed = sorted({str(c["column"])[5:] for c in d.get("where", []) if str(c["column"]).startswith("prev:")}
                    | ({d["couple"]["with"]} if d["summary"] == "coupling" and d["couple"].get("lag", 0) == 1 else set()))
    frame = _with_previous(days.sort_values(["row", "day"]), needed)
    frame = frame.assign(_signal=_signal(frame, d["signal"]).to_numpy())
    keep = np.isfinite(frame["_signal"].to_numpy(dtype=float))
    for cond in d.get("where", []):
        values = pd.to_numeric(frame[cond["column"]], errors="coerce").to_numpy(dtype=float)
        keep &= np.isfinite(values) & COMPARE[cond["op"]](values, float(cond["value"]))
    if d["summary"] == "coupling":
        other = d["couple"]["with"]
        column = f"prev:{other}" if d["couple"].get("lag", 0) == 1 else other
        frame = frame.assign(_with=pd.to_numeric(frame[column], errors="coerce").to_numpy())
        keep &= np.isfinite(frame["_with"].to_numpy(dtype=float))
    used = frame[keep]
    rows = sorted(days["row"].unique())
    out = pd.Series(np.nan, index=pd.Index(rows, name="row"), dtype=float)
    for row, g in used.groupby("row", sort=True):
        if len(g) < d["min_days"]:
            continue
        s = g["_signal"].to_numpy(dtype=float)
        summary = d["summary"]
        if summary == "mean":
            value = s.mean()
        elif summary == "sd":
            value = s.std(ddof=1)
        elif summary == "cv":
            value = s.std(ddof=1) / s.mean() if s.mean() != 0 else math.nan
        elif summary == "slope_per_30d":
            value = 30 * _slope(g["day"].to_numpy(dtype=float), s)
        elif summary == "mean_abs_change":
            day = g["day"].to_numpy()
            consecutive = np.diff(day) == 1
            value = float(np.abs(np.diff(s))[consecutive].mean()) if consecutive.any() else math.nan
        elif summary == "weekend_minus_weekday":
            weekend = g["weekday"].isin(_WEEKEND).to_numpy()
            value = s[weekend].mean() - s[~weekend].mean() if weekend.any() and (~weekend).any() else math.nan
        elif summary == "fraction":
            value = float(COMPARE[d["fraction"]["op"]](s, float(d["fraction"]["value"])).mean())
        else:  # coupling: how the signal moves with the other variable within this person
            value = _slope(g["_with"].to_numpy(dtype=float), s)
        out.loc[row] = value if np.isfinite(value) else math.nan
    return out


def describe(definition: dict[str, Any]) -> str:
    """The definition in words."""
    d = canonical(definition)
    signal = d["signal"] if isinstance(d["signal"], str) else f"{d['signal']['a']} {OPS[d['signal']['op']]} {d['signal']['b']}"
    summary = {"mean": "mean", "sd": "day-to-day SD", "cv": "coefficient of variation",
               "slope_per_30d": "trend per 30 days", "mean_abs_change": "mean absolute change between consecutive days",
               "weekend_minus_weekday": "weekend minus weekday mean"}.get(d["summary"])
    if d["summary"] == "fraction":
        text = f"share of days with {signal} {d['fraction']['op']} {d['fraction']['value']:g}"
    elif d["summary"] == "coupling":
        lag = "the day before" if d["couple"].get("lag", 0) == 1 else "the same day"
        text = f"within-person slope of {signal} on {d['couple']['with']} {lag}"
    else:
        text = f"{summary} of {signal}"
    where = [f"{c['column'].replace('prev:', 'previous day ')} {c['op']} {c['value']:g}" for c in d.get("where", [])]
    return text + (f", over days with {' and '.join(where)}" if where else "") + f" (at least {d['min_days']} days)"


def add_columns(definitions: list[dict[str, Any]], days: pd.DataFrame, table_csv: str | Path) -> list[str]:
    """Compute each definition on ``days`` and write it into the table at ``table_csv`` as its column
    (``column_name``), row by row (``days.row`` is the table's row); return the columns written."""
    path = Path(table_csv)
    table = pd.read_csv(path, float_precision="round_trip")
    written = []
    for definition in definitions:
        column = column_name(definition)
        table[column] = compute(days, definition).reindex(range(len(table))).to_numpy()
        written.append(column)
    tmp = path.with_name(path.name + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)  # complete or absent, never half-written
    return written
