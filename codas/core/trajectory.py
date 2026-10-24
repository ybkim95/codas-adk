"""A feature followed over time: the exposure as it changes during follow-up, and its course before a diagnosis.

Both read a table of blocks per row (``row``, ``block`` = days from the time origin divided by the block length,
rounded down, and one value per block), and a table of rows with the follow-up: where it starts (``start``,
days), where it ends (``end``, days) and whether it ends in the event (``event``).

* ``intervals``: each row's follow-up cut at block edges, with the exposure of each interval: the mean of the
  ``window`` blocks ending ``lag`` blocks before it (``lag`` 1: the block just before; a longer lag keeps changes
  caused by a disease not yet diagnosed out of the exposure), or the latest earlier value when that window is
  empty. ``codas.core.survival.cox_ph``
  with ``entry`` fits it as a time-varying exposure.
* ``matched_trajectory``: each case's course before its diagnosis against controls still followed and free of
  the event at that time (incidence density sampling, matched on sex and age), in windows of blocks before the
  diagnosis; per window the mean case-minus-controls difference in SD units with its 95% interval, and when the
  difference begins: ``stable`` (in every window), ``prodromal`` (only in the windows nearest the diagnosis),
  ``none`` or ``mixed``.

Nothing here names a dataset, an outcome or a feature.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _window_means(blocks: pd.DataFrame, value: str, lag: int, window: int) -> pd.DataFrame:
    """Per row and block b: the mean of ``value`` over blocks b - lag - window + 1 .. b - lag, carried forward
    from the latest earlier block with data when that range is empty."""
    wide = blocks.pivot_table(index="row", columns="block", values=value, aggfunc="mean")
    full = range(int(wide.columns.min()), int(wide.columns.max()) + lag + 2)
    wide = wide.reindex(columns=full)
    rolled = wide.T.rolling(window, min_periods=1).mean().T.ffill(axis=1)
    shifted = rolled.shift(lag, axis=1)  # the value for block b ends with block b - lag
    return shifted.stack(future_stack=True).dropna().rename("exposure").reset_index()


def intervals(people: pd.DataFrame, blocks: pd.DataFrame, value: str, *, block_days: int, lag: int,
              window: int) -> pd.DataFrame:
    """Follow-up intervals with a time-varying exposure (module doc): ``row``, ``entry``, ``stop`` (days),
    ``event`` (on the last interval of a row that ends in the event) and ``exposure``."""
    exposure = _window_means(blocks, value, lag, window)
    lookup = exposure.set_index(["row", "block"])["exposure"]
    out = []
    for row, start, end, event in people[["row", "start", "end", "event"]].itertuples(index=False):
        if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
            continue
        first, last = int(start // block_days), int(math.ceil(end / block_days) - 1)
        for b in range(first, last + 1):
            entry, stop = max(start, b * block_days), min(end, (b + 1) * block_days)
            if stop <= entry:
                continue
            out.append((row, entry, stop, int(bool(event) and stop >= end), lookup.get((row, b), np.nan)))
    frame = pd.DataFrame(out, columns=["row", "entry", "stop", "event", "exposure"])
    return frame.dropna(subset=["exposure"]).reset_index(drop=True)


def matched_trajectory(people: pd.DataFrame, blocks: pd.DataFrame, value: str, *, block_days: int,
                       windows: list[tuple[int, int]], sd: float, controls: int = 4, age_caliper: float = 3.0,
                       min_cases: int = 1, seed: int = 20251024) -> dict[str, Any]:
    """Cases against matched controls before the diagnosis (module doc). ``people`` has ``row``, ``event``,
    ``end`` (days to the event or censoring), and ``female`` and ``age`` when matching on them; ``windows``
    are block ranges relative to the diagnosis block, ``[a, b)``, earliest first; a window with ``min_cases``
    cases or fewer is not estimated."""
    rng = np.random.default_rng(seed)
    p = people.reset_index(drop=True)
    values = blocks.groupby(["row", "block"])[value].mean()
    by_row = {r: s.droplevel(0) for r, s in values.groupby(level=0)}
    cases = p[(p["event"] > 0) & np.isfinite(p["end"])]
    sex = p["female"].to_numpy() if "female" in p.columns else np.zeros(len(p))
    age = p["age"].to_numpy(dtype=float) if "age" in p.columns else np.zeros(len(p))
    end = p["end"].to_numpy(dtype=float)
    rows = p["row"].to_numpy()

    def mean_in(row: Any, ref: int, a: int, b: int) -> float:
        s = by_row.get(row)
        if s is None:
            return np.nan
        inside = s[(s.index >= ref + a) & (s.index < ref + b)]
        return float(inside.mean()) if len(inside) else np.nan

    diffs: list[list[float]] = [[] for _ in windows]
    for case in cases.itertuples(index=False):
        t = float(case.end)
        ref = int(t // block_days)
        same = (end > t) & (sex == (case.female if "female" in p.columns else 0))
        if "age" in p.columns:
            same &= np.abs(age - float(case.age)) <= age_caliper
        pool = rows[same]
        if not len(pool):
            continue
        chosen = rng.choice(pool, size=min(controls, len(pool)), replace=False)
        for k, (a, b) in enumerate(windows):
            mine = mean_in(case.row, ref, a, b)
            theirs = [mean_in(c, ref, a, b) for c in chosen]
            theirs = [v for v in theirs if np.isfinite(v)]
            if np.isfinite(mine) and theirs:
                diffs[k].append((mine - float(np.mean(theirs))) / sd)
    out_windows = []
    for (a, b), d in zip(windows, diffs):
        entry: dict[str, Any] = {"blocks": [a, b], "cases": len(d)}
        if len(d) > max(min_cases, 1):
            arr = np.asarray(d)
            mean, se = float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))
            z = stats.t.ppf(0.975, len(arr) - 1)
            entry.update(difference_sd=round(mean, 4), ci95=[round(mean - z * se, 4), round(mean + z * se, 4)],
                         p=float(2 * stats.t.sf(abs(mean / se), len(arr) - 1)) if se > 0 else float("nan"))
        else:
            entry["status"] = "too few cases"
        out_windows.append(entry)
    return {"windows": out_windows, "onset": onset([w for w in out_windows if "ci95" in w]), "value": value}


def onset(windows: list[dict[str, Any]]) -> str:
    """When the case-control difference begins, from windows ordered earliest first (module doc)."""
    if not windows:
        return "not estimated"
    signs = [(1 if w["ci95"][0] > 0 else -1 if w["ci95"][1] < 0 else 0) for w in windows]
    if not any(signs):
        return "none"
    if all(s == signs[0] and s for s in signs):
        return "stable"
    first = next(i for i, s in enumerate(signs) if s)
    late = signs[first:]
    if first > 0 and all(s == late[0] for s in late):
        return "prodromal"
    return "mixed"
