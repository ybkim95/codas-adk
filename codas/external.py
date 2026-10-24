"""Replicating registered hypotheses in another cohort, from the replication package alone.

A run's replication package (``codas.aou.replication``) holds each hypothesis's feature definition, its
registered direction and the protocol; no participant data. This module tests the same hypotheses on another
cohort's own tables where that cohort is kept, and returns aggregates only:

* the cohort's per-person table (an id column, the outcome, covariates, laboratory values) and, optionally, its
  per-person-day table (an id, a date and daily signals);
* a map from the package's names to the cohort's columns: ``signals`` (daily signal name in the package ->
  column of the day table), ``features`` (a feature taken directly from the person table), ``outcome``
  (``{"column": ...}``), ``covariates`` (package name -> column) and ``labs`` (columns to triangulate with);
* each feature is rebuilt on the cohort: a phenotype definition on the day table (``codas.core.phenotype_dsl``,
  with days counted from each person's first day), a daily signal's mean or SD over the first ``window_days``,
  or a mapped person-table column; one that cannot be built is reported with the reason;
* each hypothesis is tested by the partial Spearman correlation with the mapped covariates, one-sided in the
  registered direction, Benjamini-Hochberg across the hypotheses tested here; each feature is also correlated
  with every mapped laboratory value (adjusted for age and sex when mapped): biology the cohort can show and the
  first cohort could not.

Counts of 1-20 are masked in the output. Nothing here names a dataset, a cohort, a feature or an outcome; the
map does.

Run where the cohort's data are kept::

    python -m codas.external --package <replication folder> --people <csv> [--days <csv>] --map <json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from codas.core import phenotype_dsl
from codas.core.statistics import benjamini_hochberg, partial_spearman

SMALL = 20
WINDOW_DAYS = 180


def _mask(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("<=20" if k in ("n", "participants") and isinstance(v, int) and 1 <= v <= SMALL else _mask(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    return obj


def load_package(folder: str | Path) -> dict[str, Any]:
    folder = Path(folder)
    prereg = json.loads((folder / "pre_registration.json").read_text(encoding="utf-8"))
    features = json.loads((folder / "features.json").read_text(encoding="utf-8")) \
        if (folder / "features.json").exists() else {}
    return {"hypotheses": prereg.get("hypotheses") or [], "features": features, "prereg": prereg}


def day_table(days: pd.DataFrame, people: pd.Series, mapping: dict[str, Any]) -> pd.DataFrame:
    """The cohort's days in the package's layout: ``row`` (position in ``people``), ``day`` (from the person's
    first day), ``weekday`` (1 = Sunday) and the mapped signals under the package's names."""
    pid, date = mapping.get("participant", "participant_id"), mapping.get("date", "date")
    row_of = pd.Series(range(len(people)), index=people.astype(str).to_numpy())
    d = days[days[pid].astype(str).isin(row_of.index)].copy()
    when = pd.to_datetime(d[date])
    first = when.groupby(d[pid].astype(str)).transform("min")
    out = pd.DataFrame({"row": row_of.reindex(d[pid].astype(str)).to_numpy(), "day": (when - first).dt.days.to_numpy(),
                        "weekday": ((when.dt.dayofweek + 1) % 7 + 1).to_numpy()})
    for name, column in (mapping.get("signals") or {}).items():
        if column in d.columns:
            out[name] = pd.to_numeric(d[column], errors="coerce").to_numpy()
    return out


def build_feature(name: str, definition: dict[str, Any] | None, people: pd.DataFrame, days: pd.DataFrame | None,
                  mapping: dict[str, Any]) -> tuple[pd.Series | None, str]:
    """The feature on the cohort's people (by position) and how it was built, or None and the reason."""
    direct = (mapping.get("features") or {}).get(name)
    if direct:
        if direct not in people.columns:
            return None, f"mapped column {direct!r} is not in the person table"
        return pd.to_numeric(people[direct], errors="coerce").reset_index(drop=True), f"column {direct}"
    window = int(mapping.get("window_days") or WINDOW_DAYS)
    if days is not None and len(days):
        early = days[days["day"] < window]
        phenotype = (definition or {}).get("phenotype")
        if phenotype:
            try:
                values = phenotype_dsl.compute(early, phenotype)
            except ValueError as exc:
                return None, f"phenotype not computable here ({exc})"
            return values.reindex(range(len(people))).reset_index(drop=True), "phenotype definition"
        for summary in ("_mean", "_sd"):
            signal = name[: -len(summary)] if name.endswith(summary) else None
            if signal and signal in early.columns:
                grouped = early.groupby("row")[signal]
                counts = grouped.count()
                values = grouped.mean() if summary == "_mean" else grouped.std()
                values = values.where(counts >= int(mapping.get("min_days") or 14))
                return values.reindex(range(len(people))).reset_index(drop=True), \
                    f"{summary[1:]} of {signal} over the first {window} days"
    return None, "no mapping, daily signal or definition for it in this cohort"


def _partial(x: pd.Series, y: pd.Series, cov: pd.DataFrame | None) -> tuple[float, float, int]:
    c = np.zeros((len(x), 0)) if cov is None or not cov.shape[1] else cov.to_numpy(dtype=float)
    if not c.shape[1]:
        ok = x.notna() & y.notna()
        r, p = stats.spearmanr(x[ok], y[ok]) if ok.sum() > 2 else (math.nan, math.nan)
        return float(r), float(p), int(ok.sum())
    return partial_spearman(x.to_numpy(dtype=float), y.to_numpy(dtype=float), c)


def replicate(package: dict[str, Any], people: pd.DataFrame, days: pd.DataFrame | None,
              mapping: dict[str, Any]) -> dict[str, Any]:
    pid = mapping.get("participant", "participant_id")
    people = people.reset_index(drop=True)
    table = day_table(days, people[pid], mapping) if days is not None else None
    outcome = (mapping.get("outcome") or {}).get("column")
    y = pd.to_numeric(people[outcome], errors="coerce") if outcome in people.columns else None
    covariates = {k: v for k, v in (mapping.get("covariates") or {}).items() if v in people.columns}
    cov = people[list(covariates.values())].apply(pd.to_numeric, errors="coerce") if covariates else None
    basic = [covariates[k] for k in ("age", "female") if k in covariates]
    labs = [c for c in mapping.get("labs") or [] if c in people.columns]
    rows = []
    for h in package["hypotheses"]:
        name = h["feature"]
        x, how = build_feature(name, h.get("definition") or (package["features"].get(name) or {}), people, table, mapping)
        row: dict[str, Any] = {"feature": name, "direction": h.get("direction"), "built": how}
        if x is None:
            row["status"] = "not testable here"
            rows.append(row)
            continue
        if y is not None:
            r, p, n = _partial(x, y, cov)
            same = bool(np.isfinite(r) and np.sign(r) == (h.get("direction") or 0))
            row.update(rho=round(float(r), 4), p=float(p), n=int(n), same_direction=same,
                       p_one_sided=float(p / 2 if same else 1 - p / 2) if np.isfinite(p) else math.nan)
        if labs:
            adjust = people[basic].apply(pd.to_numeric, errors="coerce") if basic else None
            row["labs"] = {}
            for lab in labs:
                r, p, n = _partial(x, pd.to_numeric(people[lab], errors="coerce"), adjust)
                row["labs"][lab] = {"rho": round(float(r), 4), "p": float(p), "n": int(n)}
        rows.append(row)
    tested = [r for r in rows if np.isfinite(r.get("p_one_sided", math.nan))]
    for r, q in zip(tested, benjamini_hochberg([r["p_one_sided"] for r in tested])):
        r["q"] = float(q)
        r["replicated"] = bool(r["same_direction"] and q < 0.05)
    return {"outcome": outcome, "covariates": covariates, "hypotheses": rows,
            "replicated": sum(r.get("replicated", False) for r in rows), "tested": len(tested)}


def markdown(result: dict[str, Any]) -> str:
    lines = [f"## External replication ({result['replicated']} of {result['tested']} replicated)", "",
             f"Outcome {result['outcome']}; adjusted for {', '.join(result['covariates']) or 'nothing'}; partial "
             "Spearman, one-sided in the registered direction, Benjamini-Hochberg across the hypotheses tested here.",
             "", "| feature | built from | rho | n | q | replicated |", "|---|---|---|---|---|---|"]
    for r in result["hypotheses"]:
        lines.append(f"| {r['feature']} | {r['built']} | {r.get('rho', '-')} | {r.get('n', '-')} | "
                     f"{format(r['q'], '.2g') if 'q' in r else '-'} | "
                     f"{'yes' if r.get('replicated') else r.get('status', 'no')} |")
    with_labs = [r for r in result["hypotheses"] if r.get("labs")]
    if with_labs:
        labs = sorted({lab for r in with_labs for lab in r["labs"]})
        lines += ["", "Each feature against the cohort's laboratory values (partial Spearman, adjusted for age and sex):",
                  "", "| feature | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for r in with_labs:
            lines.append(f"| {r['feature']} | " + " | ".join(
                f"{r['labs'][lab]['rho']} (p {r['labs'][lab]['p']:.1g})" if lab in r["labs"] else "-" for lab in labs) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test a replication package's hypotheses on another cohort.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--people", required=True)
    parser.add_argument("--days")
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    people = pd.read_csv(args.people)
    days = pd.read_csv(args.days) if args.days else None
    result = _mask(replicate(load_package(args.package), people, days, mapping))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "external_replication.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (out / "external_replication.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
