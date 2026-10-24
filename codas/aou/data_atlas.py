"""The data atlas: what the data hold, before any hypothesis is written, from the discovery share only and in
aggregate (counts of 1-20 masked when shown).

It answers the questions a researcher asks of a new cohort before modelling it:

* **Coverage over calendar time.** For each daily wearable source of the spec, the discovery participants with a
  day in each calendar month and their mean days that month: when the data begin, how they grow, and whether a
  source starts later than the others.
* **Who has which data.** For each wearable source and each intraday phenotype source, the participants who
  have it against those who do not, compared on age, sex and every covariate of the spec by the standardised
  mean difference (|SMD| >= 0.1 is flagged): a source held by a different kind of participant is a selection
  that an analysis using it inherits.
* **Wear and the health record.** When the spec has a ``visits`` covariate (visits in the year before index),
  its Spearman correlation with days of wear and its mean by fifth of wear: a participant who wears the device
  more may also see doctors more, which changes how often a diagnosis can be recorded.
* **Repeated laboratory values during wear.** For each catalog laboratory outcome, the participants with at
  least two in-range values between their first and last valid wear day and the median number of values among
  them: who can be followed within person.
* **Survey questions answered.** The survey questions the most discovery participants answered (by concept),
  for questions a later analysis could use as covariates or outcomes.

Nothing here reads an outcome association, so the atlas spends no holdout and can be shown to anyone who may
see aggregates. Three BigQuery reads (the daily sources by month, the measurements, the survey answers); the
rest comes from the discovery person table the phenome scan already read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codas.aou import cohort
from codas.aou.cohort import CohortSpec, QueryParam, _first_wear_cte, _valid_rule, resolve_cdr
from codas.aou.privacy import MASK, SMALL_CELL_MAX, mask_small_counts
from codas.core.statistics import safe_spearman

FILE = "data_atlas.json"
SURVEY_VOCABULARY = "PPI"   # the vocabulary of All of Us survey questions
TOP_QUESTIONS = 60
SMD_NOTABLE = 0.1           # |standardised mean difference| from which a difference is flagged


def _people(spec: CohortSpec, t: Any, params: list[QueryParam]) -> list[str]:
    """The discovery participants with their first and last valid day of the primary source (the sealed share
    is left out here, in BigQuery)."""
    from codas.aou.phenome import _people as scan_people

    primary = spec.wearables[0]
    ctes = scan_people(spec, t, params)
    valid = _valid_rule(spec, [])  # the bounds are already bound by scan_people
    ctes.append("wear_span AS (\n"
                f"  SELECT w.person_id, MIN(DATE(w.{primary.date_column})) AS first_day,\n"
                f"    MAX(DATE(w.{primary.date_column})) AS last_day\n"
                f"  FROM {t(primary.table)} w\n"
                "  JOIN people p ON p.person_id = w.person_id\n"
                f"  WHERE TRUE{valid.format(a='w')}\n"
                "  GROUP BY w.person_id\n)")
    return ctes


def coverage_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """Per daily wearable source and calendar month: discovery participants with a day of it, and their days."""
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = []
    ctes = _people(spec, t, params)
    parts = []
    for i, w in enumerate(spec.wearables):
        where = ""
        if w.filter_column is not None:
            params.append((f"f{i}", "STRING", str(w.filter_value)))
            where = f"\n  WHERE LOWER(CAST(w.{w.filter_column} AS STRING)) = LOWER(@f{i})"
        parts.append(f"SELECT '{w.prefix}' AS source, FORMAT_DATE('%Y-%m', DATE(w.{w.date_column})) AS month,\n"
                     "    COUNT(DISTINCT w.person_id) AS participants, COUNT(*) AS days\n"
                     f"  FROM {t(w.table)} w JOIN people p ON p.person_id = w.person_id{where}\n"
                     "  GROUP BY month")
    return "WITH " + ",\n".join(ctes) + "\n" + "\nUNION ALL\n".join(parts), params


def repeat_labs_sql(labs: list[Any], spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """Per catalog laboratory outcome: discovery participants with at least two in-range values during their
    wear span, and the median number of values among them."""
    from codas.aou.phenome import _outcome_map

    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = []
    ctes = _people(spec, t, params) + _outcome_map(labs, t, params)
    rules = []
    for i, o in enumerate(labs):
        bounds = []
        if o.value_min is not None:
            bounds.append(f"m.value_as_number >= @o{i}_min")
            params.append((f"o{i}_min", "FLOAT64", float(o.value_min)))
        if o.value_max is not None:
            bounds.append(f"m.value_as_number <= @o{i}_max")
            params.append((f"o{i}_max", "FLOAT64", float(o.value_max)))
        if bounds:
            rules.append(f"      WHEN {i} THEN " + " AND ".join(bounds))
    rule = ("\n    AND CASE om.outcome\n" + "\n".join(rules) + "\n      ELSE TRUE END") if rules else ""
    ctes.append("during AS (\n"
                "  SELECT m.person_id, om.outcome, COUNT(*) AS n_values\n"
                f"  FROM {t('measurement')} m\n"
                "  JOIN outcome_map om ON om.concept_id = m.measurement_concept_id\n"
                "  JOIN wear_span s ON s.person_id = m.person_id\n"
                "  WHERE m.measurement_date BETWEEN s.first_day AND s.last_day\n"
                f"    AND m.value_as_number IS NOT NULL{rule}\n"
                "  GROUP BY m.person_id, om.outcome\n)")
    sql = ("WITH " + ",\n".join(ctes) + "\n"
           "SELECT outcome, COUNT(*) AS participants, APPROX_QUANTILES(n_values, 2)[OFFSET(1)] AS median_values\n"
           "FROM during WHERE n_values >= 2 GROUP BY outcome")
    return sql, params


def survey_sql(spec: CohortSpec, cdr: str, top: int = TOP_QUESTIONS) -> tuple[str, list[QueryParam]]:
    """The survey questions most discovery participants answered, with how many answered each."""
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = [("survey_vocabulary", "STRING", SURVEY_VOCABULARY), ("top", "INT64", int(top))]
    ctes = _people(spec, t, params)
    sql = ("WITH " + ",\n".join(ctes) + "\n"
           "SELECT o.observation_source_concept_id AS question, c.concept_name AS name,\n"
           "  COUNT(DISTINCT o.person_id) AS participants\n"
           f"FROM {t('observation')} o\n"
           "JOIN people p ON p.person_id = o.person_id\n"
           f"JOIN {t('concept')} c ON c.concept_id = o.observation_source_concept_id\n"
           "WHERE c.vocabulary_id = @survey_vocabulary\n"
           "GROUP BY question, name\nORDER BY participants DESC, question\nLIMIT @top")
    return sql, params


def _smd(a: pd.Series, b: pd.Series) -> float | None:
    """Standardised mean difference of ``a`` against ``b`` (pooled SD)."""
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    return round(float((a.mean() - b.mean()) / sd), 3) if sd > 0 else None


def sources_of(people: pd.DataFrame, spec: CohortSpec) -> dict[str, list[str]]:
    """Each wearable and intraday phenotype source's columns in the person table, by prefix."""
    prefixes = [w.prefix for w in spec.wearables] + [p.prefix for p in spec.phenotypes]
    return {p: [c for c in people.columns if c.startswith(p + "_")] for p in prefixes
            if any(c.startswith(p + "_") for c in people.columns)}


def selection(people: pd.DataFrame, spec: CohortSpec) -> list[dict[str, Any]]:
    """For each source: participants with and without it, compared on age, sex and the spec's covariates."""
    traits = [c for c in ["age", "female", *[v.name for v in spec.covariates]] if c in people.columns]
    out = []
    for prefix, columns in sources_of(people, spec).items():
        values = [c for c in columns if not c.endswith("_days")] or columns
        has = people[values].notna().any(axis=1)
        n_with, n_without = int(has.sum()), int((~has).sum())
        # Either group's size and the total would give the other's, so a group of 1-20 hides both.
        shown = not any(1 <= v <= SMALL_CELL_MAX for v in (n_with, n_without))
        row: dict[str, Any] = {"source": prefix, "n_with": n_with if shown else None,
                               "n_without": n_without if shown else None, "traits": {}}
        if n_with > SMALL_CELL_MAX and n_without > SMALL_CELL_MAX:
            for trait in traits:
                row["traits"][trait] = _smd(people.loc[has, trait], people.loc[~has, trait])
        row["notable"] = sorted(k for k, v in row["traits"].items() if v is not None and abs(v) >= SMD_NOTABLE)
        out.append(row)
    return out


def wear_and_visits(people: pd.DataFrame, spec: CohortSpec) -> dict[str, Any] | None:
    """How days of wear relate to visits in the record (a ``visits`` covariate), or None without one."""
    visits = next((c.name for c in spec.covariates if c.kind == "visits" and c.name in people.columns), None)
    days = f"{spec.wearables[0].prefix}_days"
    if visits is None or days not in people.columns:
        return None
    frame = people[[days, visits]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 5 * (SMALL_CELL_MAX + 1):
        return None
    rho = safe_spearman(frame[days].to_numpy(), frame[visits].to_numpy())[0]
    fifth = pd.qcut(frame[days].rank(method="first"), 5, labels=False)
    return {"visits": visits, "days": days, "rho": round(float(rho), 3),
            "mean_visits_by_fifth_of_wear": [round(float(frame.loc[fifth == k, visits].mean()), 2) for k in range(5)]}


def _query(client: Any, sql: str, params: list[QueryParam], cdr: str, out: Path, kind: str) -> pd.DataFrame:
    from codas.aou.phenome import _cached

    return _cached(client, sql, params, cdr, out, f"atlas_{kind}")


def build(spec: CohortSpec, labs: list[Any], cdr: str, client: Any, out: Path, people: pd.DataFrame,
          log: Any = print) -> dict[str, Any]:
    """The atlas (module doc), raw; its masked copy is written to ``out/FILE``."""
    out = Path(out)
    coverage = _query(client, *coverage_sql(spec, cdr), cdr, out, "coverage")
    months = []
    for row in coverage.sort_values(["source", "month"]).itertuples(index=False):
        n = int(row.participants)
        months.append({"source": row.source, "month": row.month, "participants": n,
                       "mean_days": round(float(row.days) / n, 1) if n > SMALL_CELL_MAX else None})
    repeats = []
    if labs:
        found = _query(client, *repeat_labs_sql(labs, spec, cdr), cdr, out, "repeat_labs")
        by = {int(r.outcome): r for r in found.itertuples(index=False)}
        for i, o in enumerate(labs):
            r = by.get(i)
            repeats.append({"outcome": o.name, "label": o.label, "participants": int(r.participants) if r else 0,
                            "median_values": int(r.median_values) if r is not None and r.median_values == r.median_values
                            else None})
    try:
        questions = _query(client, *survey_sql(spec, cdr), cdr, out, "survey")
        survey = [{"question": int(r.question), "name": str(r.name), "participants": int(r.participants)}
                  for r in questions.itertuples(index=False)]
    except Exception as exc:  # noqa: BLE001 - a CDR without survey answers still gets the rest of the atlas
        log(f"data atlas: survey questions not read ({type(exc).__name__})")
        survey = []
    atlas = {"participants": int(len(people)), "coverage": months, "selection": selection(people, spec),
             "wear_and_visits": wear_and_visits(people, spec), "repeat_labs": repeats, "survey": survey}
    (out / FILE).write_text(json.dumps(mask_small_counts(atlas), indent=1, default=str), encoding="utf-8")
    log(f"data atlas: {len(months)} source-months, {len(atlas['selection'])} sources compared, "
        f"{sum(r['participants'] > SMALL_CELL_MAX for r in repeats)} labs with repeated values during wear, "
        f"{len(survey)} survey questions")
    return atlas


def _n(value: Any) -> str:
    return MASK if isinstance(value, int) and 1 <= value <= SMALL_CELL_MAX else f"{value:,}" if isinstance(value, int) \
        else str(value)


def markdown(atlas: dict[str, Any]) -> str:
    """The atlas as Markdown (counts of 1-20 masked)."""
    if not atlas:
        return ""
    lines = ["## Data atlas (discovery share, before any hypothesis)", ""]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for m in atlas.get("coverage") or []:
        by_source.setdefault(m["source"], []).append(m)
    if by_source:
        lines += ["**Coverage over calendar time**: each daily source's first and last month with data, its busiest "
                  "month and the participants then.", "", "| source | first month | last month | busiest month |",
                  "|---|---|---|---|"]
        for source, months in by_source.items():
            top = max(months, key=lambda m: m["participants"] if isinstance(m["participants"], int) else 0)
            lines.append(f"| {source} | {months[0]['month']} | {months[-1]['month']} | {top['month']} "
                         f"({_n(top['participants'])}) |")
        lines.append("")
    rows = atlas.get("selection") or []
    if rows:
        lines += [f"**Who has which data**: participants with each source against those without, by standardised "
                  f"mean difference (|SMD| >= {SMD_NOTABLE} listed).", "", "| source | with | without | differ on |",
                  "|---|---|---|---|"]
        for r in rows:
            notable = ", ".join(f"{k} {r['traits'][k]:+.2f}" for k in r.get("notable") or []) or "-"
            lines.append(f"| {r['source']} | {_n(r['n_with']) if r['n_with'] is not None else MASK} | "
                         f"{_n(r['n_without']) if r['n_without'] is not None else MASK} | {notable} |")
        lines.append("")
    visits = atlas.get("wear_and_visits")
    if visits:
        lines += [f"**Wear and the health record**: days of wear ({visits['days']}) and {visits['visits']}: Spearman "
                  f"{visits['rho']:+.2f}; mean visits by fifth of wear, least to most: "
                  + ", ".join(str(v) for v in visits["mean_visits_by_fifth_of_wear"]) + ".", ""]
    repeats = [r for r in atlas.get("repeat_labs") or [] if r.get("participants")]
    if repeats:
        lines += ["**Repeated laboratory values during wear** (participants with two or more in-range values between "
                  "their first and last wear day):", "", "| lab | participants | median values |", "|---|---|---|"]
        for r in sorted(repeats, key=lambda r: -(r["participants"] if isinstance(r["participants"], int) else 0)):
            lines.append(f"| {r['label']} | {_n(r['participants'])} | {r.get('median_values') or '-'} |")
        lines.append("")
    survey = atlas.get("survey") or []
    if survey:
        lines += [f"**Survey questions answered** (the {len(survey)} most answered): "
                  + "; ".join(f"{q['name']} ({_n(q['participants'])})" for q in survey[:15]) + ".", ""]
    return "\n".join(lines)
