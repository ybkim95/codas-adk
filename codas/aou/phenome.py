"""The phenome-wide screen: every wearable feature and phenotype against every verified catalog outcome, on
the discovery share, to choose the outcomes worth a sealed-holdout test by their signal and by how little the
literature has said about it, rather than by their size.

Under the incident design the index date is a property of the person (the end of the baseline window that
starts at the first valid wear day), so a participant's features are the same in every outcome's cohort. The
screen therefore reads three tables, each once:

* the person table: every discovery participant's features and covariates, built by the extraction's own
  code (``build_cohort_sql(person_level=True)``), with the phenotypes computed once per run merged in;
* the diagnosis matrix: each discovery participant's first diagnosis (in days from the index date) and the
  number of distinct diagnosis dates, for every catalog condition, in one pass over the condition table;
* the laboratory matrix: each discovery participant's first in-range value after index + lag, for every
  catalog laboratory outcome, in one pass over the measurement table.

The sealed share is filtered out in BigQuery: no sealed row is downloaded. Each outcome's cohort (who is
eligible, who has the event and after how much follow-up, or the value) is then derived by the extraction's
rules; a test checks it against the extraction itself. Per outcome and feature:

* T0, adjusted for age and sex: for a diagnosis the Cox score test (Breslow ties) at the age-and-sex fit, in
  Neyman's C(alpha) form so a feature's own missing rows do not bias it, with the one-step log hazard ratio
  per SD (``codas.core.survival.score_tests``); for a laboratory value the partial Spearman correlation.
* T1, adjusted also for the spec's covariates and the wear-adherence counts (a missing value filled with the
  discovery median and flagged by an indicator: a screening approximation; the holdout uses multiple
  imputation).
* T0 in each of two fixed random halves of the discovery share.

Benjamini-Hochberg runs across every tested pair (T0). A hit has q below Q_LEVEL, a size that is not
negligible (``codas.core.effects.CLASSES``), the same sign with one-sided p below HALF_P in both halves, and
the same sign with p below T1_P at T1. The literature is counted for each outcome's strongest hits
(``codas.core.literature.evidence``). Outcomes are ranked by hits whose construct the literature has not
established, then by hits, then by the strongest adjusted statistic; outcomes already tested on this
holdout are left out, and only outcomes with a hit are selected.

The screen spends no holdout: it reads the discovery share only, and only registered hypotheses are tested
on the sealed share. Its tables are aggregates with counts of 1-20 masked, and no agent sees them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from codas.aou import cohort
from codas.aou.cohort import (CohortSpec, QueryParam, _first_wear_cte, _valid_rule, availability_columns,
                              build_cohort_sql, resolve_cdr, sealed_dir)
from codas.aou.privacy import MASK, SMALL_CELL_MAX, mask_small_counts
from codas.core import effects
from codas.core.statistics import benjamini_hochberg
from codas.core.survival import breslow_fit, score_tests

Q_LEVEL = 0.05   # Benjamini-Hochberg across every tested pair
HALF_P = 0.05    # one-sided, in the hit's direction, in each half
T1_P = 0.05      # two-sided, with the hit's sign, after the covariates and wear adherence
LITERATURE_HITS = int(os.getenv("CODAS_SCAN_LITERATURE_HITS", "3"))  # strongest hits per outcome looked up
SHOWN_HITS = 30   # hits listed in the summary
# How specific a hit is: the share of screened outcomes its feature is a hit for. A feature that is a hit for
# more than BROAD_SHARE of them behaves like a general marker of poor health (Julkunen et al., 2023); one that
# is a hit for at most SPECIFIC_SHARE singles out few outcomes. Below SPECIFICITY_MIN_OUTCOMES screened outcomes
# the share is not assessed.
BROAD_SHARE = 0.25
SPECIFIC_SHARE = 0.10
SPECIFICITY_MIN_OUTCOMES = 10
# The replication atlas: every (construct, outcome) pair of the screen, its literature counted; at most this many
# lookups per run (the largest outcomes first; answers are cached across runs).
ATLAS_MAX_LOOKUPS = int(os.getenv("CODAS_ATLAS_MAX_LOOKUPS", "800"))
_HALF_SEED = 20251024
_BASE = ("person_id", "last_seen_day", "age", "female")
SCAN_FILE = "phenome_scan.json"
FULL_SCAN_FILE = "phenome_scan_full_cohort.json"
FULL_PREREG = "phenome_scan_full_cohort_prereg.json"


def _bucket(n: int) -> int | str:
    return MASK if 1 <= int(n) <= SMALL_CELL_MAX else int(n)


# --- the three tables ----------------------------------------------------------------------------

def _shared_phenotypes(spec: CohortSpec) -> bool:
    return bool(spec.phenotypes) and os.getenv("CODAS_PHENOTYPE_CACHE", "1") != "0"


def person_sql(spec: CohortSpec, cdr: str, everyone: bool = False) -> tuple[str, list[QueryParam]]:
    """Every discovery participant's person-level columns (every participant's with ``everyone``), by the
    extraction's own code."""
    return build_cohort_sql(spec, cdr, phenotypes=not _shared_phenotypes(spec), person_level=True, everyone=everyone)


def _people(spec: CohortSpec, t: Any, params: list[QueryParam], everyone: bool = False) -> list[str]:
    """The discovery share's index dates (the sealed share is left out here, in BigQuery); every participant's
    with ``everyone`` (the full-cohort scan)."""
    valid = _valid_rule(spec, params)
    params.append(("window_days", "INT64", int(spec.window_days)))
    where = ""
    if spec.holdout_fraction > 0 and not everyone:
        params += [("holdout_permille", "INT64", int(round(spec.holdout_fraction * 1000))),
                   ("split_salt", "INT64", int(spec.split_salt))]
        where = f"\n  WHERE {cohort.discovery_condition(spec)}"
    return [_first_wear_cte(spec.wearables[0], t, valid),
            "people AS (\n  SELECT person_id, DATE_ADD(first_wear_date, INTERVAL @window_days DAY) AS index_date\n"
            f"  FROM first_wear{where}\n)"]


def _outcome_map(outcomes: list[Any], t: Any, params: list[QueryParam]) -> list[str]:
    """Each outcome's concepts (``outcome`` is its position in ``outcomes``), with their descendants when the
    outcome includes them, as the extraction's concept set does."""
    seeds = []
    for i, o in enumerate(outcomes):
        params.append((f"o{i}_ids", "ARRAY<INT64>", [int(c) for c in o.concept_ids]))
        descend = "TRUE" if o.outcome_spec().include_descendants else "FALSE"
        seeds.append(f"SELECT {i} AS outcome, concept_id, {descend} AS descend FROM UNNEST(@o{i}_ids) AS concept_id")
    return ["seeds AS (\n  " + "\n  UNION ALL\n  ".join(seeds) + "\n)",
            "outcome_map AS (\n  SELECT outcome, concept_id FROM seeds\n  UNION DISTINCT\n"
            "  SELECT s.outcome, ca.descendant_concept_id AS concept_id FROM seeds s\n"
            f"  JOIN {t('concept_ancestor')} ca ON ca.ancestor_concept_id = s.concept_id\n  WHERE s.descend\n)"]


def condition_matrix_sql(outcomes: list[Any], spec: CohortSpec, cdr: str,
                         everyone: bool = False) -> tuple[str, list[QueryParam]]:
    """For every discovery participant and catalog condition they were ever diagnosed with: the first
    diagnosis in days from the index date and the number of distinct diagnosis dates."""
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = []
    ctes = _people(spec, t, params, everyone) + _outcome_map(outcomes, t, params) + [
        "dx AS (\n"
        "  SELECT co.person_id, om.outcome, MIN(co.condition_start_date) AS dx_date,\n"
        "    COUNT(DISTINCT co.condition_start_date) AS dx_dates\n"
        f"  FROM {t('condition_occurrence')} co\n"
        "  JOIN outcome_map om ON om.concept_id = co.condition_concept_id\n"
        "  JOIN people p ON p.person_id = co.person_id\n"
        "  GROUP BY co.person_id, om.outcome\n)"]
    sql = ("WITH " + ",\n".join(ctes) + "\n"
           "SELECT dx.person_id, dx.outcome, DATE_DIFF(dx.dx_date, p.index_date, DAY) AS dx_day, dx.dx_dates\n"
           "FROM dx JOIN people p ON p.person_id = dx.person_id")
    return sql, params


def lab_matrix_sql(outcomes: list[Any], spec: CohortSpec, cdr: str,
                   everyone: bool = False) -> tuple[str, list[QueryParam]]:
    """For every discovery participant and catalog laboratory outcome: the first value from index + lag (and
    before index + lag + horizon, when the spec has one) within the outcome's bounds and units."""
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = [("outcome_from", "INT64", int(spec.lag_days))]
    horizon = ""
    if spec.horizon_days is not None:
        params.append(("outcome_until", "INT64", int(spec.lag_days + spec.horizon_days)))
        horizon = "\n    AND m.measurement_date < DATE_ADD(p.index_date, INTERVAL @outcome_until DAY)"
    rules = []
    for i, o in enumerate(outcomes):
        parts = []
        if o.value_min is not None:
            parts.append(f"m.value_as_number >= @o{i}_min")
            params.append((f"o{i}_min", "FLOAT64", float(o.value_min)))
        if o.value_max is not None:
            parts.append(f"m.value_as_number <= @o{i}_max")
            params.append((f"o{i}_max", "FLOAT64", float(o.value_max)))
        if o.unit_concept_ids:
            parts.append(f"m.unit_concept_id IN UNNEST(@o{i}_units)")
            params.append((f"o{i}_units", "ARRAY<INT64>", [int(u) for u in o.unit_concept_ids]))
        if parts:
            rules.append(f"      WHEN {i} THEN " + " AND ".join(parts))
    rule = ("\n    AND CASE om.outcome\n" + "\n".join(rules) + "\n      ELSE TRUE END") if rules else ""
    ctes = _people(spec, t, params, everyone) + _outcome_map(outcomes, t, params) + [
        "firsts AS (\n"
        "  SELECT m.person_id, om.outcome, m.value_as_number AS value,\n"
        "    ROW_NUMBER() OVER (PARTITION BY m.person_id, om.outcome ORDER BY m.measurement_date, m.measurement_id) AS rn\n"
        f"  FROM {t('measurement')} m\n"
        "  JOIN outcome_map om ON om.concept_id = m.measurement_concept_id\n"
        "  JOIN people p ON p.person_id = m.person_id\n"
        "  WHERE m.value_as_number IS NOT NULL\n"
        f"    AND m.measurement_date >= DATE_ADD(p.index_date, INTERVAL @outcome_from DAY){horizon}{rule}\n)"]
    sql = "WITH " + ",\n".join(ctes) + "\nSELECT person_id, outcome, value FROM firsts WHERE rn = 1"
    return sql, params


def _cached(client: Any, sql: str, params: list[QueryParam], cdr: str, out: Path, kind: str) -> pd.DataFrame:
    """A participant-level table, queried once per CDR and definition and kept in the sealed folder (it
    holds person ids)."""
    key = hashlib.sha256(json.dumps([cdr, sql, params], default=str).encode()).hexdigest()[:16]
    path = sealed_dir(out) / "phenome_cache" / f"{kind}_{key}.csv"
    if not path.exists():
        frame = cohort._query(client, sql, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").replace(path)  # complete or absent, never half-written
    return pd.read_csv(path, float_precision="round_trip")


def person_table(spec: CohortSpec, cdr: str, client: Any, out: Path, everyone: bool = False) -> pd.DataFrame:
    """The discovery share's person table (every participant's with ``everyone``), with the phenotypes computed
    once per run merged in."""
    frame = _cached(client, *person_sql(spec, cdr, everyone), cdr, out, "people")
    if _shared_phenotypes(spec):
        frame = frame.merge(cohort._phenotype_table(client, spec, cdr, out), on="person_id", how="left",
                            validate="one_to_one")
    return frame.sort_values("person_id", kind="mergesort").reset_index(drop=True)


# --- each outcome's cohort, by the extraction's rules ----------------------------------------------

def condition_cohort(people: pd.DataFrame, dx: pd.DataFrame, index: int, lag_days: int,
                     min_occurrences: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Who is in one diagnosis's cohort (``build_cohort_sql``, incident design), who has the event, and the
    days of follow-up: follow-up starts at index + lag; a person needs an EHR visit; a diagnosis before
    follow-up starts makes the person prevalent and leaves them out, as do fewer than ``min_occurrences``
    diagnosis dates; follow-up runs to the first diagnosis or the last visit and must be positive."""
    own = dx.loc[dx["outcome"] == index, ["person_id", "dx_day", "dx_dates"]]
    frame = people[["person_id", "last_seen_day"]].merge(own, on="person_id", how="left", validate="one_to_one")
    dx_day = frame["dx_day"].to_numpy(dtype=float)
    last = frame["last_seen_day"].to_numpy(dtype=float)
    case = np.isfinite(dx_day)
    enough = (frame["dx_dates"].fillna(0).to_numpy() >= min_occurrences if min_occurrences > 1
              else np.ones(len(frame), dtype=bool))
    with np.errstate(invalid="ignore"):
        eligible = np.isfinite(last) & np.where(case, (dx_day > lag_days) & enough, last > lag_days)
    return eligible, case, np.where(case, dx_day, last) - lag_days


def lab_values(people: pd.DataFrame, labs: pd.DataFrame, index: int) -> np.ndarray:
    """Each discovery participant's value of one laboratory outcome (missing: not in its cohort)."""
    own = labs.loc[labs["outcome"] == index, ["person_id", "value"]]
    return people[["person_id"]].merge(own, on="person_id", how="left", validate="one_to_one")["value"].to_numpy(float)


# --- the statistics --------------------------------------------------------------------------------

def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """``columns`` as a complete, standardised design: a missing value is filled with the median and flagged
    by an indicator; constant and collinear columns are left out."""
    parts = []
    for c in columns:
        v = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(v)
        if missing.all():
            continue
        parts.append(np.where(missing, np.nanmedian(v), v))
        if missing.any():
            parts.append(missing.astype(float))
    if not parts:
        return np.zeros((len(frame), 0))
    x = np.column_stack(parts)
    sd = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    # Keep a column unless it is constant or (nearly) a linear combination of the columns kept before it,
    # judged on their correlation matrix: one product of the design, not a rank per column.
    gram = x.T @ x / len(x)
    kept: list[int] = []
    for j in range(x.shape[1]):
        if not sd[j] > 0:
            continue
        rest = gram[j, j] - (gram[j, kept] @ np.linalg.solve(gram[np.ix_(kept, kept)], gram[kept, j]) if kept else 0.0)
        if rest > 1e-8:
            kept.append(j)
    return x[:, kept]


def _signed_z(rho: float, p: float) -> float:
    if not (np.isfinite(rho) and np.isfinite(p)):
        return math.nan
    return float(math.copysign(stats.norm.isf(max(p, 1e-300) / 2), rho)) if rho else 0.0


def _condition_tests(frame: pd.DataFrame, features: list[str], eligible: np.ndarray, event: np.ndarray,
                     time: np.ndarray, base: list[str], adjusted: list[str], half: np.ndarray) -> list[dict[str, Any]]:
    rows = eligible & frame[base].notna().all(axis=1).to_numpy()
    idx = np.flatnonzero(rows)[np.argsort(-time[rows], kind="mergesort")]
    t, e = time[idx], event[idx]
    x = frame[features].to_numpy(dtype=float)[idx]
    part = frame.iloc[idx]
    z0, z1 = _design(part, base), _design(part, base + adjusted)
    full = score_tests(t, e, z0, x, breslow_fit(t, e, z0))
    t1 = score_tests(t, e, z1, x, breslow_fit(t, e, z1))
    halves = []
    for h in (True, False):
        sel = half[idx] == h
        halves.append(score_tests(t[sel], e[sel], z0[sel], x[sel], breslow_fit(t[sel], e[sel], z0[sel])))
    out = []
    for j, feature in enumerate(features):
        hr = effects._ratio(full[j]["log_hr"])
        out.append({"feature": feature, "n": full[j]["n"], "events": full[j]["events"], "measure": "hr_per_sd",
                    "estimate": hr, "z": full[j]["z"], "p": full[j]["p"],
                    "t1_estimate": effects._ratio(t1[j]["log_hr"]), "t1_z": t1[j]["z"], "t1_p": t1[j]["p"],
                    "half_z": [halves[0][j]["z"], halves[1][j]["z"]], "magnitude": effects.magnitude("hr_per_sd", hr)})
    return out


def partial_spearman_many(x: np.ndarray, y: np.ndarray, cov: np.ndarray) -> list[tuple[float, float, int]]:
    """``codas.core.statistics.partial_spearman`` of each column of ``x`` with ``y`` given ``cov`` (complete,
    like ``y``), the same estimator for many columns at once: on the rows where the column is observed, the
    ranks of the column, of ``y`` and of each covariate; the residuals of the first two on the covariates'
    ranks; the Spearman correlation of the residuals, with df = n - 2 - k. Columns missing on the same rows
    share the covariates' ranks and fit."""
    x = np.asarray(x, dtype=float)
    k = cov.shape[1]
    observed_all = np.isfinite(x)
    groups: dict[bytes, list[int]] = {}
    for j in range(x.shape[1]):
        groups.setdefault(np.packbits(observed_all[:, j]).tobytes(), []).append(j)
    out: list[tuple[float, float, int]] = [(math.nan, math.nan, 0)] * x.shape[1]
    for cols in groups.values():
        rows = observed_all[:, cols[0]]
        n = int(rows.sum())
        if n < max(10, k + 5):
            for j in cols:
                out[j] = (math.nan, math.nan, n)
            continue
        design = np.column_stack([np.ones(n)] + [stats.rankdata(c) for c in cov[rows].T])
        ranked = np.column_stack([stats.rankdata(y[rows])] + [stats.rankdata(x[rows, j]) for j in cols])
        resid = ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
        r = np.apply_along_axis(stats.rankdata, 0, resid)
        r -= r.mean(axis=0)
        sd = np.sqrt((r * r).mean(axis=0))
        df = n - 2 - k
        for i, j in enumerate(cols, start=1):
            if sd[0] < 1e-12 or sd[i] < 1e-12:
                out[j] = (math.nan, math.nan, n)
                continue
            rho = float((r[:, 0] * r[:, i]).mean() / (sd[0] * sd[i]))
            if df <= 0 or abs(rho) >= 1.0:
                out[j] = (rho, math.nan, n)
                continue
            t = rho * math.sqrt(df / (1.0 - rho * rho))
            out[j] = (rho, float(2.0 * stats.t.sf(abs(t), df)), n)
    return out


def _lab_tests(frame: pd.DataFrame, features: list[str], y: np.ndarray, base: list[str], adjusted: list[str],
               half: np.ndarray) -> list[dict[str, Any]]:
    rows = np.isfinite(y) & frame[base].notna().all(axis=1).to_numpy()
    part, yy, h = frame.loc[rows], y[rows], half[rows]
    z0, z1 = _design(part, base), _design(part, base + adjusted)
    x = part[features].to_numpy(dtype=float)
    full, t1 = partial_spearman_many(x, yy, z0), partial_spearman_many(x, yy, z1)
    halves = [partial_spearman_many(x[h == k], yy[h == k], z0[h == k]) for k in (True, False)]
    out = []
    for j, feature in enumerate(features):
        (rho, p, n), (rho1, p1, _) = full[j], t1[j]
        out.append({"feature": feature, "n": int(n), "events": None, "measure": "rho",
                    "estimate": None if not np.isfinite(rho) else round(float(rho), 4), "z": _signed_z(rho, p),
                    "p": p, "t1_estimate": None if not np.isfinite(rho1) else round(float(rho1), 4),
                    "t1_z": _signed_z(rho1, p1), "t1_p": p1,
                    "half_z": [_signed_z(*halves[0][j][:2]), _signed_z(*halves[1][j][:2])],
                    "magnitude": effects.magnitude("rho", rho)})
    return out


def is_hit(pair: dict[str, Any]) -> bool:
    """q below Q_LEVEL, a size that is not negligible, the same sign with one-sided p below HALF_P in each
    half, and the same sign with p below T1_P after the covariates and wear adherence."""
    z = pair.get("z")
    if not isinstance(z, float) or not np.isfinite(z) or z == 0:
        return False
    sign = math.copysign(1.0, z)
    halves_agree = all(isinstance(h, float) and np.isfinite(h) and h * sign > stats.norm.isf(HALF_P)
                       for h in pair.get("half_z") or [])
    t1 = pair.get("t1_z")
    return bool(pair.get("q", 1.0) < Q_LEVEL and pair.get("magnitude") not in (None, "negligible") and halves_agree
                and len(pair.get("half_z") or []) == 2 and isinstance(t1, float) and np.isfinite(t1)
                and t1 * sign > 0 and (pair.get("t1_p") or 1.0) < T1_P)


def specificity(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per feature: the outcomes it was screened against, those it is a hit for, their share (breadth), the
    median |z| after all covariates over every screened outcome, and its class (``specific``, ``shared``,
    ``broad`` or ``not assessed``; module constants)."""
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_feature.setdefault(pair["feature"], []).append(pair)
    out = {}
    for feature, rows in by_feature.items():
        z = [abs(p["t1_z"]) for p in rows if isinstance(p.get("t1_z"), float) and np.isfinite(p["t1_z"])]
        hits = len({p["outcome"] for p in rows if p.get("hit")})
        breadth = hits / len(rows) if rows else None
        kind = ("not assessed" if len(rows) < SPECIFICITY_MIN_OUTCOMES else "broad" if breadth > BROAD_SHARE
                else "specific" if breadth <= SPECIFIC_SHARE else "shared")
        out[feature] = {"outcomes": len(rows), "hits": hits, "breadth": breadth,
                        "median_abs_t1_z": float(np.median(z)) if z else None, "specificity": kind}
    return out


def outcome_profile(result: dict[str, Any], outcome: str) -> dict[str, dict[str, Any]]:
    """Each screened feature's specificity class and breadth (``specificity``), and whether it is a hit for
    ``outcome``: what the scorecard reports for a registered feature the scan screened."""
    hit = {p["feature"] for p in result.get("pairs") or [] if p["outcome"] == outcome and p.get("hit")}
    return {f: {"specificity": v["specificity"], "breadth": v["breadth"], "hit_here": f in hit}
            for f, v in (result.get("specificity") or {}).items()}


def features_of(frame: pd.DataFrame, spec: CohortSpec) -> list[str]:
    """The person table's wearable features: not identifiers, demographics, covariates or wear-adherence
    counts, and not constant."""
    skip = {*_BASE, *availability_columns(spec), *(c.name for c in spec.covariates)}
    numeric = [c for c in frame.columns if c not in skip and pd.api.types.is_numeric_dtype(frame[c])]
    return [c for c in numeric if frame[c].nunique(dropna=True) > 1]


# --- the screen ------------------------------------------------------------------------------------

def _frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
                          + ",".join(map(str, frame.columns)).encode()).hexdigest()[:16]


def _code_digest() -> str:
    from codas.core import statistics, survival

    text = b"".join(Path(m.__file__).read_bytes() for m in (statistics, survival)) + Path(__file__).read_bytes()
    return hashlib.sha256(text).hexdigest()[:12]


def _literature(result: dict[str, Any], frame_features: list[str], spec: CohortSpec, out: Path, log: Any) -> str:
    """The literature counts for each outcome's strongest hits: whether the hit's construct is established
    for the outcome. Only the catalog's construct and outcome terms leave the machine."""
    from codas.aou import runner
    from codas.core import literature

    if not runner.literature_search_enabled() or not runner.LITERATURE_TERMS.exists():
        return "off"
    terms = literature.load_terms(runner.LITERATURE_TERMS)
    construct_of = literature.constructs_of(frame_features, terms, {p.prefix: p.kind for p in spec.phenotypes})
    cache = literature.Cache(str(out / "literature_cache.jsonl"))
    by_outcome: dict[str, list[dict[str, Any]]] = {}
    for pair in result["pairs"]:
        if pair["hit"]:
            by_outcome.setdefault(pair["outcome"], []).append(pair)
    status = "available"
    labels = {o["name"]: o["label"] for o in result["outcomes"]}
    for name, hits in by_outcome.items():
        outcome_terms = (terms.get("outcomes") or {}).get(name) or [labels[name]]
        seen: dict[str, dict[str, Any]] = {}
        for pair in sorted(hits, key=lambda p: -abs(p["t1_z"]))[:LITERATURE_HITS]:
            construct = construct_of.get(pair["feature"])
            if construct is None:
                pair["novelty"] = literature.NOT_ASSESSED
                continue
            if construct not in seen:
                entry = (terms.get("constructs") or {}).get(construct) or {}
                seen[construct] = literature.evidence(construct, entry.get("synonyms") or [], outcome_terms, cache=cache)
            evidence = seen[construct]
            pair.update(construct=construct, novelty=evidence["novelty"], papers=evidence.get("papers"))
            if evidence["novelty"] == literature.NOT_ASSESSED:
                status = f"unavailable: {evidence.get('error')}"
                log(f"literature not reachable for the scan ({evidence.get('error')}); novelty not assessed")
                return status
    return status


def replication_atlas(result: dict[str, Any], people: pd.DataFrame, features: list[str], spec: CohortSpec,
                      out: Path, log: Any = print) -> dict[str, Any]:
    """Published associations re-tested on the discovery share. Every construct the catalog names among the
    features is paired with every screened outcome and the literature is counted for the pair; a pair the
    literature has established is reported with its representative feature's estimate after all covariates
    (the construct's feature with the most participants, chosen without the outcome), Benjamini-Hochberg q
    across the atlas, and the effects the most cited papers state, quoted from their abstracts. The pairs come
    from the literature, not from this screen, so what replicates and what does not are both reported. The
    discovery share was never used to make these claims, so it tests them independently."""
    from codas.aou import runner
    from codas.core import literature

    if not runner.literature_search_enabled() or not runner.LITERATURE_TERMS.exists():
        return {"status": "off", "pairs": []}
    terms = literature.load_terms(runner.LITERATURE_TERMS)
    construct_of = literature.constructs_of(features, terms, {p.prefix: p.kind for p in spec.phenotypes})
    members: dict[str, list[str]] = {}
    for feature, construct in construct_of.items():
        members.setdefault(construct, []).append(feature)
    present = {f: int(people[f].notna().sum()) for f in construct_of}
    representative = {c: sorted(fs, key=lambda f: (-present[f], f))[0] for c, fs in members.items()}
    by_pair = {(p["outcome"], p["feature"]): p for p in result["pairs"]}
    labels = {r["name"]: r["label"] for r in result["outcomes"]}
    tested = sorted((r for r in result["outcomes"] if r.get("tested")), key=lambda r: (-int(r["n"]), r["name"]))
    cache = literature.Cache(str(out / "literature_cache.jsonl"))
    rows, lookups, status = [], 0, "available"
    for outcome in tested:
        outcome_terms = (terms.get("outcomes") or {}).get(outcome["name"]) or [labels[outcome["name"]]]
        for construct in members:
            if lookups >= ATLAS_MAX_LOOKUPS:
                status = f"available; stopped at {ATLAS_MAX_LOOKUPS} lookups (CODAS_ATLAS_MAX_LOOKUPS)"
                break
            lookups += 1
            synonyms = ((terms.get("constructs") or {}).get(construct) or {}).get("synonyms") or []
            evidence = literature.evidence(construct, synonyms, outcome_terms, cache=cache)
            if evidence["novelty"] == literature.NOT_ASSESSED:
                log(f"literature not reachable for the replication atlas ({evidence.get('error')})")
                return {"status": f"unavailable: {evidence.get('error')}", "pairs": []}
            if evidence["novelty"] != "established":
                continue
            pair = by_pair.get((outcome["name"], representative[construct])) or {}
            try:
                quoted = literature.published_effects(evidence["query"], [construct, *synonyms, *outcome_terms],
                                                      cache=cache)
            except literature.Unreachable:
                quoted = []
            rows.append({"outcome": outcome["name"], "label": outcome["label"], "construct": construct,
                         "feature": representative[construct], "papers": evidence.get("papers"),
                         "reviews": evidence.get("reviews"), "measure": pair.get("measure"),
                         "t1_estimate": pair.get("t1_estimate"), "t1_z": pair.get("t1_z"), "t1_p": pair.get("t1_p"),
                         "magnitude": pair.get("magnitude"), "hit": bool(pair.get("hit")), "published": quoted,
                         "top_cited": [t.get("key") for t in evidence.get("top_cited") or []]})
        else:
            continue
        break
    for row, q in zip(rows, benjamini_hochberg([r["t1_p"] if r["t1_p"] is not None else 1.0 for r in rows])):
        row["q"] = float(q)
    return {"status": status, "lookups": lookups, "pairs": rows,
            "associated": sum(r["q"] < Q_LEVEL for r in rows)}


def scan(outcomes: list[Any], spec: CohortSpec, cdr: str, client: Any, out: Path, *, min_events: int,
         min_n: int, log: Any = print, everyone: bool = False) -> dict[str, Any]:
    """Screen every verified catalog outcome against every feature on the discovery share (module doc).
    ``outcomes`` are catalog entries; diagnoses need ``min_events`` events and laboratory values ``min_n``
    participants in the discovery share to be screened. The result is cached by its inputs.

    With ``everyone`` it is the full-cohort scan: the same screen on every participant, sealed share included,
    as a confirmatory phenome-wide study of its own. Its settings, outcomes and features are written to
    ``FULL_PREREG`` before any test, its result goes to ``FULL_SCAN_FILE``, and every outcome it screens is
    recorded as opened in the ledgers of both shares (``mark_opened``): after it, no sealed half can confirm
    anything about those outcomes. Run it last."""
    if spec.design != "incident":
        raise ValueError("The phenome scan needs the incident design, whose index date is a person's own.")
    cdr = resolve_cdr(cdr)
    out = Path(out)
    conditions = [o for o in outcomes if o.kind == "condition"]
    labs = [o for o in outcomes if o.kind == "measurement"]
    people = person_table(spec, cdr, client, out, everyone)
    dx = (_cached(client, *condition_matrix_sql(conditions, spec, cdr, everyone), cdr, out, "diagnoses")
          if conditions else pd.DataFrame(columns=["person_id", "outcome", "dx_day", "dx_dates"]))
    values = (_cached(client, *lab_matrix_sql(labs, spec, cdr, everyone), cdr, out, "labs") if labs
              else pd.DataFrame(columns=["person_id", "outcome", "value"]))
    features = features_of(people, spec)
    settings = {"q": Q_LEVEL, "half_p": HALF_P, "t1_p": T1_P, "min_events": int(min_events), "min_n": int(min_n),
                "literature_hits": LITERATURE_HITS, **({"participants": "every"} if everyone else {})}
    if everyone:  # fixed on disk before any test is computed
        record = {"settings": settings, "features": features, "outcomes": [o.name for o in outcomes],
                  "code": _code_digest(), "cdr": cdr}
        record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
        (out / FULL_PREREG).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    identity = [[o.name, o.kind, sorted(int(c) for c in o.concept_ids), o.value_min, o.value_max, o.min_occurrences,
                 sorted(o.unit_concept_ids)] for o in outcomes]
    digest = hashlib.sha256(json.dumps([cdr, [_frame_digest(f) for f in (people, dx, values)], identity, settings,
                                        features, _code_digest()], default=str).encode()).hexdigest()[:16]
    # The raw result (its counts unmasked, for the ranking) stays in the sealed folder; the copy beside the
    # reports, which the notebook reads, has counts of 1-20 masked.
    cached = sealed_dir(out) / "phenome_cache" / f"phenome_scan_{digest}.json"
    shown = out / (FULL_SCAN_FILE if everyone else SCAN_FILE)
    if cached.exists():
        result = json.loads(cached.read_text(encoding="utf-8"))
        shown.write_text(json.dumps(mask_small_counts(result), indent=1), encoding="utf-8")
        return result
    log(f"phenome scan: {len(features)} features x {len(outcomes)} outcomes on {len(people):,} "
        + ("participants, sealed share included (full-cohort scan)" if everyone else "discovery participants"))
    half = np.random.default_rng(_HALF_SEED).random(len(people)) < 0.5
    covariates = [c.name for c in spec.covariates if c.name in people.columns]
    adherence = [c for c in availability_columns(spec) if c in people.columns]
    base = [c for c in ("age", "female") if c in people.columns]
    # Under the incident design each outcome's spec keeps every covariate (codas.aou.feasibility.spec_for): a
    # baseline value of the outcome is a covariate like any other, and a prior diagnosis has excluded the person.
    adjusted = covariates + adherence
    position = {o.name: i for i, o in enumerate(conditions)} | {o.name: i for i, o in enumerate(labs)}
    rows, pairs = [], []
    for o in outcomes:
        row: dict[str, Any] = {"name": o.name, "label": o.label, "kind": o.kind}
        if o.kind == "condition":
            eligible, event, time = condition_cohort(people, dx, position[o.name], spec.lag_days, o.min_occurrences)
            n, events = int(eligible.sum()), int((eligible & event).sum())
            row.update(n=n, events=events)
            enough = events >= min_events
            tests = (_condition_tests(people, features, eligible, event, time, base, adjusted, half) if enough else [])
        else:
            y = lab_values(people, values, position[o.name])
            n = int(np.isfinite(y).sum())
            row.update(n=n, events=None)
            enough = n >= min_n
            tests = _lab_tests(people, features, y, base, adjusted, half) if enough else []
        row.update(tested=enough, reason=None if enough else (
            f"fewer than {min_events} events in the discovery share" if o.kind == "condition"
            else f"fewer than {min_n} participants in the discovery share"))
        rows.append(row)
        pairs += [{"outcome": o.name, "kind": o.kind, **p} for p in tests]
    for pair, q in zip(pairs, benjamini_hochberg([p["p"] for p in pairs])):
        pair["q"] = float(q)
        pair["hit"] = is_hit(pair)
    result = {"settings": settings, "features": len(features), "participants": len(people), "outcomes": rows,
              "pairs": pairs, "rule": rule_text(settings), "tested_pairs": len(pairs)}
    result["literature"] = _literature(result, features, spec, out, log)
    profile = specificity(pairs)
    for pair in pairs:
        if pair["hit"]:
            f = profile[pair["feature"]]
            median = f["median_abs_t1_z"]
            pair.update(breadth=f["breadth"], specificity=f["specificity"],
                        relative_strength=abs(pair["t1_z"]) / median if median else None)
    result["specificity"] = profile
    result["atlas"] = replication_atlas(result, people, features, spec, out, log)
    for row in rows:
        hits = [p for p in pairs if p["outcome"] == row["name"] and p["hit"]]
        hits.sort(key=lambda p: -abs(p["t1_z"]))
        row.update(hits=len(hits), novel_hits=sum(p.get("novelty") in ("emerging", "unexplored") for p in hits),
                   specific_hits=sum(p.get("specificity") == "specific" for p in hits),
                   best=({k: hits[0].get(k) for k in ("feature", "measure", "estimate", "q", "t1_estimate", "t1_z",
                                                      "novelty")} if hits else None))
    text = json.dumps(_json_safe(result), indent=1)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    shown.write_text(json.dumps(mask_small_counts(json.loads(text)), indent=1), encoding="utf-8")
    log(f"phenome scan: {len(pairs):,} pairs tested, {sum(p['hit'] for p in pairs)} hits across "
        f"{sum(r['hits'] > 0 for r in rows)} outcomes")
    return json.loads(text)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def rule_text(settings: dict[str, Any]) -> str:
    return (f"Screened on the discovery share only. A hit has q < {settings['q']} (Benjamini-Hochberg across every "
            "tested pair, adjusted for age and sex), a size that is not negligible (hazard ratio per SD beyond "
            "0.95-1.05; |rho| at least 0.1), the same sign with one-sided p < "
            f"{settings['half_p']} in each of two fixed halves, and the same sign with p < {settings['t1_p']} "
            "after the covariates and wear adherence. Diagnoses need at least "
            f"{settings['min_events']} events and lab values {settings['min_n']} participants in the discovery "
            "share. Outcomes are ranked by hits whose construct the literature has not established (Europe PMC "
            f"counts for the {settings['literature_hits']} strongest), then by hits whose feature singles out few "
            f"outcomes (a hit for at most {SPECIFIC_SHARE:.0%} of those screened, rather than a general marker of "
            f"poor health that is a hit for more than {BROAD_SHARE:.0%}), then by hits, then by the strongest "
            "adjusted statistic; outcomes already tested on this holdout are skipped, and only outcomes with a "
            "hit are selected.")


# --- selection and display --------------------------------------------------------------------------

def select(result: dict[str, Any], opened: set[str], *, top_k: int, per_kind: bool = True) -> list[dict[str, Any]]:
    """The outcomes to run, best first: screened, with at least one hit, not already tested on this holdout."""
    pool = [r for r in result["outcomes"] if r.get("tested") and r.get("hits") and r["name"] not in opened]
    ranked = sorted(pool, key=lambda r: (-r["novel_hits"], -r.get("specific_hits", 0), -r["hits"],
                                         -abs((r.get("best") or {}).get("t1_z") or 0), r["name"]))
    if not per_kind:
        return ranked[:top_k]
    kinds = list(dict.fromkeys(r["kind"] for r in ranked))
    return [r for kind in kinds for r in [x for x in ranked if x["kind"] == kind][:top_k]]


def feasibility_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The screen's discovery counts in the feasibility screen's shape (raw integers, for ranking only)."""
    rows = []
    for r in result["outcomes"]:
        row = {"name": r["name"], "label": r["label"], "kind": r["kind"], "n_participants": int(r["n"])}
        if r["kind"] == "condition":
            row.update(n_cases=int(r["events"]), n_controls=int(r["n"]) - int(r["events"]))
        rows.append(row)
    return rows


def _effect(measure: str | None, value: Any) -> str:
    if value is None:
        return "-"
    return f"HR {value:.2f} per SD" if measure == "hr_per_sd" else f"rho {value:+.2f}"


def scan_table(result: dict[str, Any], skipped: dict[str, str | None] | None = None,
               chosen: list[dict[str, Any]] | None = None, opened: set[str] | None = None,
               limit: int | None = None) -> str:
    """The screen as a Markdown table, counts of 1-20 masked; safe to display. With ``limit`` only the selected
    outcomes and the others with the most hits, up to that many rows, and a count of the rest."""
    picked = {r["name"] for r in chosen or []}
    lines = [f"{result['tested_pairs']:,} feature-outcome pairs on the discovery share "
             f"({result['features']} features). {result['rule']}", "",
             "| outcome | kind | discovery participants | events | hits | not established | specific | strongest hit | "
             "selected |", "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(result["outcomes"], key=lambda r: (r["name"] not in picked, -(r.get("hits") or 0), r["name"]))
    failed = [(name, reason) for name, reason in (skipped or {}).items() if reason]
    hidden = 0
    if limit is not None and len(order) > limit:
        hidden, order = len(order) - limit, order[:limit]
    for r in order:
        best = r.get("best") or {}
        strongest = (f"{best['feature']} ({_effect(best.get('measure'), best.get('estimate'))}, q {best['q']:.1e})"
                     if best else "-")
        selected = ("yes" if r["name"] in picked else "already tested on this holdout" if r["name"] in (opened or set())
                    else (r.get("reason") or ""))
        events = "-" if r.get("events") is None else _bucket(r["events"])
        lines.append(f"| {r['label']} | {r['kind']} | {_bucket(r['n'])} | {events} | {r.get('hits', 0)} | "
                     f"{r.get('novel_hits', 0)} | {r.get('specific_hits', 0)} | {strongest} | {selected} |")
    if limit is not None and failed:
        hidden += len(failed)
        failed = []
    for name, reason in failed:
        lines.append(f"| {name} | skipped | - | - | - | - | - | - | {reason} |")
    if hidden:
        lines += ["", f"{hidden} more outcomes (fewer hits, or skipped by the concept check) are listed in "
                      "explore_summary.md."]
    return "\n".join(lines)


def hits_table(result: dict[str, Any], limit: int = SHOWN_HITS) -> str:
    """The strongest hits across the matrix, as a Markdown table."""
    hits = sorted((p for p in result["pairs"] if p["hit"]), key=lambda p: -abs(p["t1_z"]))[:limit]
    if not hits:
        return "No feature-outcome pair met every condition of a hit."
    labels = {r["name"]: r["label"] for r in result["outcomes"]}
    lines = ["| outcome | feature | T0 | q | halves (z) | adjusted T1 | literature | specificity |",
             "|---|---|---|---|---|---|---|---|"]
    for p in hits:
        literature = (f"{p['novelty']} ({p.get('papers')} papers)" if p.get("papers") is not None
                      else p.get("novelty") or "not looked up")
        lines.append(f"| {labels.get(p['outcome'], p['outcome'])} | {p['feature']} | {_effect(p['measure'], p['estimate'])} "
                     f"| {p['q']:.1e} | {p['half_z'][0]:+.1f} / {p['half_z'][1]:+.1f} | "
                     f"{_effect(p['measure'], p['t1_estimate'])} | {literature} | {_specificity_text(p)} |")
    return "\n".join(lines)


def atlas_table(result: dict[str, Any], limit: int | None = None) -> str:
    """The replication atlas as a Markdown table: each association the literature has established, its
    estimate on the discovery share and q across the atlas, and what the most cited papers state."""
    atlas = result.get("atlas") or {}
    rows = sorted(atlas.get("pairs") or [], key=lambda r: (r.get("q", 1.0), r["label"], r["construct"]))
    if not rows:
        return f"Replication atlas: no established association to re-test ({atlas.get('status', 'not run')})."
    shown = rows if limit is None else rows[:limit]
    lines = [f"Replication atlas: {len(rows)} associations the literature has established (Europe PMC counts), "
             f"re-tested on the discovery share after all covariates; {atlas.get('associated', 0)} with q < "
             f"{Q_LEVEL} across the atlas. A published effect is quoted from its paper's abstract; its units and "
             "contrast are the paper's.", "",
             "| outcome | construct (feature) | papers (reviews) | estimate here | q | published, as stated |",
             "|---|---|---|---|---|---|"]
    for r in shown:
        quote = "; ".join(f"\"{x['statement']}\" [{x['key']}]" for x in (r.get("published") or [])[:1]) or "-"
        lines.append(f"| {r['label']} | {r['construct']} ({r['feature']}) | {r.get('papers')} ({r.get('reviews')}) | "
                     f"{_effect(r.get('measure'), r.get('t1_estimate'))} | {r['q']:.1e} | {quote} |")
    if limit is not None and len(rows) > limit:
        lines += ["", f"{len(rows) - limit} more in phenome_scan.json."]
    return "\n".join(lines)


def _specificity_text(pair: dict[str, Any]) -> str:
    """``specific (hit for 4% of outcomes)``: the class and the feature's breadth."""
    kind, breadth = pair.get("specificity"), pair.get("breadth")
    if not kind:
        return "-"
    return kind if kind == "not assessed" or breadth is None else f"{kind} (hit for {breadth:.0%} of outcomes)"
