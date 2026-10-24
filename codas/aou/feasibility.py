"""Free exploration on All of Us: check a catalog of candidate outcomes, count who qualifies, and
pick the outcomes with enough participants for CoDaS to run on.

Three steps, all aggregate-only:

1. ``verify_catalog``: each outcome's concept ids must exist in the CDR concept table, sit in the
   expected domain, and carry a name containing ``expect_name``. A failing outcome is skipped and
   reported, so a wrong id never turns into an analysis.
2. ``feasibility``: for each verified outcome, the number of participants who meet the same
   wearable-window rule the extractor applies (and, for a condition, how many are cases).
3. ``rank_outcomes``: a fixed rule, not a model judgement: keep outcomes that clear a minimum
   sample (participants for a measurement, cases and controls for a condition), order by that
   sample, and take the top K, overall or for each outcome kind.

The base spec's covariates and negative controls are checked against the concept table the same
way (``verify_covariates``), and each outcome's spec carries the verified covariates and negative
controls that are not the outcome itself.

Concepts can be named by public code instead of id (``codes``: "SNOMED:44054006", "LOINC:4548-4",
"ATC:C10AA"); ``resolve_codes`` looks them up in the CDR's own vocabulary before the checks. A survey
covariate names its question by text (``question_name``); ``verify_covariates`` resolves it to exactly
one question that this CDR's survey answers use, or skips it with the questions it matched.

Counts are returned as integers for the ranking; ``feasibility_table`` masks 1-20 for display.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from codas.aou import cohort
from codas.aou.cohort import (CohortSpec, OutcomeSpec, WearableSource, build_cohort_sql, covariates_without,
                              negative_controls_without, output_columns, resolve_cdr)
from codas.aou.privacy import MASK, SMALL_CELL_MAX

_DOMAIN = {"measurement": "Measurement", "condition": "Condition", "drug": "Drug", "survey": "Observation"}
MIN_MEASUREMENT_N = 1000
MIN_CASES = 200


@dataclass
class CatalogOutcome:
    name: str
    label: str
    kind: str
    concept_ids: list[int]
    expect_name: str
    value_min: float | None = None
    value_max: float | None = None
    codes: list[str] = field(default_factory=list)

    def outcome_spec(self) -> OutcomeSpec:
        return OutcomeSpec(kind=self.kind, concept_ids=list(self.concept_ids), name=self.name,
                           value_min=self.value_min, value_max=self.value_max)


def load_catalog(path: str | Path) -> list[CatalogOutcome]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    outcomes = [CatalogOutcome(**item) for item in data["outcomes"]]
    for item in outcomes:
        OutcomeSpec(kind=item.kind, concept_ids=list(item.concept_ids), name=item.name, codes=list(item.codes)).validate()
    names = [o.name for o in outcomes]
    if len(names) != len(set(names)):
        raise ValueError("Outcome names in the catalog must be unique.")
    return outcomes


def spec_for(outcome: CatalogOutcome, base: CohortSpec, *, prefix: str = "explore") -> CohortSpec:
    """A cohort spec for ``outcome`` that reuses the wearables, window, demographics and covariates of
    ``base``; a covariate that shares a concept id with the outcome is left out."""
    target = outcome.outcome_spec()
    covariates = list(base.covariates)
    if base.design == "window":  # outcome-anchored: a covariate on the outcome's own concept is the outcome
        covariates, _dropped = covariates_without(base.covariates, target)
    spec = replace(base, name=f"{prefix}_{outcome.name}", outcome=target, wearables=list(base.wearables),
                   covariates=covariates, notes="",
                   negative_controls=negative_controls_without(base.negative_controls, target))
    return _keep_positive_controls(spec)


def _keep_positive_controls(spec: CohortSpec) -> CohortSpec:
    """``spec`` with only the positive controls that are still columns of its table."""
    columns = set(output_columns(spec))
    spec = replace(spec, positive_controls=[c for c in spec.positive_controls if c in columns])
    spec.validate()
    return spec


def resolve_codes(items: list[Any], cdr: str, client: Any, log: Any = None) -> list[Any]:
    """``items`` (catalog outcomes, covariates, negative controls) with the concept ids their ``codes``
    name in this CDR's vocabulary added. Codes are alternatives for one quantity (a lab reported under
    several formulas, a drug class by its subclasses): the ones found are kept and those this
    vocabulary lacks are left out and named through ``log``; an item with none found keeps no ids and
    fails the concept check with its codes."""
    keys = sorted({c for item in items for c in getattr(item, "codes", None) or []})
    if not keys:
        return list(items)
    sql = (f"SELECT concept_id, CONCAT(vocabulary_id, ':', concept_code) AS code_key FROM `{resolve_cdr(cdr)}.concept`\n"
           "WHERE CONCAT(vocabulary_id, ':', concept_code) IN UNNEST(@keys)")
    rows = cohort._query(client, sql, [("keys", "ARRAY<STRING>", keys)])
    found = {str(k): int(i) for i, k in zip(rows["concept_id"], rows["code_key"])}
    out = []
    for item in items:
        codes = list(getattr(item, "codes", None) or [])
        known = [c for c in codes if c in found]
        if known:
            item = replace(item, codes=known, concept_ids=sorted({*item.concept_ids, *(found[c] for c in known)}))
            if len(known) < len(codes) and log is not None:
                log(f"{item.name}: codes not in this CDR's vocabulary, left out: "
                    + ", ".join(c for c in codes if c not in found))
        elif codes:
            item = replace(item, concept_ids=[])
        out.append(item)
    return out


def resolve_survey_questions(covariates: list[Any], cdr: str, client: Any) -> tuple[list[Any], dict[str, str]]:
    """Survey covariates with ``question_name`` resolved to the single question concept whose name
    contains it among the questions this CDR's survey answers use; the others with the reason."""
    wanted = [c for c in covariates if getattr(c, "kind", "") == "survey" and c.question_name and not c.concept_ids]
    if not wanted:
        return list(covariates), {}
    cdr = resolve_cdr(cdr)
    sql = (f"SELECT c.concept_id, c.concept_name FROM `{cdr}.concept` c\n"
           f"WHERE c.concept_id IN (SELECT DISTINCT observation_source_concept_id FROM `{cdr}.observation`)\n"
           "  AND (" + " OR ".join(f"STRPOS(LOWER(c.concept_name), @q{i}) > 0" for i in range(len(wanted))) + ")")
    rows = cohort._query(client, sql, [(f"q{i}", "STRING", c.question_name.lower()) for i, c in enumerate(wanted)])
    names = dict(zip((int(i) for i in rows["concept_id"]), (str(n) for n in rows["concept_name"])))
    out, problems = [], {}
    for c in covariates:
        if c in wanted:
            hits = sorted(i for i, n in names.items() if c.question_name.lower() in n.lower())
            if len(hits) != 1:
                problems[c.name] = (f"no survey question in this CDR has '{c.question_name}' in its name" if not hits else
                                    f"'{c.question_name}' matches {len(hits)} questions: "
                                    + "; ".join(names[i] for i in hits[:5]) + " (make question_name more specific)")
                continue
            c = replace(c, concept_ids=hits, expect_name=c.expect_name or c.question_name, include_descendants=False)
        out.append(c)
    return out, problems


def survey_answers(covariate: Any, cdr: str, client: Any) -> Any:
    """The answers given to a resolved survey question, with how each is coded and how many people gave
    it (1-20 masked): what the flag means, for review before it is used."""
    import re

    cdr = resolve_cdr(cdr)
    sql = (f"SELECT a.concept_name AS answer, COUNT(DISTINCT o.person_id) AS people FROM `{cdr}.observation` o\n"
           f"LEFT JOIN `{cdr}.concept` a ON a.concept_id = o.value_source_concept_id\n"
           "WHERE o.observation_source_concept_id IN UNNEST(@ids) GROUP BY a.concept_name ORDER BY people DESC")
    frame = cohort._query(client, sql, [("ids", "ARRAY<INT64>", list(covariate.concept_ids))])
    def coded(answer: Any) -> str:
        text = str(answer or "").lower()
        if not text or re.search(covariate.missing_answers.lower(), text):
            return "missing"
        return "1" if re.search(covariate.positive_answers.lower(), text) else "0"
    frame["coded as"] = [coded(a) for a in frame["answer"]]
    frame["people"] = [MASK if 1 <= int(n) <= SMALL_CELL_MAX else int(n) for n in frame["people"]]
    return frame


def concept_check_sql(outcomes: list[CatalogOutcome], cdr: str) -> tuple[str, list[cohort.QueryParam]]:
    ids = sorted({cid for o in outcomes for cid in o.concept_ids})
    sql = (f"SELECT concept_id, concept_name, standard_concept, domain_id FROM `{resolve_cdr(cdr)}.concept`\n"
           "WHERE concept_id IN UNNEST(@ids)")
    return sql, [("ids", "ARRAY<INT64>", ids)]


def check_concepts(outcomes: list[CatalogOutcome], rows: list[dict[str, Any]]) -> dict[str, str | None]:
    """Map each outcome name to None (verified) or the reason it fails."""
    by_id = {int(r["concept_id"]): r for r in rows}
    result: dict[str, str | None] = {}
    for o in outcomes:
        problem = None
        if not o.concept_ids and getattr(o, "codes", None):
            result[o.name] = f"codes {', '.join(o.codes)} not found in this CDR's vocabulary"
            continue
        for cid in o.concept_ids:
            row = by_id.get(cid)
            if row is None:
                problem = f"concept {cid} not in the CDR"
            elif str(row.get("domain_id")) != _DOMAIN[o.kind]:
                problem = f"concept {cid} is in domain {row.get('domain_id')}, not {_DOMAIN[o.kind]}"
            elif o.expect_name.lower() not in str(row.get("concept_name", "")).lower():
                problem = f"concept {cid} is '{row.get('concept_name')}', expected a name with '{o.expect_name}'"
            if problem:
                break
        result[o.name] = problem
    return result


def verify_catalog(outcomes: list[Any], cdr: str, client: Any) -> dict[str, str | None]:
    if not outcomes:
        return {}
    frame = cohort._query(client, *concept_check_sql(outcomes, cdr))
    return check_concepts(outcomes, frame.to_dict(orient="records"))


def verify_covariates(base: CohortSpec, cdr: str, client: Any, log: Any = None) -> tuple[CohortSpec, dict[str, str]]:
    """``base`` with only the covariates and negative controls whose concepts pass the catalog's concept
    check, and the reason each dropped one failed (negative controls are keyed ``negative control <name>``)."""
    if not base.covariates and not base.negative_controls:
        return base, {}
    covariates, failed = resolve_survey_questions(resolve_codes(base.covariates, cdr, client, log), cdr, client)
    negative_controls = resolve_codes(base.negative_controls, cdr, client, log)
    base = replace(base, covariates=covariates, negative_controls=negative_controls)
    failed.update({name: reason for name, reason in verify_catalog(base.covariates, cdr, client).items() if reason})
    for c in base.covariates:  # a survey flag needs an answer coded 1 and one coded 0
        if c.kind == "survey" and c.name not in failed:
            coded = set(survey_answers(c, cdr, client)["coded as"])
            if not {"0", "1"} <= coded:
                failed[c.name] = (f"answers to '{c.question_name}' are coded only as {sorted(coded)}; adjust "
                                  "positive_answers so some answers are 1 and some 0")
    bad_nc = {f"negative control {name}": reason
              for name, reason in verify_catalog(base.negative_controls, cdr, client).items() if reason}
    spec = replace(base, wearables=list(base.wearables), covariates=[c for c in base.covariates if c.name not in failed],
                   negative_controls=[c for c in base.negative_controls if f"negative control {c.name}" not in bad_nc])
    return _keep_positive_controls(spec), {**failed, **bad_nc}


def feasibility_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[cohort.QueryParam]]:
    """Count participants (and cases) meeting the extractor's wearable-window rule for ``spec``.

    Uses the extractor's own SQL with only the primary wearable's first column, so the counts match
    what an extraction would produce without computing every feature.
    """
    primary = spec.wearables[0]
    keep = [primary.columns[0]] + ([spec.valid_day["column"]] if spec.valid_day else [])
    slim = replace(
        spec, wearables=[WearableSource(table=primary.table, date_column=primary.date_column,
                                        columns=list(dict.fromkeys(keep)), prefix=primary.prefix,
                                        filter_column=primary.filter_column, filter_value=primary.filter_value)],
        include_demographics=False, covariates=[], holdout_fraction=0.0, negative_controls=[], positive_controls=[],
        phenotypes=[], panel=None,
    )
    inner, params = build_cohort_sql(slim, cdr)
    sql = (f"SELECT COUNT(*) AS n_participants, SUM({spec.outcome.name}) AS outcome_sum\n"
           f"FROM (\n{inner}\n) AS cohort")
    return sql, params


def feasibility(outcomes: list[CatalogOutcome], base: CohortSpec, cdr: str, client: Any) -> list[dict[str, Any]]:
    """Participant (and case/control) counts for each outcome. Raw integers, for ranking only."""
    rows = []
    for o in outcomes:
        frame = cohort._query(client, *feasibility_sql(spec_for(o, base), cdr))
        n = int(frame.iloc[0]["n_participants"] or 0)
        row: dict[str, Any] = {"name": o.name, "label": o.label, "kind": o.kind, "n_participants": n}
        if o.kind == "condition":
            cases = int(frame.iloc[0]["outcome_sum"] or 0)
            row.update(n_cases=cases, n_controls=n - cases)
        rows.append(row)
    return rows


def _effective(row: dict[str, Any]) -> int:
    if row["kind"] == "condition":
        return min(row.get("n_cases", 0), row.get("n_controls", 0))
    return row["n_participants"]


def rank_outcomes(
    rows: list[dict[str, Any]],
    *,
    top_k: int = 3,
    min_measurement_n: int = MIN_MEASUREMENT_N,
    min_cases: int = MIN_CASES,
    per_kind: bool = False,
) -> list[dict[str, Any]]:
    """Outcomes clearing the minimum sample, largest effective sample first (ties by name).

    With ``per_kind`` the top K is taken separately for each outcome kind, so outcomes of one kind
    (whose effective sample is counted differently) cannot crowd out the others."""
    eligible = [r for r in rows
                if (r["kind"] == "measurement" and r["n_participants"] >= min_measurement_n)
                or (r["kind"] == "condition" and min(r.get("n_cases", 0), r.get("n_controls", 0)) >= min_cases)]
    ranked = sorted(eligible, key=lambda r: (-_effective(r), r["name"]))
    if not per_kind:
        return ranked[:top_k]
    kinds = list(dict.fromkeys(r["kind"] for r in ranked))
    return [r for kind in kinds for r in [x for x in ranked if x["kind"] == kind][:top_k]]


def _show(n: Any) -> str:
    if not isinstance(n, int):
        return "-"
    return MASK if 1 <= n <= SMALL_CELL_MAX else f"{n:,}"


def feasibility_table(rows: list[dict[str, Any]], skipped: dict[str, str | None] | None = None,
                      chosen: list[dict[str, Any]] | None = None, opened: set[str] | None = None) -> str:
    """A Markdown table of the screen with counts of 1-20 masked; safe to display. ``opened`` names outcomes
    already tested on this holdout."""
    picked = {r["name"] for r in chosen or []}
    lines = ["| outcome | kind | participants | cases | controls | selected |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        selected = ("yes" if r["name"] in picked else
                    "already tested on this holdout" if r["name"] in (opened or set()) else "")
        lines.append(f"| {r['label']} | {r['kind']} | {_show(r['n_participants'])} | {_show(r.get('n_cases'))} | "
                     f"{_show(r.get('n_controls'))} | {selected} |")
    for name, reason in (skipped or {}).items():
        if reason:
            lines.append(f"| {name} | skipped | - | - | - | {reason} |")
    return "\n".join(lines)
